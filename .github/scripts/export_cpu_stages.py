# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Capture unchanged CPU golden stages on a byte-bound, previously exported fixture."""

import argparse
from contextlib import ExitStack
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from unittest.mock import patch


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    assert not path.exists(), path.name
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--entry', choices=('c1a', 'c2a'), required=True)
p.add_argument('--fixtures', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--pypto-source', type=Path, required=True)
p.add_argument('--expected-pypto', choices=(
    'afe81773c33ef041d97f81e473c06618b961ecda', 'c42e22e9210f5d92a6fcad36d1c8c6e149745073'), required=True)
p.add_argument('--expected-machine', choices=('x86_64', 'aarch64'), required=True)
args = p.parse_args()
repo = Path(__file__).resolve().parents[2]
model = repo/'models/deepseek_v4_1_flash'
output = args.output.resolve()
assert not os.environ.get('TASK_DEVICE') and not output.exists()
assert platform.machine() == args.expected_machine
output.mkdir(parents=True)
contract_path = Path(__file__).with_name('cpu_stage_contract.json')
contract = read(contract_path)
entry = contract['entries'][args.entry]
fixture_dir = args.fixtures.resolve()/args.entry
case_dir = fixture_dir/entry['directory']
assert sha(fixture_dir/'host.json') == entry['host_sha256']
assert sha(fixture_dir/'summary.json') == entry['summary_sha256']
assert sha(case_dir/'manifest.json') == entry['manifest_sha256']
export_host, fixture = read(fixture_dir/'host.json'), read(case_dir/'manifest.json')
assert fixture['rows'] == 72 and fixture['cpu_reference_self_check_passed']
source_map = export_host['source_sha256']
assert hashlib.sha256(json.dumps(source_map, sort_keys=True, separators=(',', ':')).encode()).hexdigest() == contract['source_digest']
assert len(source_map) == 246


def source_identity():
    return {name: sha(repo/name) for name in source_map}


assert source_identity() == source_map
pypto_source = args.pypto_source.resolve()
git = lambda where: subprocess.check_output(['git', '-C', str(where), 'rev-parse', 'HEAD'], text=True).strip()
assert git(pypto_source) == args.expected_pypto
assert git(pypto_source/'runtime') == 'd8f997fabd1b6988b69a94feef61ab669dda0eea'
assert (pypto_source/'runtime/pto_isa.pin').read_text().strip() == 'b1d460ee8c001585ede36b68ba3678dc5a226bbf'
filename = f'decode_{args.entry}_full.py'
sys.argv = [str(model/filename), '-p', 'a5', '-d', '0', '--compile-only']
sys.path[:0] = [str(model), str(repo)]
sys.dont_write_bytecode = True
os.environ['PYPTO_BENCH'] = '0'

import torch
import golden
from golden import runner
from pypto.ir import CompiledProgram, DistributedCompiledProgram
from pypto.jit.decorator import JITFunction
from pypto.runtime import device_runner
from pypto.runtime.worker import ChipWorker
from simpler_setup.runtime_compiler import RuntimeCompiler
import pypto.ir

assert torch.__version__.split('+')[0] == '2.10.0'
torch.set_num_threads(16)
module_hashes = {name: sha(importlib.import_module(name).__file__)
                 for name in ('pypto', 'pypto.pypto_core', 'pypto.jit.decorator', 'simpler', 'torch._C')}
for relative in ('__init__.py', 'jit/decorator.py'):
    assert sha(Path(importlib.import_module('pypto').__file__).parent/relative) == sha(pypto_source/'python/pypto'/relative)


def raw_sha(value):
    return hashlib.sha256(memoryview(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy())).hexdigest()


def tensor_record(value):
    return dict(shape=list(value.shape), dtype=str(value.dtype), bytes=value.numel()*value.element_size(), raw_sha256=raw_sha(value))


def metadata():
    result = dict(machine=platform.machine(), python=sys.version, torch=torch.__version__,
        cpu_capability=torch.backends.cpu.get_cpu_capability(), torch_config=torch.__config__.show(),
        threads=torch.get_num_threads(), interop_threads=torch.get_num_interop_threads(),
        float32_matmul_precision=torch.get_float32_matmul_precision(),
        mkldnn_enabled=torch.backends.mkldnn.enabled, mkldnn_available=torch.backends.mkldnn.is_available(),
        mkldnn_deterministic=torch.backends.mkldnn.deterministic,
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        thread_environment={k: os.environ.get(k) for k in (
            'PYPTO_GOLDEN_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
            'ATEN_CPU_CAPABILITY', 'OMP_DYNAMIC', 'MKL_DYNAMIC')},
        pypto_commit=args.expected_pypto, module_sha256=module_hashes,
        driver_sha256=sha(__file__), contract_sha256=sha(contract_path),
        original_export_run=contract['run_id'], original_export_head=contract['head_sha'],
        original_manifest_sha256=entry['manifest_sha256'])
    for label, owner in [('backend', torch.backends), ('mkldnn', torch.backends.mkldnn)]:
        try:
            result[label+'_fp32_precision'] = owner.fp32_precision
        except (AttributeError, RuntimeError):
            result[label+'_fp32_precision'] = 'unavailable'
    return result


initial = metadata()
write(output/'host.json', initial)
input_values = {}
expected_values = {}
bound_fields = {}
specs = {s['name']: s for s in fixture['specs']}
for phase, fields in fixture['fields'].items():
    for name, field in fields.items():
        path = case_dir/'data'/phase/(name+'.pt')
        assert sha(path) == field['file_sha256']
        value = torch.load(path, weights_only=True)
        assert tensor_record(value) == {k: field[k] for k in ('shape', 'dtype', 'bytes', 'raw_sha256')}
        (input_values if phase == 'in' else expected_values)[name] = value
        bound_fields[str(path.relative_to(case_dir))] = field['file_sha256']
assert set(input_values) == {s['name'] for s in fixture['specs'] if 'shape' not in s or s['direction'] in ('in','inout')}
assert set(expected_values) == {s['name'] for s in fixture['specs'] if s.get('direction') in ('out','inout')}
assert int(input_values['num_tokens']) == 72
forbidden = []
stages = {}
counts = {}
original_matmul = torch.matmul
full_name = f'golden_{args.entry}'


def forbid(*unused, **ignored):
    forbidden.append('compiler, process, or device boundary')
    raise AssertionError(forbidden[-1])


def save_stage(label, values):
    count = counts.get(label, 0)
    counts[label] = count + 1
    key = f'{label}-{count:02d}'
    assert key not in stages
    stages[key] = {name: value.detach().clone() for name, value in values.items() if isinstance(value, torch.Tensor)}


def trace(frame, event, result):
    name = frame.f_code.co_name
    if Path(frame.f_code.co_filename).parent != model:
        return None
    caller = frame.f_back.f_code.co_name if frame.f_back else ''
    values = frame.f_locals
    publication = caller == '_publish_quantized_cache' or (
        frame.f_back and frame.f_back.f_back and frame.f_back.f_back.f_code.co_name == '_publish_quantized_cache')
    allowed = name in ('golden_hc_pre', 'golden_compressor_ratio1', 'golden_compressor_ratio2_paged',
                       'golden_index_key', full_name)
    allowed = allowed or (name == 'quantize_mxfp4_cache' and publication)
    allowed = allowed or (name == 'golden_rope_tail' and caller == full_name)
    allowed = allowed or (name == 'golden_rms_norm' and (caller in (full_name, 'golden_compressor_ratio1', 'golden_index_key')
        or (caller == 'golden_compressor_ratio2_paged' and frame.f_back.f_locals.get('token') == 21)))
    if not allowed:
        return None
    if event == 'line' and name == 'golden_compressor_ratio2_paged':
        if frame.f_lineno == pooled_line and values.get('token') == 21:
            save_stage('c2-pair21', {k: values[k] for k in ('pair_kv','pair_score','pooled')})
    if event == 'return':
        selected = {
            'golden_hc_pre': ('products','first_pair','second_pair'),
            'golden_rms_norm': ('x','weight','value'),
            'golden_compressor_ratio1': ('x','projected'),
            'golden_compressor_ratio2_paged': ('x','kv','score','latent','publish'),
            'golden_index_key': ('latent','projected','key'),
            'golden_rope_tail': ('x',),
            'quantize_mxfp4_cache': ('value','grouped','amax','raw_scale','stored_scale','scale_value','normalized','payload'),
            full_name: ('attention_input','x','latent','key','rotated','index_projection','index_key','compressed_slots'),
        }[name]
        data = {k: values[k] for k in selected if k in values}
        if isinstance(result, torch.Tensor):
            data['result'] = result
        if name == 'quantize_mxfp4_cache':
            label = f'quant-{values["scale_format"]}-{values["group_size"]}'
        elif name == 'golden_rms_norm':
            label = name+'-'+caller
        else:
            label = name
        save_stage(label, data)
    return trace


def capture_matmul(*positional, **kwargs):
    result = original_matmul(*positional, **kwargs)
    frame = inspect.currentframe().f_back
    if Path(frame.f_code.co_filename).parent == model and frame.f_code.co_name in (
        'golden_compressor_ratio1','golden_compressor_ratio2_paged','golden_index_key'):
        save_stage('matmul-'+frame.f_code.co_name, {'result': result})
    return result


def scratch(poison_outputs):
    values = {}
    for name, spec in specs.items():
        if 'shape' not in spec:
            values[name] = input_values[name].item()
        elif spec['direction'] in ('in', 'inout'):
            values[name] = input_values[name].clone()
        else:
            dtype = getattr(torch, spec['dtype'].split('.')[-1])
            fill = float('nan') if dtype.is_floating_point else -197
            values[name] = torch.full(spec['shape'], fill if poison_outputs else 0, dtype=dtype)
    return values


error = None
completed = False
comparisons = {}
try:
    with ExitStack() as stack:
        targets = [(runner,'_dispatch'), (device_runner,'_execute_on_device'), (device_runner,'_compile_and_assemble'),
                   (ChipWorker,'__init__'), (CompiledProgram,'__call__'), (DistributedCompiledProgram,'__call__'),
                   (RuntimeCompiler,'compile'), (RuntimeCompiler,'compile_sim_context'), (JITFunction,'_compile'),
                   (pypto.ir,'compile'), (golden,'run'), (subprocess,'Popen'), (os,'system')]
        targets += [(device_runner.KernelCompiler,n) for n in dir(device_runner.KernelCompiler)
                    if n.startswith('compile') and callable(getattr(device_runner.KernelCompiler,n))]
        for owner, name in targets:
            stack.enter_context(patch.object(owner, name, forbid))
        module = importlib.import_module(f'decode_{args.entry}_full')
        fn = getattr(module, full_name)
        assert tuple(getattr(module, f'l3_decode_{args.entry}_full').param_names) == tuple(specs)
        compressor_lines = (model/'compressor.py').read_text().splitlines()
        matches = [i+1 for i,line in enumerate(compressor_lines) if 'latent[token] = golden_rms_norm(pooled.to(x.dtype), norm_weight)' in line]
        assert len(matches) == 1
        pooled_line = matches[0]
        baseline = scratch(False)
        fn(baseline)
        captured = scratch(True)
        with patch.object(torch, 'matmul', capture_matmul):
            assert sys.gettrace() is None
            sys.settrace(trace)
            try:
                fn(captured)
            finally:
                sys.settrace(None)
        for name, expected in expected_values.items():
            actual, normal = captured[name], baseline[name]
            assert actual.shape == normal.shape == expected.shape and actual.dtype == normal.dtype == expected.dtype
            assert raw_sha(actual) == raw_sha(normal), ('trace or pure-output initialization changed result',name)
            mismatch = actual.contiguous().reshape(-1).view(torch.uint8) != expected.contiguous().reshape(-1).view(torch.uint8)
            comparisons[name] = dict(traced_vs_untraced_exact=True, record=tensor_record(actual),
                exported_raw_sha256=raw_sha(expected), different_bytes_vs_export=int(mismatch.sum()))
        assert counts.get('golden_hc_pre') == 1 and counts.get(full_name) == 1
        assert counts.get('quant-bf16-16') == 1 and counts.get('quant-e8m0-32') == 1
        if args.entry == 'c2a':
            assert counts.get('c2-pair21') == 1
        stage_records = {}
        stage_dir = output/'stages'
        stage_dir.mkdir()
        for label, values in stages.items():
            stage_records[label] = {}
            for name, value in values.items():
                path = stage_dir/f'{label}--{name}.pt'
                torch.save(value, path)
                stage_records[label][name] = dict(tensor_record(value), file_sha256=sha(path), file=path.name)
        write(output/'stages.json', stage_records)
        write(output/'outputs.json', comparisons)
        completed = True
except BaseException as exc:
    error = type(exc).__name__+': '+str(exc).replace(str(repo),'<repo>').replace(str(output),'<output>').replace(str(args.fixtures.resolve()),'<fixtures>')
    raise
finally:
    post = metadata()
    unchanged = source_identity() == source_map and initial == post
    unchanged = unchanged and all(sha(case_dir/relative) == digest for relative,digest in bound_fields.items())
    write(output/'summary.json', dict(passed=completed and unchanged and not forbidden, error=error,
        source_inputs_and_settings_unchanged=unchanged, forbidden_calls=forbidden,
        captured_stages=counts, host_sha256=sha(output/'host.json'),
        scope='Full72 unchanged canonical CPU golden on exact prior x86 inputs; all original final outputs compared to untraced execution with independent pure-output initialization. No reference replacement, callback-gate change, model compilation, or native execution.'))
    assert unchanged and not forbidden
