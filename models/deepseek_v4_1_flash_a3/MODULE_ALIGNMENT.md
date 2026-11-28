# A3 module ownership

## Default distributed forward

| Responsibility | Owner |
| --- | --- |
| Prefill rank loop, layer schedule, six attention modes and round-based MoE | `prefill_fwd.py` |
| Decode rank loop, encoder and decoder layer schedules | `decode_fwd.py` |
| Phase-specific attention and indexing | `prefill_*.py`, `decode_*.py` |
| Routing, expert dispatch and combine, point-to-point signaling | `moe.py`, `expert_routed.py`, `expert_shared.py` |
| Topology, phase geometry, checkpoint loading and resident weight layouts | `config.py`, `weights.py` |
| Prepared tensor inputs, cache geometry and Decode aliases | `host.py`, `utils.py` |
| Compile, verify manifests and select execution | `driver.py`, `generate.py` |
| Weight and cache lifetime, request submission and sampling | `generation.py`, `replay.py` |
| Layer and logit comparison, routing reports and profiling | `validation.py`, `reference.py`, `telemetry.py`, `swimlane.py` |

The forward entries import their kernels from this directory. W8A8 row quantization
is in `activation_quant.py` and cache encoding is in `quant.py`. Q and KV projections
use `blocked_rmsnorm.py`. The output projection uses inverse RoPE from `rope.py`.
The bundled 8K request is `input_ids.json`.
