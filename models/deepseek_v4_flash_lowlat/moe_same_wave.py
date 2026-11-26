# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Epilogues to call inside the original 24-cluster expert wave and after TP."""

import pypto.language as pl
import pypto.language.distributed as pld

from .config import FLASH as M
from .decode_shapes import MOE_TOKENS as T, TP

D = M.hidden_size
HC = M.hc_mult
BAND = 1024
TOKEN_TILE = 3
SORT_WAVES = T
ROUTE_READY_ROW = 0
GROUP_READY_ROW = 1
LOCAL_SIGNAL_ROWS = 2
LOCAL_SIGNAL_COLS = 32
LANE = TP * ((T + 15) // 16 * 16)
TP_ROWS = LANE // TP


@pl.jit.inline(auto_scope=False)
def publish_sort_row(
    arrived: pld.DistributedTensor[[LOCAL_SIGNAL_ROWS, LOCAL_SIGNAL_COLS], pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
):
    pl.system.cacheinvalid()
    pl.system.fence()
    pld.system.notify(target=arrived, peer=my_rank, offsets=[ROUTE_READY_ROW, 0], value=1,
                      op=pld.NotifyOp.AtomicAdd)


@pl.jit.inline(auto_scope=False)
def wait_for_sort_rows(
    arrived: pld.DistributedTensor[[LOCAL_SIGNAL_ROWS, LOCAL_SIGNAL_COLS], pl.INT32],
    epoch: pl.Scalar[pl.INT32],
):
    pld.system.wait(signal=arrived, offsets=[ROUTE_READY_ROW, 0],
                    expected=epoch*SORT_WAVES, cmp=pld.WaitCmp.Ge)
    pl.system.cacheinvalid()
    pl.system.fence()




@pl.jit.inline
def reduce_and_hc(
    window: pld.DistributedTensor[[2*LANE,D],pl.FP32],
    residual: pl.Tensor[[T,HC,D],pl.FP32],
    post: pl.Tensor[[T,HC],pl.FP32],
    comb: pl.Tensor[[T,HC*HC],pl.FP32],
    output: pl.Tensor[[T,HC,D],pl.FP32],
    lane_base: pl.Scalar[pl.INDEX],
    barrier_tid: pl.Scalar[pl.TASK_ID],
):
    """Sum ranks in order and retain the original BF16 compatibility round."""
    src=pl.reshape(residual,[T*HC,D])
    dst=pl.reshape(output,[T*HC,D])
    post_row=pl.reshape(post,[1,T*HC])
    comb_row=pl.reshape(comb,[1,T*HC*HC])
    with pl.spmd((T//TOKEN_TILE)*(D//BAND),name_hint="moe_reduce_and_hc_post",
                 deps=[barrier_tid],allow_early_resolve=True):
        block=pl.tile.get_block_idx()
        t0=block//(D//BAND)*TOKEN_TILE
        d0=block%(D//BAND)*BAND
        acc=pl.load(window,[lane_base+t0,d0],[TOKEN_TILE,BAND])
        for rank in pl.range(1,TP):
            part=pl.load(window,[lane_base+rank*TP_ROWS+t0,d0],[TOKEN_TILE,BAND])
            acc=pl.add(acc,part)
        narrowed=pl.cast(acc,target_type=pl.BF16,mode="rint")
        reduced=pl.cast(narrowed,target_type=pl.FP32)
        post_panel=pl.load(post_row,[0,0],[1,T*HC])
        comb_panel=pl.load(comb_row,[0,0],[1,T*HC*HC])
        for row in pl.pipeline(TOKEN_TILE,stage=2):
            token=t0+row
            vec=pl.tile.slice(reduced,[1,BAND],[row,0])
            # One strided panel load, re-used for all four output channels.
            residual_panel=pl.load(src,[token*HC,d0],[HC,BAND])
            out_panel=pl.create_tile([HC,BAND],dtype=pl.FP32)
            for h in pl.unroll(HC):
                post_w=pl.tile.read(post_panel,[0,token*HC+h])
                mixed=pl.mul(vec,post_w)
                for source in pl.unroll(HC):
                    before=pl.tile.slice(residual_panel,[1,BAND],[source,0])
                    comb_w=pl.tile.read(comb_panel,[0,token*HC*HC+source*HC+h])
                    contribution=pl.mul(before,comb_w)
                    mixed=pl.add(mixed,contribution)
                out_panel[h:h+1,:]=mixed
            pl.store(out_panel,[token*HC,d0],dst)


@pl.jit.inline(auto_scope=False)
def publish_group(
    arrived: pld.DistributedTensor[[LOCAL_SIGNAL_ROWS, LOCAL_SIGNAL_COLS], pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
):
    pl.system.cacheinvalid()
    pl.system.fence()
    pld.system.notify(target=arrived, peer=my_rank, offsets=[GROUP_READY_ROW, 0], value=1,
                      op=pld.NotifyOp.AtomicAdd)


@pl.jit.inline(auto_scope=False)
def wait_for_group(
    arrived: pld.DistributedTensor[[LOCAL_SIGNAL_ROWS, LOCAL_SIGNAL_COLS], pl.INT32],
    epoch: pl.Scalar[pl.INT32],
):
    pld.system.wait(signal=arrived, offsets=[GROUP_READY_ROW, 0],
                    expected=epoch, cmp=pld.WaitCmp.Ge)
    pl.system.cacheinvalid()
    pl.system.fence()
