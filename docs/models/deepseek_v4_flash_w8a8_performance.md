# DeepSeek V4-Flash W8A8 performance

This page is the canonical presentation reference for the maintained
DeepSeek V4-Flash W8A8 performance reports. It does not replace the immutable
raw measurements or the compatible previous-result baseline used for
regression tracking.

## Supplied snapshot

The supplied configuration is `TP=1, DP=EP=16, GBS=16x4, SeqLen=8K, MTP=1,
EPLB`. Lower latency is better.

| Operator | Supplied PyPTO | This test | AscendC | This test / AscendC |
| --- | ---: | ---: | ---: | ---: |
| Attention CSA | 357 us | - | 465 us | - |
| Attention HCA | 261 us | - | 307 us | - |
| Attention SWA | 243 us | - | 280 us | - |
| MoE | 477 us | - | 479 us | - |
| Decode Main | 36.1 ms | - | 37.7 ms | - |
| Decode MTP | 1,162 us | - | 1,268 us | - |

`This test` is populated from the current contract-valid measurement. The
ratio is calculated from the unrounded test value after unit conversion and
is rounded to three decimal places. A ratio below 1 means that the current
test is faster than AscendC.

The supplied snapshot does not identify its source revision, collection time,
exact command, or aggregation statistic. Its EP16/TP1 configuration also
differs from the maintained EPLB EP8 workload and its TP4 decode cases. Treat
these values as supplied presentation references, not as strict
same-configuration regression baselines.

## Maintained report format

Every reported result uses the same five columns shown above. Source commit,
source tree, contract, toolchain and device epochs, selected rank or device,
validation status, and collection status are recorded outside the table.

The maintained cases and their measurement contracts are defined in
[Performance Tracking](../debug-and-tune/performance-tracking.md). The tracker
pins this page and the `Supplied snapshot` section by content digest. A change
to this presentation reference creates a new reference snapshot; it does not
schedule an NPU rerun.
