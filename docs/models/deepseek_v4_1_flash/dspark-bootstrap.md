# DSpark terminal-prefill bootstrap composition

The terminal prefill handoff previously submitted five chip programs per rank.
They resolved the first token, prepared drafter geometry, populated draft KV and
ran the three draft layers, sampled proposals, and published persistent request
state. Each program boundary exposed intermediate buffers to the host runtime.

`prefill_bootstrap_l2.py` provides `l3_prefill_bootstrap_dspark`, which submits
one `l2_prefill_bootstrap_dspark` per rank. The L2 composes the existing bodies
in the same order. The state and geometry modules export inline and standalone
forms of shared implementations, as the drafter already does. Markov output
uses that same pattern. Attention, MoE, projection, sampling, and their
quantization boundaries retain their existing implementations.

The caller supplies completed target hidden taps, draft weights and KV,
request slots and generations, page and RoPE tables, sampling scratch, and
communication epochs. `state_meta`, `state_tokens`, and draft KV are mutable.
The host weight annotation retains the existing logical ND contract, while
the rank-local LM head has its required NZ layout. The L3 owns communication
window allocation and rank placement. Request preparation remains in Serving.

## Validation scope

This change was developed against official Lib main `873c2452` and official
Serving main `ffab1ab0`. The latter pins Lib `3dc359fe`, which predates the
DSpark interface migration to `dspark_init`, `decode_metadata`, and `markov`.
Serving main still imports the removed state and Markov modules. Combining
the two latest main branches requires additional Serving ABI integration.

On four A5 devices, the generated host loop submitted one chip program per
rank. The five-stage control and fused program produced identical state,
proposals, confidence values and draft KV across six replays each. The
fixture used synthetic weights, 64 prompt tokens on one active request,
12 slots per rank and four local experts per rank. After two warmups per
variant, median bootstrap wall time fell from 19.92 ms to 12.93 ms across
four measurements per variant. These measurements use Lib base 873c2452,
PyPTO 3db18db1, Simpler 355ce301, PTOAS 0.72 and CANN 9.3. Full-model
TTFT and TPOT require separate streaming measurements.
