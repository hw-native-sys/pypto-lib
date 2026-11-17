"""Frozen structural-gate controls (experiment artifact, not model code)."""


def mla_proj(x):
    return x


def attention_core(q):
    return q


def moe_gate(x):
    return x


def expert_routed(x):
    return x


def layer_fn(x):
    mla_proj(x)
    attention_core(x)
    moe_gate(x)
    expert_routed(x)
    return x


def tile_entry(x, blocks):
    for _block in range(blocks):
        x = layer_fn(x)
    return x


def run_stack(x, layers):
    for _layer in range(layers):
        x = layer_fn(x)
    return x


def tokenwise_entry(x, tokens, layers):
    for _token in range(tokens):
        x = run_stack(x, layers)
    return x


def renamed_layer(x):
    mla_proj(x)
    attention_core(x)
    moe_gate(x)
    expert_routed(x)
    return x


def renamed_stack(x, layers):
    for _j in range(layers):
        x = renamed_layer(x)
    return x


def renamed_entry(x, tokens):
    for _i in range(tokens):
        x = renamed_stack(x, 40)
    return x


# Complete-model-stack controls (embedding -> 40 layers -> head). The tile entry
# batches the layer stack once; the tokenwise entry wraps the complete stack in
# a prompt-token loop and must be rejected; the renamed wrapper must not evade
# the check.
def embed(x):
    return x


def lm_head(x):
    return x


def complete_layer(x):
    mla_proj(x)
    attention_core(x)
    moe_gate(x)
    expert_routed(x)
    return x


def complete_rank(x, layers=40):
    x = embed(x)
    for _layer in range(layers):
        x = complete_layer(x)
    return lm_head(x)


def complete_tile_entry(x):
    return complete_rank(x)


def tokenwise_complete_entry(x, tokens):
    for _token in range(tokens):
        x = complete_rank(x)
    return x


def renamed_complete_rank(x):
    return complete_rank(x)


def renamed_complete_entry(x, tokens):
    for _t in range(tokens):
        x = renamed_complete_rank(x)
    return x


def renamed_tile_entry(x):
    return renamed_complete_rank(x)
