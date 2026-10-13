# DeepSeek V4.1 Flash vision encoder and visual aligner

[`vision.py`](../../../models/deepseek_v4_1_flash/vision.py) implements the BF16
ViT and visual aligner for A5, following the
[official vision implementation](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/2cba9e42aa026125f3ed06c6d98c1db82f7ca027/inference/vision.py).
The released configuration uses 32 blocks,
1,024 channels, 16 attention heads, a 2,816-channel SwiGLU intermediate, and
14-by-14 RGB patches. Each block applies RMSNorm, full bidirectional attention
with split-half 2D RoPE, and a SwiGLU MLP with residual additions. The final
RMSNorm produces the image features. The aligner zero-pads the grid, packs
3-by-3 neighborhoods in channel-major order, and applies two biased projections
with GELU to produce 5,120-channel image embeddings.

`VisionEncoder` accepts caller-provided CPU BF16 patches, input-major contiguous
weights, and the patch-grid height and width. Patches may be flattened or shaped
`[height * width, 3, 14, 14]`, with each patch flattened in channel, height, width
order. `load_vision_weights` loads only the vision and aligner tensors from the
released sharded safetensors checkpoint. RMSNorm weights are FP32. One invocation
processes one image; reuse the encoder object to retain its JIT cache.
Configurable profiles require a channel width divisible by 128 and a head
width divisible by 32, up to 128 channels per head.

Full native execution needs the deferred FIFO retirement fix in
[PTO ISA MR 1834](https://gitcode.com/cann/pto-isa/merge_requests/1834).
Use a PyPTO toolchain containing that fix.

For the released profile, the returned tensors are `[height * width, 1024]`
features and `[ceil(height / 3) * ceil(width / 3), 5120]` aligned embeddings.
Other profiles use their `VisionConfig.dim`, `downsample`, and `language_dim`
values for these dimensions. The caller owns
image resizing and normalization, image delimiters and newline embeddings, and
placement of the aligned rows in the language-model input. The ViT's 2D RoPE
tables are distinct from text-model RoPE tables.

The standalone `main()` validates both device outputs against an independent
Torch reference with FP32 math SDPA. The released 32-layer profile requires
finite outputs, feature relative L2 at most 5%, embedding relative L2 at most
3%, and cosine similarity at least 0.999 for both outputs. Worst-row relative
L2 is reported separately: the checkpoint can produce nearly zero feature
rows. Reduced fixtures retain 2% global and 5% worst-row relative L2 gates.
These are engineering thresholds, not official DeepSeek accuracy standards.
It defaults to a reduced two-layer fixture; `--checkpoint`
selects the released dimensions and real weights. `--patches` supplies a tensor
of preprocessed image patches and `--grid` declares its grid. Use an allocated
device for execution. `--compile-only` generates an artifact without device
validation; `--runtime-dir` reuses that artifact, and `--golden-data` replays
frozen input and expected-output tensors. `--ring-heap` overrides temporary
storage per runtime ring; the callable API accepts the same `ring_heap` setting
through `run_config`.
The largest validated grids use 16 GiB per runtime ring.
