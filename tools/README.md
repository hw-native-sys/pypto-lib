# pypto-lib tools

## export_all_kernel_insight.py

Exports MindStudio Insight / msprof op-simulator traces for every generated
PTOAS InCore kernel in a `build_output/<case>` directory.

Use it directly on an existing build:

```bash
python tools/export_all_kernel_insight.py --build-dir build_output/Qwen3Decode_<timestamp>
```

Or let the tool run the case first and export from the build it produces:

```bash
python tools/export_all_kernel_insight.py \
  --case models/qwen3/14b/decode_fwd.py \
  -- --max-seq --enable-l2-swimlane
```

Output is written under the selected build directory as
`kernel_insight_all_funcs_<timestamp>/`. The file
`latest_all_funcs_kernel_insight_export_root.txt` points to the newest export.
