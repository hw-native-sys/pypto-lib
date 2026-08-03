# PyPTO Coding

Use this chapter when writing or reviewing PyPTO-Lib kernels.

- [PyPTO Coding Style](pypto-coding-style.md) is the canonical reference for
  kernel forms, scopes, loops, tensor operations, tiling, naming, and source
  layout.
- [CCE Extern Kernel](cce-extern-kernel.md) covers hand-written mixed cube and
  vector kernels called through `pl.jit.extern`, including runtime, ABI,
  synchronization, and validation constraints.

Start with the coding style for every kernel change. Use the extern-kernel page
only when the implementation crosses from PyPTO into hand-written CCE code.
