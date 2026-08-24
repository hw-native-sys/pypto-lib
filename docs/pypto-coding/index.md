# PyPTO Coding

Use this chapter when writing or reviewing PyPTO-Lib kernels.

- [PyPTO Coding Style](pypto-coding-style.md) is the canonical reference for
  kernel forms, scopes, loops, tensor operations, tiling, naming, and source
  layout.
- [Distributed Programming](distributed-programming.md) covers multi-card (L3)
  kernels: the host driver, HCCL window buffers, cross-rank data movement, and
  the notify / wait protocols that order it.
- [CCE Extern Kernel](cce-extern-kernel.md) covers hand-written mixed cube and
  vector kernels called through `pl.jit.extern`, including runtime, ABI,
  synchronization, and validation constraints.

Start with the coding style for every kernel change. Add the distributed page
when the kernel spans more than one card. Use the extern-kernel page only when
the implementation crosses from PyPTO into hand-written CCE code.
