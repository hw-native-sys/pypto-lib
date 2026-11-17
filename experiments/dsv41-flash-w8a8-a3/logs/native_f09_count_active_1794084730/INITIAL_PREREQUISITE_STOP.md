# Prerequisite STOP before CPU prequeue or model imports

Root supplied lifecycle acceptance receipt1794084680 SHA b15fb649104e6bfffbf255ffadb8751dd485fa93126a80ec05963d57f2c3c9d6 uses key `oracle_helper_sha256` with the expected e45c7529...999d3 digest. The frozen83 route12872cd0...7bf8 requires `oracle_integrity_sha256` in prerequisites(). The exact supplied receipt therefore fails the canonical route's prerequisite check even though the intended digest matches.

Reported directly to Root and Main immediately. No model import, CPU queue task, fixture/tensor/header/prepare/compiler/NPU call, positive stage receipt or protected-source edit occurred. Root owns a corrected NEW acceptance receipt; preserve this original receipt and all frozen83 source bytes. This note is a precise input-schema mismatch, not an upstream compiler/runtime bug.
