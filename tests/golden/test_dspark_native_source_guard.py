# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""The native W2 selector must reject unreviewed Cube or shared ABI changes."""
import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


path = Path(__file__).resolve().parents[2] / 'models/deepseek_v4_flash_lowlat/native_source.py'
spec = importlib.util.spec_from_file_location('dspark_native_source_guard_test', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class NativeSourceGuardTests(unittest.TestCase):
    def test_only_reviewed_cube_and_shared_declarations_are_replaced(self):
        source = ('int shared;\n  #if defined(__DAV_CUBE__)\n  old_cube();\n'
                  '  #endif // __DAV_CUBE__\n  #if defined(__DAV_VEC__)\n  vector_body();\n'
                  '  #endif // __DAV_VEC__\n')
        fragment = '  #if defined(__DAV_CUBE__)\n  reviewed_cube();\n  #endif // __DAV_CUBE__\n'
        digest = hashlib.sha256(module.compiled_cube_cpp(source).encode()).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            native = Path(directory) / 'native.inc'
            native.write_text(fragment)
            with patch.object(module, 'NATIVE_SOURCE', native), \
                 patch.object(module, 'GENERATED_CUBE_DIGESTS', frozenset({digest})), \
                 patch.object(module, 'fingerprint', return_value='approved'):
                replaced = module.replace_cube_source(source, 'approved')
                self.assertIn('reviewed_cube();', replaced)
                self.assertIn('vector_body();', replaced)
                self.assertNotIn('old_cube();', replaced)
                with self.assertRaisesRegex(ValueError, 'Unrecognized'):
                    module.replace_cube_source(source.replace('old_cube();', 'different_cube();'), 'approved')
                with self.assertRaisesRegex(ValueError, 'Unrecognized'):
                    module.replace_cube_source(source.replace('int shared;', 'long shared;'), 'approved')
                with self.assertRaisesRegex(ValueError, 'fingerprint'):
                    module.replace_cube_source(source, 'different')


if __name__ == '__main__':
    unittest.main()
