# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


class TokenizerAdapter:
    def encode(self, text: str) -> list[int]:
        raise NotImplementedError

    def decode(self, token_ids: list[int]) -> str:
        raise NotImplementedError

    @property
    def bos_token_id(self) -> int | None:
        return None

    @property
    def eos_token_id(self) -> int | None:
        return None

    @property
    def pad_token_id(self) -> int | None:
        return None


@dataclass
class TransformersTokenizerAdapter(TokenizerAdapter):
    tokenizer: object

    @classmethod
    def from_pretrained(cls, model_dir: str, trust_remote_code: bool = False) -> "TransformersTokenizerAdapter":
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "transformers is required for the current local Hugging Face tokenizer adapter."
            ) from exc

        tokenizer = AutoTokenizer.from_pretrained(
            str(Path(model_dir)),
            local_files_only=True,
            trust_remote_code=trust_remote_code,
            use_fast=True,
        )
        return cls(tokenizer=tokenizer)

    def encode(self, text: str) -> list[int]:
        return list(self.tokenizer.encode(text, add_special_tokens=False))

    def decode(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)

    @property
    def bos_token_id(self) -> int | None:
        return self.tokenizer.bos_token_id

    @property
    def eos_token_id(self) -> int | None:
        return self.tokenizer.eos_token_id

    @property
    def pad_token_id(self) -> int | None:
        return self.tokenizer.pad_token_id
