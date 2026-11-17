# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Pure-Python DeepSeek-V4.1 BPE tokenizer for the generation driver.

The checkpoint ships a Hugging Face ``tokenizers`` BPE definition
(``tokenizer.json``: 128000 base tokens, 127741 merges, 1283 added tokens),
but the model environment installs neither ``tokenizers``, ``transformers`` nor
``regex``.  This module reimplements only the part the host-side generation
driver needs, using the Python standard library:

* the four-stage ``pre_tokenizer`` sequence from ``tokenizer.json``: isolate
  1-3 digit numbers, isolate the shipped CJK ranges, apply the DeepSeek/GPT
  letter/punctuation/whitespace split, then apply the ByteLevel byte mapping;
* byte-level BPE with merges applied by rank, one pretoken at a time;
* all added tokens (special and non-special) matched literally and kept atomic,
  so a special token can never be split by pretokenization or BPE;
* ``encode(text) -> list[int]`` and ``decode(ids) -> str`` with no BOS/EOS
  insertion, mirroring ``add_bos_token=false`` / ``add_eos_token=false``.

The reference regex uses the Unicode classes ``\\p{L}``, ``\\p{M}``, ``\\p{P}``,
``\\p{S}`` and ``\\p{N}``.  Python's ``re`` does not support those classes, so
``_build_category_ranges`` builds explicit Unicode general-category ranges once from
``unicodedata``.  This is exact for the Unicode version shipped with the
running Python; codepoints assigned in a newer Unicode release than the
reference Rust regex engine (or than this Python) can be classified
differently.  The affected classes are ``L``, ``M``, ``P``, ``S`` and ``N``.

``encode_messages`` renders a deliberately small plain-text chat subset of
``<MODEL_ROOT>/encoding/encoding.py`` (system/user/assistant/latest-reminder
turns, ``thinking_mode="chat"``).  Tool calls, images, thinking mode, reasoning
effort, task tokens and context carry-over stay with the generation driver; the
shipped ``encoding.py`` remains the prompt-format reference.

Run ``python tokenizer.py`` for the bounded, hand-checkable validation entry.
"""

import argparse
import json
import os
import re
import sys
import unicodedata
from pathlib import Path

DEFAULT_MODEL_ROOT = "/mnt/data/models/DeepSeek-V4.1-Flash-w8a8"
TOKENIZER_FILE = "tokenizer.json"
BASE_VOCAB_SIZE = 128000
TOTAL_VOCAB_SIZE = 129280

# Special-token spellings used by the shipped prompt-format reference.
BOS_TOKEN = "<｜begin▁of▁sentence｜>"
EOS_TOKEN = "<｜end▁of▁sentence｜>"
# tokenizer.json added token id 2; tokenizer_config.json aliases pad_token to EOS.
PAD_TOKEN = "<｜▁pad▁｜>"
USER_TOKEN = "<｜User｜>"
ASSISTANT_TOKEN = "<｜Assistant｜>"
SYSTEM_TOKEN = "<｜System｜>"
LATEST_REMINDER_TOKEN = "<｜latest_reminder｜>"
THINK_START_TOKEN = "<think>"
THINK_END_TOKEN = "</think>"
IMAGE_PLACEHOLDER = "<｜deepseek_image｜>"

# Character classes used by the shipped pre_tokenizer Split sequence.
_CATEGORY_LETTERS = ("L", "M", "P", "S", "N")
_CJK_PATTERN = "[\\u4e00-\\u9fa5\\u3040-\\u309f\\u30a0-\\u30ff]+"


def _build_category_ranges():
    """Return contiguous codepoint runs for each needed Unicode category."""
    runs = {letter: [] for letter in _CATEGORY_LETTERS}
    open_run = {letter: None for letter in _CATEGORY_LETTERS}
    for codepoint in range(0x110000):
        category = unicodedata.category(chr(codepoint))[0]
        for letter in _CATEGORY_LETTERS:
            current = open_run[letter]
            if category == letter:
                if current is None:
                    open_run[letter] = [codepoint, codepoint]
                elif current[1] == codepoint - 1:
                    current[1] = codepoint
                else:
                    runs[letter].append((current[0], current[1]))
                    open_run[letter] = [codepoint, codepoint]
            elif current is not None:
                runs[letter].append((current[0], current[1]))
                open_run[letter] = None
    for letter in _CATEGORY_LETTERS:
        current = open_run[letter]
        if current is not None:
            runs[letter].append((current[0], current[1]))
    return runs


_CATEGORY_RANGES = _build_category_ranges()


def _class_body(letters):
    """Render the union of category runs as an ``re`` character-class body."""
    parts = []
    for letter in letters:
        for first, last in _CATEGORY_RANGES[letter]:
            if first == last:
                parts.append("\\U%08x" % first)
            else:
                parts.append("\\U%08x-\\U%08x" % (first, last))
    return "".join(parts)


def _compile_pretokenizer_patterns():
    """Compile the Split sequence exactly as expressed by tokenizer.json."""
    number_re = re.compile("[%s]{1,3}" % _class_body(("N",)))
    cjk_re = re.compile(_CJK_PATTERN)
    # The pattern below is tokenizer.json's third Split regex with the
    # \p{...} classes replaced by explicit general-category ranges.
    split_re = re.compile(
        "[!\"#$%&'()*+,\\-./:;<=>?@\\[\\\\\\]^_`{|}~][A-Za-z]+"
        f"|[^\\r\\n{_class_body(('L', 'P', 'S'))}]?[{_class_body(('L', 'M'))}]+"
        f"| ?[{_class_body(('P', 'S'))}]+[\\r\\n]*"
        "|\\s*[\\r\\n]+"
        "|\\s+(?!\\S)"
        "|\\s+"
    )
    return number_re, cjk_re, split_re


_NUMBER_RE, _CJK_RE, _SPLIT_RE = _compile_pretokenizer_patterns()


def _bytes_to_unicode():
    """Build the GPT-2/ByteLevel byte -> printable-unicode alphabet."""
    direct = list(range(ord("!"), ord("~") + 1))
    direct += list(range(ord("¡"), ord("¬") + 1))
    direct += list(range(ord("®"), ord("ÿ") + 1))
    codepoints = direct[:]
    extra = 0
    for byte in range(256):
        if byte not in direct:
            direct.append(byte)
            codepoints.append(256 + extra)
            extra += 1
    return {byte: chr(codepoint) for byte, codepoint in zip(direct, codepoints)}


_BYTE_ENCODER = _bytes_to_unicode()
_BYTE_DECODER = {symbol: byte for byte, symbol in _BYTE_ENCODER.items()}


def _split_isolated(pattern, text):
    """Apply one tokenizers ``Split`` with behavior ``Isolated`` to one piece."""
    pieces = []
    last = 0
    for match in pattern.finditer(text):
        if match.start() > last:
            pieces.append(text[last:match.start()])
        if match.end() > match.start():
            pieces.append(match.group())
        last = match.end()
    if last < len(text):
        pieces.append(text[last:])
    return pieces


class DeepSeekTokenizer:
    """Byte-level BPE tokenizer loaded from a checkpoint ``tokenizer.json``."""

    def __init__(self, tokenizer_file):
        path = Path(tokenizer_file)
        with path.open(encoding="utf-8") as handle:
            data = json.load(handle)
        model = data["model"]
        if model.get("type") != "BPE":
            raise ValueError("expected a BPE tokenizer, got %r" % (model.get("type"),))

        self.tokenizer_file = path
        self.vocab = {token: int(index) for token, index in model["vocab"].items()}
        self.id_to_token = {index: token for token, index in self.vocab.items()}
        self.merges = list(model["merges"])
        self.merge_ranks = {}
        for rank, merge in enumerate(self.merges):
            left, middle, right = merge.partition(" ")
            if not middle:
                raise ValueError("malformed merge entry %r" % (merge,))
            self.merge_ranks.setdefault((left, right), rank)

        self.added_tokens = list(data["added_tokens"])
        self.added_by_id = {int(entry["id"]): entry for entry in self.added_tokens}
        self.added_by_content = {entry["content"]: entry for entry in self.added_tokens}
        ordered = sorted(self.added_by_content, key=lambda content: (-len(content), content))
        self._added_re = re.compile("|".join(re.escape(content) for content in ordered))

    def pretokenize(self, text):
        """Return the raw-text pieces produced by the JSON Split sequence."""
        pieces = [text]
        for pattern in (_NUMBER_RE, _CJK_RE, _SPLIT_RE):
            split_pieces = []
            for piece in pieces:
                split_pieces.extend(_split_isolated(pattern, piece))
            pieces = split_pieces
        return pieces

    def _split_added(self, text):
        """Split text into (is_added, value) spans, keeping added tokens atomic."""
        spans = []
        last = 0
        for match in self._added_re.finditer(text):
            if match.start() > last:
                spans.append((False, text[last:match.start()]))
            spans.append((True, self.added_by_content[match.group()]["id"]))
            last = match.end()
        if last < len(text):
            spans.append((False, text[last:]))
        return spans

    def _bpe(self, symbols):
        """Apply BPE merges by rank, merging every occurrence of the best pair."""
        symbols = list(symbols)
        while len(symbols) > 1:
            best_rank = None
            best_pair = None
            for index in range(len(symbols) - 1):
                pair = (symbols[index], symbols[index + 1])
                rank = self.merge_ranks.get(pair)
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank = rank
                    best_pair = pair
            if best_pair is None:
                break
            left, right = best_pair
            merged = []
            index = 0
            while index < len(symbols):
                if index + 1 < len(symbols) and symbols[index] == left and symbols[index + 1] == right:
                    merged.append(left + right)
                    index += 2
                else:
                    merged.append(symbols[index])
                    index += 1
            symbols = merged
        return symbols

    def encode(self, text):
        """Encode one string; added tokens are emitted atomically by id."""
        if not isinstance(text, str):
            raise TypeError("encode expects str, got %s" % type(text).__name__)
        ids = []
        for is_added, value in self._split_added(text):
            if is_added:
                ids.append(value)
                continue
            for piece in self.pretokenize(value):
                if not piece:
                    continue
                symbols = "".join(_BYTE_ENCODER[byte] for byte in piece.encode("utf-8"))
                for token in self._bpe(symbols):
                    try:
                        ids.append(self.vocab[token])
                    except KeyError as error:
                        raise ValueError("BPE produced an unknown token %r" % (token,)) from error
        return ids

    def decode(self, ids, skip_special_tokens=False):
        """Decode ids; base tokens go through the byte decoder, added tokens verbatim."""
        parts = []
        byte_buffer = bytearray()
        for token_id in ids:
            token_id = int(token_id)
            added = self.added_by_id.get(token_id)
            if added is not None:
                if skip_special_tokens and added.get("special"):
                    continue
                if byte_buffer:
                    parts.append(byte_buffer.decode("utf-8", errors="replace"))
                    byte_buffer = bytearray()
                parts.append(added["content"])
                continue
            token = self.id_to_token.get(token_id)
            if token is None:
                raise ValueError("token id %d is outside the vocabulary" % token_id)
            for symbol in token:
                byte = _BYTE_DECODER.get(symbol)
                if byte is None:
                    raise ValueError("token %r contains a non-byte-level symbol" % (token,))
                byte_buffer.append(byte)
        if byte_buffer:
            parts.append(byte_buffer.decode("utf-8", errors="replace"))
        return "".join(parts)

    def encode_added_token(self, content):
        """Return the id reserved for an added token, or None."""
        entry = self.added_by_content.get(content)
        return None if entry is None else entry["id"]


def _default_model_root():
    return os.environ.get("DEEPSEEK_V41_MODEL_ROOT", DEFAULT_MODEL_ROOT)


_TOKENIZER_CACHE = {}


def load_tokenizer(model_root=None):
    """Load and cache the tokenizer for ``model_root`` (or the default root)."""
    root = Path(model_root or _default_model_root())
    path = root / TOKENIZER_FILE
    key = str(path)
    tokenizer = _TOKENIZER_CACHE.get(key)
    if tokenizer is None:
        tokenizer = DeepSeekTokenizer(path)
        _TOKENIZER_CACHE[key] = tokenizer
    return tokenizer


_DEFAULT_TOKENIZER = None


def default_tokenizer():
    global _DEFAULT_TOKENIZER
    if _DEFAULT_TOKENIZER is None:
        _DEFAULT_TOKENIZER = load_tokenizer()
    return _DEFAULT_TOKENIZER


def encode(text):
    """Encode ``text`` with the default checkpoint tokenizer."""
    return default_tokenizer().encode(text)


def decode(ids, skip_special_tokens=False):
    """Decode ``ids`` with the default checkpoint tokenizer."""
    return default_tokenizer().decode(ids, skip_special_tokens=skip_special_tokens)


def _render_plain_message(index, messages):
    role = messages[index]["role"]
    content = messages[index]["content"]
    prompt = ""
    if index == 0 and role == "system":
        prompt += SYSTEM_TOKEN
    if role == "system":
        if index > 0:
            prompt += SYSTEM_TOKEN
        prompt += content
    elif role == "user":
        prompt += USER_TOKEN + content
    elif role == "latest_reminder":
        prompt += LATEST_REMINDER_TOKEN + content
    elif role == "assistant":
        prompt += content + EOS_TOKEN
    else:
        raise ValueError("unsupported role %r" % (role,))
    next_role = messages[index + 1]["role"] if index + 1 < len(messages) else None
    if next_role not in (None, "assistant", "latest_reminder"):
        return prompt
    if role == "user" or (role == "system" and index > 0):
        prompt += ASSISTANT_TOKEN + THINK_END_TOKEN
    return prompt


def encode_messages(messages, thinking_mode="chat", add_default_bos_token=True):
    """Render the plain-text chat subset shared with ``encoding.py``.

    Only text-only ``system``, ``user``, ``assistant`` and ``latest_reminder``
    turns are supported in ``thinking_mode="chat"``.  The reference file remains
    authoritative for tools, images, thinking mode, reasoning effort, task
    tokens and context carry-over.
    """
    if thinking_mode != "chat":
        raise ValueError('only thinking_mode="chat" is implemented; see $MODEL_ROOT/encoding/encoding.py')
    allowed = {"role", "content"}
    roles = {"system", "user", "assistant", "latest_reminder"}
    for index, message in enumerate(messages):
        unknown = set(message) - allowed
        if unknown:
            raise ValueError("message %d uses unsupported keys: %s" % (index, sorted(unknown)))
        if message.get("role") not in roles:
            raise ValueError("message %d has unsupported role %r" % (index, message.get("role")))
        if not isinstance(message.get("content"), str):
            raise ValueError("message %d content must be a str" % index)
    prompt = BOS_TOKEN if add_default_bos_token else ""
    for index in range(len(messages)):
        prompt += _render_plain_message(index, messages)
    return prompt


_FIXTURES = (
    ("english", "Hello, world!"),
    ("chinese", "你好，世界！"),
    ("digits", "12345"),
    ("digits mixed", "版本 3.14 发布于 2025 年"),
    ("code", "def f(x):\n    return x + 1\n"),
    ("punctuation", "Hello!!! What's next? (yes: no.)"),
    ("whitespace", "alpha   beta\n\n\tgamma  "),
    ("combining marks", "Café naive"),
    ("symbols", "OK 👍🏽 — done."),
)

_SPLIT_CASES = (
    ("12345", ["123", "45"]),
    ("a1234b", ["a", "123", "4", "b"]),
    ("abc123def", ["abc", "123", "def"]),
    ("3.14", ["3", ".", "14"]),
    ("你好世界", ["你好世界"]),
    ("中a文", ["中", "a", "文"]),
    ("Hello!!!", ["Hello", "!!!"]),
    ("Hello, world!", ["Hello", ",", " world", "!"]),
    ("a  b", ["a", " ", " b"]),
    ("a  ", ["a", "  "]),
)


def _check_assets(tokenizer):
    assert len(tokenizer.vocab) == BASE_VOCAB_SIZE, "unexpected base vocab size"
    assert len(tokenizer.id_to_token) == BASE_VOCAB_SIZE, "base vocab ids are not contiguous"
    assert max(tokenizer.id_to_token) + 1 == BASE_VOCAB_SIZE, "unexpected base id range"
    assert len(tokenizer.merges) == 127741, "unexpected merge count"
    assert len(tokenizer.merge_ranks) == len(tokenizer.merges), "duplicate merge pairs"
    assert len(tokenizer.added_tokens) == 1283, "unexpected added-token count"
    added_ids = [int(entry["id"]) for entry in tokenizer.added_tokens]
    assert len(set(added_ids)) == len(added_ids), "duplicate added-token ids"
    assert all(0 <= token_id < TOTAL_VOCAB_SIZE for token_id in added_ids), "added id out of range"
    assert tokenizer.encode_added_token(BOS_TOKEN) == 0, "BOS token id changed"
    assert tokenizer.encode_added_token(EOS_TOKEN) == 1, "EOS token id changed"
    assert tokenizer.encode_added_token(PAD_TOKEN) == 2, "pad token id changed"
    for byte, symbol in _BYTE_ENCODER.items():
        assert tokenizer.vocab.get(symbol) is not None, "missing byte-level token for byte %d" % byte
    for left, right in tokenizer.merge_ranks:
        assert left + right in tokenizer.vocab, "merge result missing from vocab: %r" % (left + right,)
    for token_id, token in tokenizer.id_to_token.items():
        if token_id in tokenizer.added_by_id:
            continue
        assert all(symbol in _BYTE_DECODER for symbol in token), "non-byte-level base token %r" % (token,)


def _check_fixtures(tokenizer):
    for name, text in _FIXTURES:
        ids = tokenizer.encode(text)
        assert isinstance(ids, list), "encode must return a list"
        assert ids == tokenizer.encode(text), "encode is not deterministic for %s" % name
        assert all(isinstance(token_id, int) and 0 <= token_id < TOTAL_VOCAB_SIZE for token_id in ids), name
        assert ids, "non-empty text encoded to no ids: %s" % name
        assert tokenizer.decode(ids) == text, "round-trip failed for %s" % name
        assert tokenizer.decode(ids, skip_special_tokens=True) == text, (
            "skip-special decode failed for %s" % name
        )
    assert tokenizer.encode("") == [], "empty text must encode to no ids"
    assert tokenizer.decode([]) == "", "empty id list must decode to an empty string"
    try:
        tokenizer.decode([TOTAL_VOCAB_SIZE])
    except ValueError:
        pass
    else:
        raise AssertionError("out-of-range id must raise ValueError")


def _check_tokenizer_config(tokenizer):
    config_file = tokenizer.tokenizer_file.parent / "tokenizer_config.json"
    if not config_file.exists():
        return
    with config_file.open(encoding="utf-8") as handle:
        config = json.load(handle)
    assert config.get("add_bos_token") is False, "add_bos_token must stay false"
    assert config.get("add_eos_token") is False, "add_eos_token must stay false"
    assert config["bos_token"]["content"] == BOS_TOKEN, "BOS token spelling changed"
    assert config["eos_token"]["content"] == EOS_TOKEN, "EOS token spelling changed"


def _check_pretokenizer(tokenizer):
    for text, expected in _SPLIT_CASES:
        actual = tokenizer.pretokenize(text)
        assert actual == expected, "pretokenizer split changed for %r: %r != %r" % (text, actual, expected)
    assert tokenizer.pretokenize("12345") == ["123", "45"], "1-3 digit rule not applied"
    assert tokenizer.pretokenize("你好世界") == ["你好世界"], "CJK run was not isolated"
    assert tokenizer.pretokenize("中a文") == ["中", "a", "文"], "CJK run boundaries changed"


def _check_added_tokens(tokenizer):
    for entry in tokenizer.added_tokens:
        content = entry["content"]
        token_id = int(entry["id"])
        ids = tokenizer.encode(content)
        assert ids == [token_id], "added token %r was split: %r" % (content, ids)
        assert tokenizer.decode([token_id]) == content, "added token %r did not decode verbatim" % (content,)
    special = [entry for entry in tokenizer.added_tokens if entry.get("special")]
    space_ids = tokenizer.encode(" ")
    assert len(space_ids) == 1, "a single space must encode to one token"
    combined = " ".join(entry["content"] for entry in special)
    expected = []
    for index, entry in enumerate(special):
        if index:
            expected.extend(space_ids)
        expected.append(int(entry["id"]))
    actual = tokenizer.encode(combined)
    assert actual == expected, "special tokens were not kept atomic in a combined string"
    assert tokenizer.decode(actual) == combined, "combined special-token round-trip failed"


def _check_messages(tokenizer):
    probe = "Who are you?"
    prompt = encode_messages([{"role": "user", "content": probe}])
    expected = BOS_TOKEN + USER_TOKEN + probe + ASSISTANT_TOKEN + THINK_END_TOKEN
    assert prompt == expected, "single user-turn prompt does not match encoding.py"
    reference = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there! How can I help you?"},
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "The capital of France is Paris."},
    ]
    reference_prompt = encode_messages(reference)
    reference_file = tokenizer.tokenizer_file.parent / "encoding" / "tests" / "test_output_2.txt"
    if reference_file.exists():
        expected_reference = reference_file.read_text(encoding="utf-8")
        assert reference_prompt == expected_reference, (
            "plain chat subset differs from encoding.py test_output_2"
        )
    return probe, prompt


def run_validation(model_root=None):
    tokenizer = load_tokenizer(model_root)
    print("tokenizer file:", tokenizer.tokenizer_file)
    print("base vocab: %d merges: %d added: %d" % (
        len(tokenizer.vocab), len(tokenizer.merges), len(tokenizer.added_tokens)
    ))
    _check_assets(tokenizer)
    print("PASS asset integrity (vocab, merges, byte alphabet, added tokens)")
    _check_tokenizer_config(tokenizer)
    print("PASS tokenizer_config (no BOS/EOS insertion, special-token spellings)")
    _check_pretokenizer(tokenizer)
    print("PASS pretokenizer splits (numbers, CJK, punctuation, whitespace)")
    _check_fixtures(tokenizer)
    print("PASS fixture round-trips (%d fixtures)" % len(_FIXTURES))
    _check_added_tokens(tokenizer)
    print("PASS added-token atomicity (%d tokens)" % len(tokenizer.added_tokens))
    probe, prompt = _check_messages(tokenizer)
    print("PASS plain chat message subset (encoding.py reference)")
    probe_ids = tokenizer.encode(probe)
    prompt_ids = tokenizer.encode(prompt)
    assert tokenizer.decode(probe_ids) == probe, "probe round-trip failed"
    assert tokenizer.decode(prompt_ids) == prompt, "prompt round-trip failed"
    assert all(0 <= token_id < TOTAL_VOCAB_SIZE for token_id in probe_ids + prompt_ids), (
        "probe ids out of range"
    )
    print('"Who are you?" raw ids:', probe_ids)
    print('"Who are you?" plain chat prompt ids:', prompt_ids)
    print("all tokenizer assertions passed")
    return tokenizer


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-root", default=None, help="directory holding tokenizer.json")
    args = parser.parse_args(argv)
    run_validation(args.model_root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
