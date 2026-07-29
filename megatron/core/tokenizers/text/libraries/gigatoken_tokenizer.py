# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import gigatoken as gt

from .abstract_tokenizer import MegatronTokenizerTextAbstract

logger = logging.getLogger(__name__)


class GigaTokenTokenizer(MegatronTokenizerTextAbstract):
    """GigaToken tokenizer https://github.com/marcelroed/gigatoken."""

    def __init__(self, tokenizer_path: str):
        #self.tokenizer = gt.Tokenizer.from_tiktoken(f"{tokenizer_path}.tiktoken", pretokenizer="nemotron")
        self.tokenizer = gt.Tokenizer(tokenizer_path).as_tiktoken()

    def text_to_ids(self, text: str | List[str] | Path) -> List[int]:
        return self.tokenizer.encode(text)

    def ids_to_text(self, ids: List[int]) -> str:
        return str(self.tokenizer.decode(ids), encoding="utf-8")

    def text_to_tokens(self, text: str) -> List[str]:
        raise NotImplementedError("This method is not supported for gigatoken.")

    def ids_to_tokens(self, ids: List[int]) -> List[str]:
        raise NotImplementedError("This method is not supported for gigatoken.")

    def tokens_to_text(self, tokens: List[str]) -> str:
        raise NotImplementedError("This method is not supported for gigatoken.")

    def tokens_to_ids(self, tokens: List[str]) -> List[int]:
        raise NotImplementedError("This method is not supported for gigatoken.")

    def add_special_tokens(self, special_tokens_dict: dict) -> int:
        raise NotImplementedError("This method is not supported for gigatoken.")

    @property
    def vocab_size(self) -> int:
        """Returns size of tokenizer vocabulary."""
        return self.tokenizer.vocab_size

    @property
    def vocab(self) -> dict[int, bytes]:
        return self.tokenizer.vocab

    def tokens_to_ids(self, tokens: List[str]) -> List[int]:
        """Converts list of tokens to it's ids."""
        ids = self.tokenizer.convert_tokens_to_ids(tokens)
        return ids

    @property
    def eod(self) -> int:
        """Returns EOD token id."""
        return 1 #self.text_to_ids(self.tokenizer.eot_token)[0]