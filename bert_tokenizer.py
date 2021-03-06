# -*- coding: utf-8 -*-
import torch
from transformers import AutoTokenizer

from torchnlp.encoders import Encoder
from torchnlp.encoders.text import stack_and_pad_tensors
from torchnlp.encoders.text.text_encoder import TextEncoder

from settings import MAX_QUESTION_LENGTH, MAX_PASSAGE_LENGTH, MAX_SEQ_LENGTH
from settings import PRETRAINED_MODEL_NAME


tokenizer_kwargs = {'max_length': MAX_SEQ_LENGTH,
                    'max_q_length': MAX_QUESTION_LENGTH,
                    'max_p_length': MAX_PASSAGE_LENGTH}


class BERTTextEncoder(TextEncoder):
    """
    Wrapper arround BERT tokenizer.
    """

    def __init__(self, pretrained_model=PRETRAINED_MODEL_NAME) -> None:
        self.enforce_reversible = False
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)

        if hasattr(self.tokenizer, 'ids_to_tokens'):
            self.itos = self.tokenizer.ids_to_tokens
            self.size = len(self.itos)
        else:
            self.size = self.tokenizer.vocab_size

    @property
    def unk_index(self) -> int:
        """ Returns the index used for the unknown token. """
        return self.tokenizer.unk_token_id

    @property
    def bos_index(self) -> int:
        """ Returns the index used for the begin-of-sentence token. """
        return self.tokenizer.cls_token_id

    @property
    def eos_index(self) -> int:
        """ Returns the index used for the end-of-sentence token. """
        return self.tokenizer.sep_token_id

    @property
    def padding_index(self) -> int:
        """ Returns the index used for padding. """
        return self.tokenizer.pad_token_id

    @property
    def vocab(self) -> list:
        """
        Returns:
            list: List of tokens in the dictionary.
        """
        return self.tokenizer.vocab

    @property
    def vocab_size(self) -> int:
        """
        Returns:
            int: Number of tokens in the dictionary.
        """
        return self.size

    def encode(self, sequence: str) -> torch.Tensor:
        """ Encodes a 'sequence'.
        :param sequence: String 'sequence' to encode.

        Returns:
            - torch.Tensor: Encoding of the 'sequence'.
        """
        sequence = TextEncoder.encode(self, sequence)
        vector = self.tokenizer.encode(sequence)
        return torch.tensor(vector)

    def batch_encode(self, iterator, dim=0, **kwargs) -> (torch.Tensor, torch.Tensor):
        """
        :param iterator (iterator): Batch of text to encode.
        :param dim (int, optional): Dimension along which to concatenate tensors.
        :param **kwargs: Keyword arguments passed to 'encode'.

        Returns
            torch.Tensor, torch.Tensor: Encoded and padded batch of sequences; Original lengths of
                sequences.
        """
        return stack_and_pad_tensors(
            Encoder.batch_encode(self, iterator, **kwargs),
            padding_index=self.padding_index,
            dim=dim,
        )

    def batch_encode_plus(self, iterator, **kwargs):
        return self.tokenizer.batch_encode_plus(iterator, **kwargs)