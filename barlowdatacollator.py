import torch
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
import torch
from transformers import PreTrainedTokenizerBase, BertTokenizer, BatchEncoding, BertModel, BertConfig
import pdb
from copy import deepcopy

@dataclass
class DataCollatorForLanguageModeling:
    """
    Data collator used for language modeling. Inputs are dynamically padded to the maximum length of a batch if they
    are not all of the same length.
    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        mlm (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not to use masked language modeling. If set to :obj:`False`, the labels are the same as the
            inputs with the padding tokens ignored (by setting them to -100). Otherwise, the labels are -100 for
            non-masked tokens and the value to predict for the masked token.
        mlm_probability (:obj:`float`, `optional`, defaults to 0.15):
            The probability with which to (randomly) mask tokens in the input, when :obj:`mlm` is set to :obj:`True`.
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
    .. note::
        For best performance, this data collator should be used with a dataset having items that are dictionaries or
        BatchEncoding, with the :obj:`"special_tokens_mask"` key, as returned by a
        :class:`~transformers.PreTrainedTokenizer` or a :class:`~transformers.PreTrainedTokenizerFast` with the
        argument :obj:`return_special_tokens_mask=True`.
    """

    tokenizer: PreTrainedTokenizerBase
    mlm: bool = True
    mlm_probability: float = 0.15
    pad_to_multiple_of: Optional[int] = None

    def __post_init__(self):
        if self.mlm and self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. "
                "You should pass `mlm=False` to train on causal language modeling instead."
            )

    def __call__(
        self, examples: List[Union[List[int], torch.Tensor, Dict[str, torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # Handle dict or lists with proper padding and conversion to tensor.
        if isinstance(examples[0], (dict, BatchEncoding)):
            batch = self.tokenizer.pad(examples, return_tensors="pt", pad_to_multiple_of=self.pad_to_multiple_of)
        else:
            print('Error. This collator only works with dicts or BatchEncoded inputs')
#             batch = {"input_ids": _collate_batch(examples, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)}

        # If special token mask has been preprocessed, pop it from the dict.
        special_tokens_mask = batch.pop("special_tokens_mask", None)
        batch_1 = deepcopy(batch)
        batches = []

        batch["input_ids"], masked_indices_1 = self.mask_tokens(batch["input_ids"], special_tokens_mask=special_tokens_mask)
        batches.append(batch)
        batch_1["input_ids"], masked_indices_2 = self.mask_tokens(batch_1["input_ids"], special_tokens_mask=special_tokens_mask)
        batches.append(batch_1)
        
        batches.append({'masked_indices': masked_indices_1 | masked_indices_2})
            
        return batches
#         if self.mlm:
#             batch["input_ids"], batch["labels"] = self.mask_tokens(batch["input_ids"], special_tokens_mask=special_tokens_mask)
#         else:
#             labels = batch["input_ids"].clone()
#             if self.tokenizer.pad_token_id is not None:
#                 labels[labels == self.tokenizer.pad_token_id] = -100
#             batch["labels"] = labels
#         return batch

    def mask_tokens(
        self, inputs: torch.Tensor, special_tokens_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 1.0)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # The rest of the time (20% of the time) we keep the masked input tokens unchanged
        
#         # 10% of the time, we replace masked input tokens with random word
#         indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
#         random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
#         inputs[indices_random] = random_words[indices_random]

#         # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels