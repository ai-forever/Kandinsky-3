import torch
from transformers import T5Tokenizer


class T5TextConditionProcessor:

    def __init__(self, tokens_length, processor_path):
        self.tokens_length = tokens_length
        self.processor = T5Tokenizer.from_pretrained(processor_path)

    def encode(self, text=None, negative_text=None):
        encoded = self.processor(text, max_length=self.tokens_length, truncation=True)
        pad_length = self.tokens_length - len(encoded['input_ids'])
        input_ids = encoded['input_ids'] + [self.processor.pad_token_id] * pad_length
        attention_mask = encoded['attention_mask'] + [0] * pad_length
        condition_model_input = {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long)
        }

        if negative_text is not None:
            negative_encoded = self.processor(negative_text, max_length=self.tokens_length, truncation=True)
            negative_input_ids = negative_encoded['input_ids'][:len(encoded['input_ids'])]
            negative_input_ids[-1] = self.processor.eos_token_id
            negative_pad_length = self.tokens_length - len(negative_input_ids)
            negative_input_ids = negative_input_ids + [self.processor.pad_token_id] * negative_pad_length
            negative_attention_mask = encoded['attention_mask'] + [0] * pad_length
            negative_condition_model_input = {
                'input_ids': torch.tensor(negative_input_ids, dtype=torch.long),
                'attention_mask': torch.tensor(negative_attention_mask, dtype=torch.long)
            }
        else:
            negative_condition_model_input = None
        return condition_model_input, negative_condition_model_input
