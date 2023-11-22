from abc import abstractmethod

import torch
import torchvision.transforms as T
from transformers import T5Tokenizer, CLIPImageProcessor


class ConditionProcessor:

    def __init__(self, tokens_length):
        self.tokens_length = tokens_length

    def encode(self, conditions_embeddings):
        condition_model_input = {}
        for model_name, embeddings in conditions_embeddings.items():
            num_tokens, embed_dim = embeddings.shape
            pad_embeds = np.zeros((self.tokens_length[model_name] - num_tokens, embed_dim), dtype=embeddings.dtype)
            embeddings = np.append(embeddings, pad_embeds, axis=0)
            attention_mask = np.append(np.ones(num_tokens), np.zeros(self.tokens_length[model_name] - num_tokens))
            condition_model_input[model_name] = {
                'embeddings': torch.tensor(embeddings, dtype=torch.float32),
                'attention_mask': torch.tensor(attention_mask, dtype=torch.long)
            }
        return condition_model_input


class T5TextConditionProcessor:

    def __init__(self, tokens_length, processor_names):
        self.tokens_length = tokens_length['t5']
        self.processor = T5Tokenizer.from_pretrained(processor_names['t5'])

    def encode(self, text=None, negative_text=None):
        encoded = self.processor(text, max_length=self.tokens_length, truncation=True)
        pad_length = self.tokens_length - len(encoded['input_ids'])
        input_ids = encoded['input_ids'] + [self.processor.pad_token_id] * pad_length
        attention_mask = encoded['attention_mask'] + [0] * pad_length
        condition_model_input = {'t5': {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long)
        }}

        if negative_text is not None:
            negative_encoded = self.processor(negative_text, max_length=self.tokens_length, truncation=True)
            negative_input_ids = negative_encoded['input_ids'][:len(encoded['input_ids'])]
            negative_input_ids[-1] = self.processor.eos_token_id
            negative_pad_length = self.tokens_length - len(negative_input_ids)
            negative_input_ids = negative_input_ids + [self.processor.pad_token_id] * negative_pad_length
            negative_attention_mask = encoded['attention_mask'] + [0] * pad_length
            negative_condition_model_input = {'t5': {
                'input_ids': torch.tensor(negative_input_ids, dtype=torch.long),
                'attention_mask': torch.tensor(negative_attention_mask, dtype=torch.long)
            }}
        else:
            negative_condition_model_input = None
        return condition_model_input, negative_condition_model_input
