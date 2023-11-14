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

    def encode(self, text=None, image=None, ):
        encoded = self.processor(text, max_length=self.tokens_length, truncation=True)
        pad_length = self.tokens_length - len(encoded['input_ids'])
        input_ids = encoded['input_ids'] + [self.processor.pad_token_id] * pad_length
        attention_mask = encoded['attention_mask'] + [0] * pad_length
        condition_model_input = {'t5': {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long)
        }}
        return condition_model_input


class TextImageConditionProcessor:

    def __init__(self, tokens_length, processor_names):
        self.tokens_length = tokens_length
        self.processors = {'text': {}, 'image': {}}
        self.init_processors(processor_names)

    @abstractmethod
    def init_processors(self, processor_names):
        return

    def encode(self, text=None, image=None):
        condition_model_input = {}
        for key, processor in self.processors['text'].items():
            encoded = processor(text, max_length=self.tokens_length[key], truncation=True)
            pad_length = self.tokens_length[key] - len(encoded['input_ids'])
            input_ids = encoded['input_ids'] + [processor.pad_token_id] * pad_length
            attention_mask = encoded['attention_mask'] + [0] * pad_length
            condition_model_input[key] = {
                'input_ids': torch.tensor(input_ids, dtype=torch.long),
                'attention_mask': torch.tensor(attention_mask, dtype=torch.long)
            }
        for key, processor in self.processors['image'].items():
            image = processor(image)['pixel_values'][0]
            condition_model_input[key] = {
                'pixel_values': torch.tensor(image, dtype=torch.float16)
            }
        return condition_model_input

class DummyConditionProcessor(TextImageConditionProcessor):
    def __init__(self, tokens_length, processor_names):
        self.tokens_length = tokens_length

    def encode(self, text=None, image=None):
        condition_model_input= {
            'input_ids': torch.ones(self.tokens_length, dtype=torch.long),
            'attention_mask': torch.ones(self.tokens_length, dtype=torch.long)
        }
        return condition_model_input


class T5CLIPVisionConditionProcessor(TextImageConditionProcessor):

    def init_processors(self, processor_names):
        t5_processor = T5Tokenizer.from_pretrained(processor_names['t5'])
        clip_processor = CLIPImageProcessor.from_pretrained(processor_names['clip'])
        self.processors['text']['t5'] = t5_processor
        self.processors['image']['clip_vision'] = clip_processor
