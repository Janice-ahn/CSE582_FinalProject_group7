from transformers import BertTokenizer, RobertaTokenizer, BertModel, RobertaModel
import torch


class BertEncoder:
    def __init__(self, pretrained_model_name='bert-base-uncased', device='cpu'):
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)
        self.model = BertModel.from_pretrained(pretrained_model_name).to(device)
        self.device = device
        self.model.eval()

    def encode(self, utterance):
        inputs = self.tokenizer(utterance, padding=True, truncation=True, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        embeddings = outputs.pooler_output
        return embeddings


class RobertaEncoder:
    def __init__(self, pretrained_model_name='roberta-base', device='cpu'):
        self.tokenizer = RobertaTokenizer.from_pretrained(pretrained_model_name)
        self.model = RobertaModel.from_pretrained(pretrained_model_name).to(device)
        self.device = device
        self.model.eval()

    def encode(self, utterance):
        inputs = self.tokenizer(utterance, padding=True, truncation=True, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        embeddings = outputs.pooler_output
        return embeddings
