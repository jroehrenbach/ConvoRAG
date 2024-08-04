from transformers import AutoTokenizer, AutoModel, AutoConfig
import torch

class EmbeddingModel:
    def __init__(self, model_name='distilbert-base-uncased'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.config = AutoConfig.from_pretrained(model_name)
        self.embedding_dimension = self.config.hidden_size
    
    def embed_text(self, text):
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True)
        outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).detach().numpy()
    
    def sequential_embeddings(self, chunks):
        combined_embedding = self.embed_text(chunks[0])
        for chunk in chunks[1:]:
            combined_input = f"{combined_embedding} {chunk}"
            combined_embedding = self.embed_text(combined_input)
        return combined_embedding
    