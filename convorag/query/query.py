from ..preprocessing.text_preprocessor import TextPreprocessor
from ..embedding.embedding import EmbeddingModel
from ..vectorstore.vectorstore import VectorStore
from ..database.database import Database
import numpy as np

class QuerySystem:
    def __init__(self, embedding_model, vectorstore):
        self.embedding_model = embedding_model
        self.vectorstore = vectorstore
    
    def preprocess_query(self, query):
        preprocessor = TextPreprocessor(query)
        return preprocessor.preprocess()

    def generate_query_embedding(self, preprocessed_query):
        return self.embedding_model.embed_text(preprocessed_query)

    def search_vectorstore(self, query_embedding, k=5):
        return self.vectorstore.search(query_embedding, k)

    def retrieve_conversations(self, indices):
        return [
            self.vectorstore.get_conversation_by_index(idx)
            for idx in indices
        ]
    
    def query(self, query_text, k=5):
        query_chunks = self.preprocess_query(query_text)
        query_embedding = self.generate_query_embedding(query_chunks[0])
        distances, indices = self.search_vectorstore(query_embedding, k)
        conversations = self.retrieve_conversations(indices[0])
        conversation_ids = [c["id"] for c in conversations]
        result = []
        for i in range(len(conversations)):
            conversation = conversations[i]
            if conversation["id"] in conversation_ids[:i]:
                continue
            conversation["distance"] = distances[0][i]
            result.append(conversation)
        return result

def querysystem_from_path(db_path):
    embedding_model = EmbeddingModel()
    db = Database(db_path)
    vectorstore = VectorStore(db, embedding_model.embedding_dimension)
    return QuerySystem(embedding_model, vectorstore)
