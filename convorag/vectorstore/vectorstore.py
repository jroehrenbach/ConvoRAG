# vectorstore.py
import faiss
import numpy as np
from ..database.database import Database
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class VectorStore:
    def __init__(self, db, dimension):
        self.index = faiss.IndexFlatL2(dimension)
        self.index_to_conversation_id = []
        if type(db) == str:
            self.db = Database(db)
        else:
            self.db = db
        self._add_embeddings_from_db()
    
    def _add_embeddings_from_db(self):
        conversations = self.db.read_conversations()
        for conversation in conversations:
            if not self.db.has_embedding(conversation["id"]):
                continue
            embedding_np = self.db.get_embedding_by_conversation_id(
                conversation["id"]
            )
            self.add_embedding(conversation["id"], embedding_np)
    
    def add_embedding(self, conversation_id, embedding):
        embedding_np = np.array(embedding, dtype=np.float32).reshape(1, -1)
        self.index.add(embedding_np)
        self.index_to_conversation_id.append(conversation_id)  # Store the mapping
    
    def store_embedding(self, conversation_id, embedding):
        self.add_embedding(conversation_id, embedding)
        self.db.store_embedding(conversation_id, embedding)

    def search(self, query_embedding, k=5):
        query_np = np.array(query_embedding, dtype=np.float32)
        if len(query_np.shape) == 1:
            query_np = query_np.reshape(1, -1)
        indices = self.index.search(query_np, k)
        return indices
    
    def get_conversation_by_index(self, idx):
        conversation_id = self.index_to_conversation_id[idx]
        return self.db.get_conversation_by_id(conversation_id)
