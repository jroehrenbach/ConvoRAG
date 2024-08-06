# vectorstore.py
import faiss
import numpy as np
from ..database.database import Database
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class VectorStore:
    def __init__(self, db, dimension):
        self.index = faiss.IndexFlatL2(dimension)
        self.index_to_embedding_id = []
        if type(db) == str:
            self.db = Database(db)
        else:
            self.db = db
        self._add_embeddings_from_db()
    
    def _add_embeddings_from_db(self):
        embedding_ids = self.db.read_embedding_ids()
        for embedding_id in embedding_ids:
            embedding_item = self.db.get_embedding_by_id(
                embedding_id
            )
            self.index.add(embedding_item["embedding"])
            self.index_to_embedding_id.append(embedding_id)
    
    def store_embeddings(self, conversation_id, embeddings):
        self.db.store_embeddings(conversation_id, embeddings)
        embedding_ids = self.db.get_embedding_ids_by_conversation_id(
            conversation_id
        )
        for embedding in embeddings:
            self.index.add(embedding)
        self.index_to_embedding_id.extend(embedding_ids)

    def search(self, query_embedding, k=5):
        query_np = np.array(query_embedding, dtype=np.float32)
        if len(query_np.shape) == 1:
            query_np = query_np.reshape(1, -1)
        indices = self.index.search(query_np, k)
        return indices
    
    def get_conversation_by_index(self, idx):
        embedding_id = self.index_to_embedding_id[idx]
        conversation_id = self.db.get_conversation_id_by_embedding_id(
            embedding_id
        )
        return self.db.get_conversation_by_id(conversation_id)
