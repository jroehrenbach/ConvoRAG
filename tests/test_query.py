import unittest
import numpy as np
from convorag.preprocessing.text_preprocessor import TextPreprocessor
from convorag.embedding.embedding import EmbeddingModel
from convorag.vectorstore.vectorstore import VectorStore
from convorag.database.database import Database
from convorag.query.query import QuerySystem, querysystem_from_path

class TestQuerySystem(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.db_path = 'tests/data/conversations.db'
        cls.query_system = querysystem_from_path(cls.db_path)
        cls.test_query_text = "All conversation about AI"
        cls.k = 5  # Increased k value
    
    def test_preprocess_query(self):
        preprocessed_query = self.query_system.preprocess_query(self.test_query_text)
        self.assertIsInstance(preprocessed_query, list)
        self.assertGreater(len(preprocessed_query), 0)
    
    def test_generate_query_embedding(self):
        preprocessed_query = self.query_system.preprocess_query(self.test_query_text)
        query_embedding = self.query_system.generate_query_embedding(preprocessed_query[0])  # Use the first chunk
        self.assertIsInstance(query_embedding, np.ndarray)
        self.assertEqual(query_embedding.shape[1], self.query_system.embedding_model.embedding_dimension)
    
    def test_search_vectorstore(self):
        preprocessed_query = self.query_system.preprocess_query(self.test_query_text)
        query_embedding = self.query_system.generate_query_embedding(preprocessed_query[0])  # Use the first chunk
        distances, indices = self.query_system.search_vectorstore(query_embedding, self.k)
        self.assertIsInstance(distances, np.ndarray)
        self.assertIsInstance(indices, np.ndarray)
    
    def test_retrieve_conversations(self):
        preprocessed_query = self.query_system.preprocess_query(self.test_query_text)
        query_embedding = self.query_system.generate_query_embedding(preprocessed_query[0])  # Use the first chunk
        distances, indices = self.query_system.search_vectorstore(query_embedding, self.k)
        conversations = self.query_system.retrieve_conversations(indices)
        self.assertIsInstance(conversations, list)
        self.assertGreaterEqual(len(conversations), 1)
    
    def test_query(self):
        results = self.query_system.query(self.test_query_text, k=self.k)
        self.assertIsInstance(results, list)
        self.assertGreaterEqual(len(results), 1)
        for result in results:
            self.assertIsInstance(result, tuple)

if __name__ == '__main__':
    unittest.main()
