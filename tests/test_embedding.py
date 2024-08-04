import unittest
from transformers import AutoTokenizer, AutoModel, AutoConfig, DistilBertTokenizer, DistilBertTokenizerFast, DistilBertModel, DistilBertConfig
import numpy as np
from convorag.embedding.embedding import EmbeddingModel


class TestEmbeddingModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # This setup is done once for all tests
        cls.model_name = 'distilbert-base-uncased'
        cls.embedding_model = EmbeddingModel(model_name=cls.model_name)

    def test_initialization(self):
        # Test if the model initializes correctly
        self.assertIsInstance(self.embedding_model.tokenizer, (DistilBertTokenizer, DistilBertTokenizerFast))
        self.assertIsInstance(self.embedding_model.model, DistilBertModel)
        self.assertIsInstance(self.embedding_model.config, DistilBertConfig)
        self.assertEqual(self.embedding_model.embedding_dimension, self.embedding_model.config.hidden_size)

    def test_embed_text(self):
        # Test the embed_text function
        text = "This is a test sentence."
        embedding = self.embedding_model.embed_text(text)
        
        self.assertIsInstance(embedding, np.ndarray)
        self.assertEqual(embedding.shape[1], self.embedding_model.embedding_dimension)
    
    def test_sequential_embeddings(self):
        # Test the sequential_embeddings function
        chunks = ["This is the first chunk.", "This is the second chunk."]
        combined_embedding = self.embedding_model.sequential_embeddings(chunks)
        
        self.assertIsInstance(combined_embedding, np.ndarray)
        self.assertEqual(combined_embedding.shape[1], self.embedding_model.embedding_dimension)
    
    def test_embed_text_consistency(self):
        # Ensure that embedding the same text twice results in the same output
        text = "Consistency test sentence."
        embedding1 = self.embedding_model.embed_text(text)
        embedding2 = self.embedding_model.embed_text(text)
        
        np.testing.assert_array_almost_equal(embedding1, embedding2)

    def test_sequential_embeddings_consistency(self):
        # Ensure that sequential embedding of the same chunks results in the same output
        chunks = ["This is the first chunk.", "This is the second chunk."]
        combined_embedding1 = self.embedding_model.sequential_embeddings(chunks)
        combined_embedding2 = self.embedding_model.sequential_embeddings(chunks)
        
        np.testing.assert_array_almost_equal(combined_embedding1, combined_embedding2)

if __name__ == '__main__':
    unittest.main()