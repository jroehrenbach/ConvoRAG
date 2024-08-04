import unittest
from convorag.preprocessing.text_preprocessor import TextPreprocessor

class TestTextPreprocessor(unittest.TestCase):

    def setUp(self):
        with open('tests/data/conversation_doc.txt', 'r', encoding='utf-8') as file:
            self.raw_text = file.read()
        self.pipeline = TextPreprocessor(self.raw_text, token_limit=10)  # Example token limit for testing

    def test_clean_text(self):
        self.pipeline.clean_text()
        self.assertNotIn('\t', self.pipeline.raw_text, "Text cleaning failed to remove tab spaces.")
    
    def test_tokenize_text(self):
        self.pipeline.clean_text()
        self.pipeline.tokenize_text()
        self.assertGreater(len(self.pipeline.tokens), 0, "Tokenization failed to generate tokens.")
    
    def test_lemmatize_text(self):
        self.pipeline.clean_text()
        self.pipeline.tokenize_text()
        self.pipeline.lemmatize_text()
        lemmatized_tokens = [token for token in self.pipeline.tokens if token.isalpha()]
        self.assertGreater(len(lemmatized_tokens), 0, "Lemmatization failed to generate lemmatized tokens.")
    
    def test_reconstruct_text(self):
        self.pipeline.clean_text()
        self.pipeline.tokenize_text()
        self.pipeline.lemmatize_text()
        processed_text = self.pipeline.reconstruct_text()
        self.assertIsInstance(processed_text, str, "Reconstructed text is not a string.")
    
    def test_preprocess(self):
        processed_chunks = self.pipeline.preprocess()
        self.assertIsInstance(processed_chunks, list, "Preprocessed output is not a list of chunks.")
        self.assertGreater(len(processed_chunks), 0, "Preprocessed output is empty.")
        for chunk in processed_chunks:
            self.assertIsInstance(chunk, str, "Each chunk is not a string.")
            self.assertLessEqual(len(chunk.split()), 10, "Chunk exceeds the token limit.")

if __name__ == '__main__':
    unittest.main()
