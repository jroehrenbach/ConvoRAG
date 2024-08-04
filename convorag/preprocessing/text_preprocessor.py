import spacy
import re

class TextPreprocessor:
    def __init__(self, text, token_limit=512):
        self.nlp = spacy.load('en_core_web_sm')
        self.raw_text = text
        self.tokens = []
        self.chunks = []
        self.token_limit = token_limit
        self.tag_pattern = re.compile(r'<[^>]+>')

    def clean_text(self):
        # Only remove unwanted characters while preserving tags and specific characters
        cleaned_text = self.raw_text.replace("\t", " ")
        self.raw_text = cleaned_text

    def tokenize_text(self):
        # Split text into parts: tags and regular text
        parts = self.tag_pattern.split(self.raw_text)
        tags = self.tag_pattern.findall(self.raw_text)
        
        for part in parts:
            doc = self.nlp(part)
            for token in doc:
                if not token.is_punct or token.text in [":", "\n"]:
                    self.tokens.append(token)
            # Insert tags back into their original positions
            if tags:
                self.tokens.append(tags.pop(0))

    def lemmatize_text(self):
        lemmatized_tokens = []
        for token in self.tokens:
            if isinstance(token, str) and self.tag_pattern.match(token):
                lemmatized_tokens.append(token)
            elif not token.is_stop and token.lemma_ != '-PRON-':
                lemmatized_tokens.append(token.lemma_.lower() if not token.is_punct else token.text)
        self.tokens = lemmatized_tokens

    def reconstruct_text(self):
        return ''.join([
            token if isinstance(token, str) and self.tag_pattern.match(token) else ' ' + token
            for token in self.tokens
        ]).strip()

    def chunk_text(self):
        chunk = []
        current_length = 0

        for token in self.tokens:
            token_length = 1 if isinstance(token, str) and self.tag_pattern.match(token) else len(token.split())
            if current_length + token_length > self.token_limit:
                self.chunks.append(''.join([
                    t if isinstance(t, str) and self.tag_pattern.match(t) else ' ' + t
                    for t in chunk
                ]).strip())
                chunk = []
                current_length = 0
            chunk.append(token)
            current_length += token_length
        
        if chunk:
            self.chunks.append(''.join([
                t if isinstance(t, str) and self.tag_pattern.match(t) else ' ' + t
                for t in chunk
            ]).strip())

    def preprocess(self):
        self.clean_text()
        self.tokenize_text()
        self.lemmatize_text()
        self.chunk_text()
        return self.chunks
