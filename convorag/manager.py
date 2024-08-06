from .database.database import Database
from .importer.importer import Importer
from .preprocessing.conversation_formatter import ConversationFormatter
from .preprocessing.text_preprocessor import TextPreprocessor
from .embedding.embedding import EmbeddingModel
from .vectorstore.vectorstore import VectorStore
from .query.query import QuerySystem

EMBEDDING_MODEL_NAME='distilbert-base-uncased'

class ConversationManager:
    def __init__(self, db_path, embedding_model_name=EMBEDDING_MODEL_NAME):
        self.db = Database(db_path)
        self.embedding_model = EmbeddingModel(model_name=embedding_model_name)
        self.vectorstore = VectorStore(
            self.db, self.embedding_model.embedding_dimension
        )
        self.query_system = QuerySystem(
            self.embedding_model, self.vectorstore
        )
    
    def import_chatgpt_conversations(self, path):
        importer = Importer(self.db)
        importer.import_chatgpt_conversations(path)
    
    def process_conversations(self):
        conversations = self.db.read_conversations()
        for conversation in conversations:
            if self.db.conversation_has_embeddings(conversation["id"]):
                continue
            messages = self.db.read_messages_by_conversation_id(
                conversation["id"]
            )
            chunks = ConversationFormatter(messages).get_qa_chunks()
            embeddings = []
            for chunk in chunks:
                sub_chunks = TextPreprocessor(chunk).preprocess()
                embedding = self.embedding_model.sequential_embeddings(sub_chunks)
                embeddings.append(embedding)
            self.vectorstore.store_embeddings(conversation["id"], embeddings)
            self.db.mark_conversation_as_embedded(conversation["id"])
    
    def query(self, query_text, k=5):
        return self.query_system.query(query_text, k)
