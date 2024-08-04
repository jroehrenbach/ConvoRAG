from .chatgpt_extractor import ChatGPTExtractor
from ..database.database import Database

class Importer:
    def __init__(self, db):
        if type(db) == str:
            self.db = Database(db)
        else:
            self.db = db
    
    def import_chatgpt_conversations(self, path):
        chatgpt_extractor = ChatGPTExtractor(path)
        conversations = chatgpt_extractor.extract_conversations()
        messages = chatgpt_extractor.extract_messages()
        self.db.insert_conversations(conversations)
        self.db.insert_messages(messages)
        