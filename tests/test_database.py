import unittest
import os
import json
from datetime import datetime
from convorag.database.database import Database

class TestDatabase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Define the path to the test conversations.json file
        cls.conversations_file_path = 'tests/data/conversations_extracted.json'
        cls.messages_file_path = 'tests/data/messages_extracted.json'

        # Ensure the test files exists
        if not os.path.exists(cls.conversations_file_path):
            raise FileNotFoundError(f"Test file {cls.conversations_file_path} does not exist.")
        if not os.path.exists(cls.messages_file_path):
            raise FileNotFoundError(f"Test file {cls.messages_file_path} does not exist.")
        
        # Read the test files
        cls.conversations = json.load(open(cls.conversations_file_path, "r"))
        cls.messages = json.load(open(cls.messages_file_path, "r"))

        # Define the path to the test SQLite database
        cls.db_path = 'tests/data/test_chatgpt_conversations.db'

        # Create an instance of the Database and import the extracted data
        cls.db = Database(cls.db_path)
        cls.db.insert_conversations(cls.conversations)
        cls.db.insert_messages(cls.messages)

    @classmethod
    def tearDownClass(cls):
        # Close the database connection
        cls.db.conn.close()

        # Clean up the test database file
        if os.path.exists(cls.db_path):
            os.remove(cls.db_path)

    def test_read_conversations(self):
        # Test reading all conversations from the database
        conversations = self.db.read_conversations()
        self.assertGreater(len(conversations), 0, "No conversations read from the database.")

    def test_read_messages(self):
        # Test reading all messages from the database
        messages = self.db.read_messages()
        self.assertGreater(len(messages), 0, "No messages read from the database.")

if __name__ == '__main__':
    unittest.main()
