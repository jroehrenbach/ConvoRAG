import unittest
from datetime import datetime
import os
from convorag.importer.chatgpt_extractor import ChatGPTExtractor

class TestChatGPTExtractor(unittest.TestCase):
    def setUp(self):
        # Define the path to the test conversations.json file
        self.test_file_path = 'tests/data/conversations.json'  # Replace with the actual path

        # Ensure the test file exists
        if not os.path.exists(self.test_file_path):
            self.fail(f"Test file {self.test_file_path} does not exist.")

        # Create an instance of the ChatGPTExtractor
        self.extractor = ChatGPTExtractor(self.test_file_path)

    def test_extract_conversations(self):
        # Test extracting all conversations
        conversations = self.extractor.extract_conversations()
        self.assertGreater(len(conversations), 0, "No conversations extracted.")

    def test_extract_messages(self):
        # Test extracting all messages
        messages = self.extractor.extract_messages()
        self.assertGreater(len(messages), 0, "No messages extracted.")

if __name__ == '__main__':
    unittest.main()
