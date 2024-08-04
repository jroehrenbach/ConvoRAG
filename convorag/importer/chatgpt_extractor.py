import json
import os
from datetime import datetime

class ChatGPTExtractor:
    def __init__(self, file_path):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist.")
        with open(file_path, 'r', encoding='utf-8') as file:
            self.data = json.load(file)

    def extract_conversations(self, start_datetime=None, end_datetime=None):
        conversations = []
        for conv in self.data:
            create_time = conv.get('create_time')
            if self._is_within_datetime_range(create_time, start_datetime, end_datetime):
                conversation = {
                    'id': conv['id'],
                    'title': conv.get('title', ''),
                    'create_time': create_time,
                    'update_time': conv.get('update_time', None)
                }
                conversations.append(conversation)
        return conversations

    def extract_messages(self, start_datetime=None, end_datetime=None):
        messages = []
        for conv in self.data:
            create_time = conv.get('create_time')
            if not self._is_within_datetime_range(create_time, start_datetime, end_datetime):
                continue

            conversation_id = conv['id']  # Assuming title is unique for each conversation
            started = False
            for msg_id, msg_data in conv.get('mapping', {}).items():
                message_data = msg_data.get('message')
                if message_data is None:
                    continue  # Skip messages without content

                if not started and message_data['author']['role'] == 'user':
                    started = True

                if started:
                    create_time = message_data.get('create_time')
                    if self._is_within_datetime_range(create_time, start_datetime, end_datetime):
                        message = {
                            'id': msg_id,
                            'conversation_id': conversation_id,
                            'parent_id': msg_data.get('parent'),
                            'create_time': create_time,
                            'author_role': message_data['author'].get('role'),
                            'content_type': message_data['content'].get('content_type'),
                            'content': self._extract_content(message_data),
                            'language': message_data['content'].get('language'),
                            'metadata': json.dumps(message_data.get('metadata', {})),
                            'model_type': message_data.get('model_slug')
                        }
                        messages.append(message)
        return messages

    def _extract_content(self, message):
        content_type = message['content'].get('content_type')
        if content_type == 'text':
            return ' '.join(message['content'].get('parts', []))
        elif content_type == 'code':
            return message['content'].get('text', '')
        elif content_type == 'tether_browsing_display':
            return message['content'].get('result', '')
        elif message['author']['role'] == 'system':
            return message['content'].get('parts', [''])[0]
        else:
            return ''  # Default case for unknown content types

    def _is_within_datetime_range(self, timestamp, start_datetime, end_datetime):
        if timestamp is None:
            return False
        timestamp_dt = datetime.fromtimestamp(timestamp)
        if start_datetime and timestamp_dt < start_datetime:
            return False
        if end_datetime and timestamp_dt > end_datetime:
            return False
        return True
