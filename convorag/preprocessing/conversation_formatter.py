class ConversationFormatter:
    def __init__(self, messages):
        self.messages = messages
        self.message_ids = [message["id"] for message in messages]
        self.messages_dict = {message["id"]: message for message in messages}
        self.qa_chunks = []

    @staticmethod
    def combine_messages_to_string(messages):
        formatted_messages = []
        for message in messages:
            content = message["content"].replace("</", "").replace("<", "").replace(">", "")
            formatted_message = (
                f"<message>\n"
                f"\t<role>: {message['author_role']}\n"
                f"\t<type>: {message['content_type']}\n"
                f"\t<content>\n{content}\n\t</content>\n"
                f"</message>"
            )
            formatted_messages.append(formatted_message)
        return "\n".join(formatted_messages)
    
    def get_last_message(self):
        return max(self.messages, key=lambda x: x["create_time"])
    
    def get_parent_message(self, message):
        return self.messages_dict[message["parent_id"]]
    
    def generate_qa_chunks(self, last_message):
        chunk_messages = [last_message]

        # find all message in qa pair (can be more than 2!)
        while True:
            parent_message = self.get_parent_message(chunk_messages[-1])
            chunk_messages.append(parent_message)
            if parent_message["author_role"] == "user":
                break

        # generate chunk string
        qa_chunk = self.combine_messages_to_string(chunk_messages[::-1])
        self.qa_chunks.append(qa_chunk)
        
        # continue if last parent_message has parent
        if not parent_message["parent_id"] in self.message_ids:
            return
        else:
            last_message = self.get_parent_message(parent_message)
            self.generate_qa_chunks(last_message)
    
    def get_qa_chunks(self):
        last_message = self.get_last_message()
        self.generate_qa_chunks(last_message)
        return self.qa_chunks[::-1]
