class ConversationFormatter:
    @staticmethod
    def combine_messages_to_string(messages):
        formatted_messages = []
        for message in messages:
            content = message["content"].replace("</", "").replace("<", "").replace(">", "")
            formatted_message = (
                f"<message>\n"
                f"<role>: {message['author_role']}\n"
                f"<type>: {message['content_type']}\n"
                f"<content>\n{content}\n</content>\n"
                f"</message>"
            )
            formatted_messages.append(formatted_message)
        return "\n".join(formatted_messages)
    