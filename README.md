# ConvoRAG: Conversational Retrieval-Augmented Generation

## Overview

ConvoRAG is a system designed to query and retrieve ChatGPT conversations using embeddings and FAISS vector store for efficient similarity search. The project utilizes a combination of SQLite for database management, spaCy for text preprocessing, Hugging Face Transformers for embedding generation, and FAISS for vector storage and search.

## Features

- Store and manage conversations and their embeddings.
- Efficiently search and retrieve conversations similar to a given query.
- Modular architecture for easy extension and customization.

## Technologies Used

- SQLite: For database management.
- spaCy: For text preprocessing.
- Hugging Face Transformers: For embedding generation.
- FAISS: For efficient similarity search.

## Requirements

- Python 3.7+
- SQLite
- spaCy
- Hugging Face Transformers
- FAISS

## Installation

1. Clone the repository:

   ```
   git clone https://github.com/yourusername/ConvoRAG.git
   cd ConvoRAG
   ```

2. Create and activate a virtual environment:

   ```
   python -m venv convo_env
   source convo_env/bin/activate  # On Windows, use `convo_env\Scripts\activate`
   ```

3. Install the required packages:

   ```
   pip install -r requirements.txt
   ```

4. Download spaCy model:

   ```
   python -m spacy download en_core_web_sm
   ```

## Usage

### Example with `convorag.manager.ConversationManager`

Below is an example of how to use `ConversationManager` to extract conversations, generate embeddings, and perform a query.

1. **Extract and Store Conversations**

   ```
   from convorag.manager import ConversationManager

   # Initialize ConversationManager with database path
   manager = ConversationManager(db_path='tests/data/conversations.db')

   # Import conversations from a file and store them in the database
   manager.import_conversations(file_path='tests/data/conversations.json')
   ```

2. **Process Conversations**

   ```
   # Process all conversations to generate and store embeddings
   manager.process_conversations()
   ```

3. **Query Conversations**

   ```
   # Perform a query to find similar conversations
   query_text = "What did we discuss about AI?"
   results = manager.query(query_text)

   for result in results:
       print(result)
   ```

## Project Structure

```
ConvoRAG/
├── convorag/
│   ├── __init__.py
│   ├── manager.py
│   ├── database/
│   │   ├── __init__.py
│   │   ├── database.py
│   ├── embedding/
│   │   ├── __init__.py
│   │   ├── embedding.py
│   ├── importer/
│   │   ├── __init__.py
│   │   ├── chatgpt_extractor.py
│   │   ├── importer.py
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── conversation_formatter.py
│   │   ├── text_preprocessor.py
│   ├── query/
│   │   ├── __init__.py
│   │   ├── query.py
│   ├── vectorstore/
│   │   ├── __init__.py
│   │   ├── vectorstore.py
├── tests/
│   ├── __init__.py
│   ├── test_query.py
├── requirements.txt
├── README.md
```

## TODO

- Improve embedding quality and experiment with different models.
- Write comprehensive docstrings for all functions and classes.
- Optimize text preprocessing for better chunking and tokenization.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgements

- [spaCy](https://spacy.io/)
- [Hugging Face](https://huggingface.co/)
- [FAISS](https://github.com/facebookresearch/faiss)

For any issues or contributions, feel free to create a pull request or open an issue on the GitHub repository.
