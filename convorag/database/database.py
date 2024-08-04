import sqlite3
import numpy as np

class Database:
    def __init__(self, db_path):
        self.conn = sqlite3.connect(db_path)
        self._create_tables()

    def _create_tables(self):
        with self.conn:
            self.conn.execute('''
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                title TEXT,
                create_time REAL,
                update_time REAL
            );
            ''')

            self.conn.execute('''
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                conversation_id TEXT,
                parent_id TEXT,
                create_time REAL,
                author_role TEXT,
                content_type TEXT,
                content TEXT,
                language TEXT,
                metadata TEXT,
                model_type TEXT,
                FOREIGN KEY (conversation_id) REFERENCES conversations (id),
                FOREIGN KEY (parent_id) REFERENCES messages (id)
            );
            ''')
            self.conn.execute('''
            CREATE TABLE IF NOT EXISTS embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER,
                embedding BLOB,
                FOREIGN KEY(conversation_id) REFERENCES conversations(id)
            )
            ''')

    def insert_conversations(self, conversations):
        with self.conn:
            for conv in conversations:
                self.conn.execute('''
                INSERT OR IGNORE INTO conversations (id, title, create_time, update_time)
                VALUES (?, ?, ?, ?)
                ''', (conv['id'], conv['title'], conv['create_time'], conv['update_time']))

    def insert_messages(self, messages):
        with self.conn:
            for msg in messages:
                self.conn.execute('''
                INSERT OR IGNORE INTO messages (id, conversation_id, parent_id, create_time, author_role, content_type, content, language, metadata, model_type)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    msg['id'], msg['conversation_id'], msg['parent_id'], msg['create_time'],
                    msg['author_role'], msg['content_type'], msg['content'], msg['language'], msg['metadata'],
                    msg['model_type']
                ))

    def read_conversations(self, start_datetime=None, end_datetime=None):
        query = "SELECT * FROM conversations"
        params = []
        conditions = []
        if start_datetime:
            conditions.append("create_time >= ?")
            params.append(start_datetime.timestamp())
        if end_datetime:
            conditions.append("create_time <= ?")
            params.append(end_datetime.timestamp())
        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        cursor = self.conn.execute(query, params)
        columns = [description[0] for description in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def read_messages(self, start_datetime=None, end_datetime=None):
        query = "SELECT * FROM messages"
        params = []
        conditions = []
        if start_datetime:
            conditions.append("create_time >= ?")
            params.append(start_datetime.timestamp())
        if end_datetime:
            conditions.append("create_time <= ?")
            params.append(end_datetime.timestamp())
        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        cursor = self.conn.execute(query, params)
        columns = [description[0] for description in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def read_messages_by_conversation_id(self, conversation_id):
        query = "SELECT * FROM messages WHERE conversation_id = ?"
        params = [conversation_id]

        cursor = self.conn.execute(query, params)
        columns = [description[0] for description in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def read_last_n_conversations(self, n):
        query = "SELECT * FROM conversations ORDER BY create_time DESC LIMIT ?"
        params = [n]

        cursor = self.conn.execute(query, params)
        columns = [description[0] for description in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def store_embedding(self, conversation_id, embedding):
        with self.conn:
            self.conn.execute('''
                INSERT INTO embeddings (conversation_id, embedding) VALUES (?, ?)
            ''', (conversation_id, embedding.tobytes()))
    
    def has_embedding(self, conversation_id):
        with self.conn:
            result = self.conn.execute('''
                SELECT EXISTS(
                    SELECT 1 FROM embeddings WHERE conversation_id=?
                )
            ''', (conversation_id,)).fetchone()
            return result[0] == 1
    
    def get_embedding_by_conversation_id(self, conversation_id):
        with self.conn:
            result = self.conn.execute('''
                SELECT embedding FROM embeddings WHERE conversation_id=?
            ''', (conversation_id,)).fetchone()
            return np.frombuffer(result[0], dtype=np.float32) if result else None
        
    def get_conversation_by_id(self, conversation_id):
        with self.conn:
            result = self.conn.execute('''
                SELECT * FROM conversations WHERE id=?
            ''', (conversation_id,)).fetchone()
            if result:
                return {
                    'id': result[0],
                    'title': result[1]
                }
            return None
