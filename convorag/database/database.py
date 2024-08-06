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
                update_time REAL,
                has_embeddings INTEGER DEFAULT 0 CHECK (has_embeddings IN (0, 1))
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
                parent_id INTEGER,
                embedding BLOB,
                FOREIGN KEY(conversation_id) REFERENCES conversations(id)
                FOREIGN KEY(parent_id) REFERENCES embeddings(id)
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
    
    def conversation_has_embeddings(self, conversation_id):
        cursor = self.conn.cursor()
        cursor.execute('SELECT has_embeddings FROM conversations WHERE id = ?',
                       (conversation_id,))
        result = cursor.fetchone()
        if result:
            return result[0] == 1
        return False

    def mark_conversation_as_embedded(self, conversation_id):
        with self.conn:
            self.conn.execute('UPDATE conversations SET has_embeddings = 1 '
                              'WHERE id = ?', (conversation_id,))
        
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
    
    def get_conversation_id_by_embedding_id(self, embedding_id):
        cursor = self.conn.execute('SELECT conversation_id FROM embeddings WHERE id = ?',
                                   (embedding_id,))
        result = cursor.fetchone()
        if result:
            return result[0]
        return None

    def read_embedding_ids(self):
        cursor = self.conn.execute('SELECT id FROM embeddings')
        return [row[0] for row in cursor.fetchall()]

    def get_embedding_by_id(self, embedding_id):
        cursor = self.conn.execute('SELECT * FROM embeddings WHERE id = ?',
                                   (embedding_id,))
        result = cursor.fetchone()
        embedding_np = np.frombuffer(
            result[3], dtype=np.float32
        ).reshape(1, -1)
        if result:
            return {
                'id': result[0],
                'conversation_id': result[1],
                'parent_id': result[2],
                'embedding': embedding_np
            }
        return None
    
    def get_embedding_ids_by_conversation_id(self, conversation_id):
        cursor = self.conn.execute('SELECT id FROM embeddings WHERE conversation_id = ?',
                                   (conversation_id,))
        return [row[0] for row in cursor.fetchall()]

    def store_embeddings(self, conversation_id, embeddings):
        with self.conn:
            parent_id = None
            for embedding in embeddings:
                self.conn.execute('''
                INSERT INTO embeddings (conversation_id, parent_id, embedding)
                VALUES (?, ?, ?)
                ''', (conversation_id, parent_id, embedding.tobytes()))
                parent_id = self.conn.execute('SELECT last_insert_rowid()').fetchone()[0]
        self.mark_conversation_as_embedded(conversation_id)
