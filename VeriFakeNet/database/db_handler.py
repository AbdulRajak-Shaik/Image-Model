import sqlite3
import os
from datetime import datetime

class DBHandler:
    def __init__(self, db_path="database/history.db"):
        self.db_path = db_path
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.init_db()

    def get_connection(self):
        return sqlite3.connect(self.db_path)

    def init_db(self):
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Create table if it doesn't exist
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                filename TEXT,
                prediction TEXT,
                confidence REAL,
                trust_score REAL,
                interpretation TEXT
            )
        ''')
        
        conn.commit()
        conn.close()

    def log_prediction(self, filename, prediction, confidence, trust_score, interpretation):
        conn = self.get_connection()
        cursor = conn.cursor()
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        cursor.execute('''
            INSERT INTO predictions (timestamp, filename, prediction, confidence, trust_score, interpretation)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (timestamp, filename, prediction, confidence, trust_score, interpretation))
        
        conn.commit()
        conn.close()

    def get_history(self, limit=50):
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT timestamp, filename, prediction, confidence, trust_score, interpretation 
            FROM predictions 
            ORDER BY id DESC 
            LIMIT ?
        ''', (limit,))
        
        rows = cursor.fetchall()
        conn.close()
        
        return rows
