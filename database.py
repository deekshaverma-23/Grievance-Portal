import sqlite3

def init_db():
    """Initializes the SQLite database and creates the complaints table."""
    conn = sqlite3.connect('grievances.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS complaints (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            complaint_text TEXT NOT NULL,
            sentiment TEXT,
            priority TEXT,
            resolution TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def save_complaint(complaint_text, sentiment, priority, resolution):
    """Saves a processed complaint to the database."""
    conn = sqlite3.connect('grievances.db')
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO complaints (complaint_text, sentiment, priority, resolution)
        VALUES (?, ?, ?, ?)
    ''', (complaint_text, sentiment, priority, resolution))
    conn.commit()
    conn.close()

def get_all_complaints():
    """Retrieves all complaints from the database."""
    conn = sqlite3.connect('grievances.db')
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM complaints ORDER BY timestamp DESC")
    complaints = cursor.fetchall()
    conn.close()
    return complaints

if __name__ == '__main__':
    init_db()