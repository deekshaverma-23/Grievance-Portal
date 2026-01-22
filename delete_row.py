import sqlite3

conn = sqlite3.connect("grievances.db")
cur = conn.cursor()

row_id = 30   #change this to the ID you want to delete

cur.execute("DELETE FROM complaints WHERE id = ?", (row_id,))

conn.commit()
conn.close()

print(f"Deleted row with ID {row_id}")
