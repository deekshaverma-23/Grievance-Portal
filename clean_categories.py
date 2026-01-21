import sqlite3

conn = sqlite3.connect('grievances.db')
cur = conn.cursor()

# Remove anything after newline
cur.execute("""
UPDATE complaints
SET category = substr(category, 1, instr(category, char(10)) - 1)
WHERE instr(category, char(10)) > 0
""")

conn.commit()
conn.close()
print("Category cleanup complete!")
