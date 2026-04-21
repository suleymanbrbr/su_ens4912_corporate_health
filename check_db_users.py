import sqlite3

conn = sqlite3.connect('data/sut_knowledge_base.db')
c = conn.cursor()
c.execute("SELECT id, username, email, is_approved, hashed_password FROM users")
for row in c.fetchall():
    print(row)
