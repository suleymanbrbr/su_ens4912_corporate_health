import re

def refactor_file(filename):
    with open(filename, 'r') as f:
        content = f.read()

    # Imports
    content = content.replace('import sqlite3', 'import psycopg2\nimport psycopg2.extras\nfrom psycopg2.extras import Json\nimport os')
    
    # Connection
    content = content.replace('sqlite3.connect(DB_PATH)', 'psycopg2.connect(dsn=os.getenv("DATABASE_URL"), cursor_factory=psycopg2.extras.DictCursor)')
    content = content.replace('sqlite3.connect(DB_PATH, check_same_thread=False)', 'psycopg2.connect(dsn=os.getenv("DATABASE_URL"), cursor_factory=psycopg2.extras.DictCursor)')
    content = content.replace('conn.row_factory = sqlite3.Row', '# conn.row_factory is defined by cursor_factory')

    # Tables
    content = content.replace('CREATE VIRTUAL TABLE IF NOT EXISTS title_search USING fts5(chunk_id, header_text)', 'CREATE INDEX IF NOT EXISTS chunks_fts_idx ON chunks USING GIN (to_tsvector(''turkish'', header_text || '' '' || text_content));')
    
    # Types
    content = content.replace('INTEGER DEFAULT 0', 'SMALLINT DEFAULT 0')
    content = content.replace('INTEGER DEFAULT 1', 'SMALLINT DEFAULT 1')
    
    # Replaces ? with %s intelligently where it looks like a parameter
    content = re.sub(r"execute\((.*?),\s*(.*?\))", lambda m: "execute(" + m.group(1).replace('?', '%s') + ", " + m.group(2) if '?' in m.group(1) else m.group(0), content, flags=re.DOTALL)
    
    # Some specific replacements for FTS
    content = content.replace('FROM title_search WHERE header_text MATCH %s', "FROM chunks WHERE to_tsvector('turkish', header_text) @@ plainto_tsquery('turkish', %s)")
    
    with open(filename, 'w') as f:
        f.write(content)

refactor_file('backend/api_server.py')
