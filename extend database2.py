import sqlite3

# Connect to the SQLite database (or create it if it doesn't exist)
conn = sqlite3.connect("chatbot.db")
cursor = conn.cursor()

# Create the conversation_history table
cursor.execute("""
CREATE TABLE IF NOT EXISTS conversation_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_input TEXT NOT NULL,
    bot_response TEXT NOT NULL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
)
""")

# Create the entities table
cursor.execute("""
CREATE TABLE IF NOT EXISTS entities (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    properties TEXT,
    connections TEXT
)
""")

# Create the knowledge_base table
cursor.execute("""
CREATE TABLE IF NOT EXISTS knowledge_base (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    concept TEXT NOT NULL,
    data TEXT
)
""")

# Create the neurons table
cursor.execute("""
CREATE TABLE IF NOT EXISTS neurons (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    threshold REAL NOT NULL,
    decay REAL NOT NULL
)
""")

# Create the synapses table
cursor.execute("""
CREATE TABLE IF NOT EXISTS synapses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    weight REAL NOT NULL,
    learning_rate REAL NOT NULL
)
""")

# Commit the changes and close the connection
conn.commit()
conn.close()

print("Database and tables created successfully!")