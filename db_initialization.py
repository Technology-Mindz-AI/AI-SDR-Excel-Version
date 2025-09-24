import sqlite3

# Database path
DB_PATH = "queue.db"

# --- DB Setup ---


def init_db(logger):
    logger.info(
        "[init_db] Initializing the call_queue and customer_data databases and ensuring schema.")
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    # Call queue for process management
    # Call queue table
    c.execute('''
        CREATE TABLE IF NOT EXISTS call_queue (
            call_id INTEGER PRIMARY KEY AUTOINCREMENT,
            customer_name TEXT,
            customer_id TEXT,
            phone_number TEXT UNIQUE,
            email TEXT,
            customer_requirements TEXT,
            specific_prompt TEXT,
            to_call TEXT,
            notes TEXT,
            tasks TEXT,
            status TEXT DEFAULT 'queued',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            called_at TIMESTAMP
        )
    ''')

    # Customer data table
    c.execute('''
        CREATE TABLE IF NOT EXISTS customer_data (
            call_id INTEGER PRIMARY KEY,
            customer_id TEXT,
            customer_name TEXT,
            phone_number TEXT UNIQUE,
            email TEXT,
            customer_requirements TEXT,
            specific_prompt TEXT,
            last_call_status TEXT,
            country_code TEXT,
            industry TEXT,
            company_name TEXT,
            location TEXT,
            to_call TEXT,
            notes TEXT,
            tasks TEXT
        )
    ''')
    conn.commit()
    conn.close()

# sqlite connection closed.
