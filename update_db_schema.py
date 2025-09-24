import sqlite3
from logger_config import logger

DB_PATH = "queue.db"

def update_database_schema():
    """Update the database schema to add the specific_prompt column"""
    logger.info("[update_database_schema] Updating database schema...")
    
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    try:
        # Check if call_queue table exists
        c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='call_queue'")
        if not c.fetchone():
            logger.error("[update_database_schema] call_queue table does not exist. Please run db_initialization.py first.")
            return False
        
        # Check if customer_data table exists
        c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='customer_data'")
        if not c.fetchone():
            logger.error("[update_database_schema] customer_data table does not exist. Please run db_initialization.py first.")
            return False
        
        # Check if specific_prompt column exists in call_queue table
        c.execute("PRAGMA table_info(call_queue)")
        columns = [column[1] for column in c.fetchall()]
        
        if 'specific_prompt' not in columns:
            logger.info("[update_database_schema] Adding specific_prompt column to call_queue table")
            c.execute("ALTER TABLE call_queue ADD COLUMN specific_prompt TEXT")
        
        # Check if specific_prompt column exists in customer_data table
        c.execute("PRAGMA table_info(customer_data)")
        columns = [column[1] for column in c.fetchall()]
        
        if 'specific_prompt' not in columns:
            logger.info("[update_database_schema] Adding specific_prompt column to customer_data table")
            c.execute("ALTER TABLE customer_data ADD COLUMN specific_prompt TEXT")
        
        conn.commit()
        logger.info("[update_database_schema] Database schema updated successfully")
        return True
        
    except Exception as e:
        logger.error(f"[update_database_schema] Error updating schema: {e}")
        conn.rollback()
        return False
    finally:
        conn.close()

if __name__ == "__main__":
    update_database_schema()