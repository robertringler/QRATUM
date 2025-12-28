use anyhow::{Context, Result};
use rusqlite::{Connection, params};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

pub struct Database {
    conn: Arc<Mutex<Connection>>,
}

impl Database {
    /// Create a new database connection
    pub fn new(db_path: PathBuf) -> Result<Self> {
        let conn = Connection::open(&db_path)
            .context("Failed to open SQLite database")?;
        
        let db = Database {
            conn: Arc::new(Mutex::new(conn)),
        };
        
        db.initialize_schema()?;
        log::info!("Database initialized at {:?}", db_path);
        
        Ok(db)
    }
    
    /// Initialize database schema
    fn initialize_schema(&self) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        
        conn.execute(
            "CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL,
                data TEXT
            )",
            [],
        )?;
        
        conn.execute(
            "CREATE TABLE IF NOT EXISTS computations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                type TEXT NOT NULL,
                status TEXT NOT NULL,
                input TEXT,
                output TEXT,
                created_at INTEGER NOT NULL,
                completed_at INTEGER,
                FOREIGN KEY(session_id) REFERENCES sessions(id)
            )",
            [],
        )?;
        
        conn.execute(
            "CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at INTEGER NOT NULL
            )",
            [],
        )?;
        
        log::info!("Database schema initialized");
        Ok(())
    }
    
    /// Store a setting
    pub fn set_setting(&self, key: &str, value: &str) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_secs() as i64;
        
        conn.execute(
            "INSERT OR REPLACE INTO settings (key, value, updated_at) VALUES (?1, ?2, ?3)",
            params![key, value, now],
        )?;
        
        Ok(())
    }
    
    /// Get a setting
    pub fn get_setting(&self, key: &str) -> Result<Option<String>> {
        let conn = self.conn.lock().unwrap();
        
        let mut stmt = conn.prepare("SELECT value FROM settings WHERE key = ?1")?;
        let mut rows = stmt.query(params![key])?;
        
        if let Some(row) = rows.next()? {
            let value: String = row.get(0)?;
            Ok(Some(value))
        } else {
            Ok(None)
        }
    }
    
    /// Create a new session
    pub fn create_session(&self, name: &str) -> Result<i64> {
        let conn = self.conn.lock().unwrap();
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_secs() as i64;
        
        conn.execute(
            "INSERT INTO sessions (name, created_at, updated_at) VALUES (?1, ?2, ?3)",
            params![name, now, now],
        )?;
        
        Ok(conn.last_insert_rowid())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn test_database_creation() {
        let temp_dir = std::env::temp_dir();
        let db_path = temp_dir.join("test_qratum.db");
        
        // Clean up if exists
        let _ = fs::remove_file(&db_path);
        
        let result = Database::new(db_path.clone());
        assert!(result.is_ok());
        
        // Clean up
        let _ = fs::remove_file(&db_path);
    }
    
    #[test]
    fn test_settings() {
        let temp_dir = std::env::temp_dir();
        let db_path = temp_dir.join("test_qratum_settings.db");
        
        // Clean up if exists
        let _ = fs::remove_file(&db_path);
        
        let db = Database::new(db_path.clone()).unwrap();
        
        // Set and get setting
        db.set_setting("test_key", "test_value").unwrap();
        let value = db.get_setting("test_key").unwrap();
        assert_eq!(value, Some("test_value".to_string()));
        
        // Clean up
        let _ = fs::remove_file(&db_path);
    }
}
