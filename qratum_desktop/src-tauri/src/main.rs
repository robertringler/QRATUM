// Prevents additional console window on Windows in release builds
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod backend;
mod commands;
mod tray;

use anyhow::{Context, Result};
use backend::database::Database;
use std::sync::{Arc, Mutex};
use tauri::Manager;

// Application state
pub struct AppState {
    db: Arc<Mutex<Database>>,
}

fn main() -> Result<()> {
    // Initialize logging
    env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Info)
        .init();

    log::info!("Starting QRATUM Desktop Edition");
    log::info!("Version: {}", env!("CARGO_PKG_VERSION"));
    log::info!("Platform: {}", std::env::consts::OS);

    // Initialize backend
    let app_result = tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .setup(|app| {
            log::info!("Setting up application...");

            // Get app data directory
            let app_dir = app.path().app_data_dir()
                .context("Failed to get app data directory")?;
            
            // Create directory if it doesn't exist
            std::fs::create_dir_all(&app_dir)
                .context("Failed to create app data directory")?;
            
            // Initialize database
            let db_path = app_dir.join("qratum.db");
            log::info!("Database path: {:?}", db_path);
            
            let db = Database::new(db_path)
                .context("Failed to initialize database")?;
            
            // Initialize kernel
            backend::kernel::initialize()
                .context("Failed to initialize kernel")?;
            
            // Store database in app state
            app.manage(AppState {
                db: Arc::new(Mutex::new(db)),
            });
            
            // Create system tray
            tray::create_tray(app.handle())
                .context("Failed to create system tray")?;
            
            log::info!("Application setup complete");
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            commands::get_health_status,
            commands::get_cpu_usage,
            commands::get_memory_usage,
            commands::get_disk_usage,
            commands::get_kernel_status,
            commands::execute_computation,
            commands::get_app_info,
        ])
        .run(tauri::generate_context!());

    match app_result {
        Ok(_) => {
            log::info!("Application exited normally");
            Ok(())
        }
        Err(e) => {
            log::error!("Application error: {}", e);
            Err(anyhow::anyhow!("Application error: {}", e))
        }
    }
}
