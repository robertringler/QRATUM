#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::sync::{Arc, Mutex};
use tauri::{CustomMenuItem, Manager, SystemTray, SystemTrayMenu};

mod backend;
mod codegen;
mod commands;
mod qr_os_supreme;
mod tray;

// Lightweight in-memory database (no SQLite)
#[derive(Default)]
pub struct AppState {
    logs: Arc<Mutex<Vec<backend::LogEntry>>>,
}

fn main() {
    // System tray setup
    let tray_menu = SystemTrayMenu::new()
        .add_item(CustomMenuItem::new("show".to_string(), "Show"))
        .add_item(CustomMenuItem::new("hide".to_string(), "Hide"))
        .add_native_item(tauri::SystemTrayMenuItem::Separator)
        .add_item(CustomMenuItem::new("quit".to_string(), "Quit"));

    let tray = SystemTray::new().with_menu(tray_menu);

    let app = tauri::Builder::<tauri::Wry>::default()
        .manage(AppState::default())
        .system_tray(tray)
        .on_system_tray_event(tray::handle_tray_event)
        .invoke_handler(tauri::generate_handler![
            // Core commands
            commands::get_health,
            commands::execute_kernel,
            commands::get_logs,
            commands::generate_code,
            commands::validate_code,
            // Quantum simulation
            commands::run_bell_state,
            commands::run_quantum_teleportation,
            commands::run_ghz_state,
            commands::get_quantum_state,
            commands::apply_quantum_gate,
            // AI inference
            commands::run_ai_inference,
            commands::classify_text,
            commands::embed_text,
            // Combined operations
            commands::run_supremacy_test,
            commands::get_os_supreme_stats,
            commands::get_pod_config,
            // DCGE benchmarking
            commands::run_dcge_benchmark,
            commands::get_binary_metrics,
            commands::get_failure_modes,
        ])
        .build(tauri::generate_context!())
        .expect("error while building tauri application");

    app.run(|_app_handle, event| match event {
        tauri::RunEvent::ExitRequested { api, .. } => {
            api.prevent_exit();
        }
        _ => {}
    });
}
