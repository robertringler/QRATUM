use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::sync::{Arc, Mutex};
use tokio::runtime::Runtime;
use tungstenite::connect;
use url::Url;

// -- 1. Internal State Structures --
#[derive(serde::Deserialize, serde::Serialize, Clone, Default)]
struct QradleState {
    epoch: u64,
    validator_zone_heatmap: [f32; 4], // Z0-Z3
    slashing_vector: f32,
    latest_zk_proof: String,
}

lazy_static::lazy_static! {
    static ref GLOBAL_STATE: Arc<Mutex<QradleState>> = Arc::new(Mutex::new(QradleState::default()));
    static ref RUNTIME: Runtime = Runtime::new().unwrap();
}

// -- 2. Background Telemetry Loop --
fn start_telemetry_stream(url_str: String) {
    RUNTIME.spawn(async move {
        let (mut socket, _) = connect(Url::parse(&url_str).unwrap()).expect("Can't connect to Aethernet");
        
        loop {
            if let Ok(msg) = socket.read() {
                if let Ok(text) = msg.to_text() {
                    // Zero-copy parsing could be added here for optimization
                    if let Ok(new_state) = serde_json::from_str::<QradleState>(text) {
                        let mut lock = GLOBAL_STATE.lock().unwrap();
                        *lock = new_state;
                    }
                }
            }
        }
    });
}

// -- 3. The FFI Bridge (Callable from C++ Unreal) --

#[no_mangle]
pub extern "C" fn soi_initialize(endpoint: *const c_char) {
    let c_str = unsafe { CStr::from_ptr(endpoint) };
    let url = c_str.to_string_lossy().into_owned();
    start_telemetry_stream(url);
}

#[no_mangle]
pub extern "C" fn soi_get_epoch() -> u64 {
    GLOBAL_STATE.lock().unwrap().epoch
}

#[no_mangle]
pub extern "C" fn soi_get_zone_heat(zone_idx: usize) -> f32 {
    let state = GLOBAL_STATE.lock().unwrap();
    if zone_idx < 4 { state.validator_zone_heatmap[zone_idx] } else { 0.0 }
}

#[no_mangle]
pub extern "C" fn soi_get_slashing_vector() -> f32 {
    GLOBAL_STATE.lock().unwrap().slashing_vector
}

#[no_mangle]
pub extern "C" fn soi_get_proof(buffer: *mut c_char, length: usize) {
    let state = GLOBAL_STATE.lock().unwrap();
    let c_str = CString::new(state.latest_zk_proof.clone()).unwrap();
    // Safety: In production, use strict buffer copying routines here
    unsafe {
        let bytes = c_str.as_bytes_with_nul();
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), buffer as *mut u8, std::cmp::min(bytes.len(), length));
    }
}

// -- 4. Additional Helper Functions --

/// Get the current status as a JSON string
#[no_mangle]
pub extern "C" fn soi_get_status_json(buffer: *mut c_char, length: usize) -> i32 {
    let state = GLOBAL_STATE.lock().unwrap();
    let json = serde_json::to_string(&*state).unwrap_or_else(|_| "{}".to_string());
    let c_str = CString::new(json).unwrap();
    
    unsafe {
        let bytes = c_str.as_bytes_with_nul();
        let copy_len = std::cmp::min(bytes.len(), length);
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), buffer as *mut u8, copy_len);
        copy_len as i32
    }
}

/// Check if the telemetry system is initialized
#[no_mangle]
pub extern "C" fn soi_is_initialized() -> bool {
    true // Simplified - in production would check connection state
}

/// Shutdown the telemetry system gracefully
#[no_mangle]
pub extern "C" fn soi_shutdown() {
    // In production, this would gracefully close the websocket connection
    // and cleanup resources
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_state() {
        let state = QradleState::default();
        assert_eq!(state.epoch, 0);
        assert_eq!(state.validator_zone_heatmap, [0.0, 0.0, 0.0, 0.0]);
        assert_eq!(state.slashing_vector, 0.0);
        assert_eq!(state.latest_zk_proof, "");
    }
}
