//! QRATUM bridge ingestion — reads the kernel daemon / UE5 state.json snapshot.
//!
//! State path: `%LOCALAPPDATA%/QRATUM/bridge/state.json` (Windows) or
//! `$HOME/.local/share/QRATUM/bridge/state.json` (Unix). The schema mirrors
//! `UQRATUMBridgeSubsystem::SnapshotAndPublish` and `qratum_kernel_daemon.py`.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CIIRBlock {
    pub phi: f32,
    pub delta_w_norm: f32,
    pub active_constraints: i32,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CRSBlock {
    pub active_stratum: i32,
    pub iss_residual: f32,
    pub stratum_count: i32,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct QuaSimBlock {
    pub s: f32,
    pub quantum_bound: bool,
    pub samples: i32,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct QuBICBlock {
    pub epoch: i64,
    pub nodes: i32,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RMHDBlock {
    pub reconnection_rate: f32,
    pub psi_flux: f32,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MVRIBlock {
    pub ema: f32,
    pub gate_mask: i32,
    pub status: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RWBBlock {
    pub sequence: i64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CGLBlock {
    pub lambda_min: f32,
    pub rank_deficient: bool,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BridgeState {
    pub phase: String,
    pub epoch_tick: i64,
    pub wall_clock_seconds: f64,
    pub ciir: CIIRBlock,
    pub crs: CRSBlock,
    pub ric_dispatch_weights: Vec<f32>,
    pub quasim: QuaSimBlock,
    pub qubic: QuBICBlock,
    pub rmhd: RMHDBlock,
    pub mvri: MVRIBlock,
    pub rwb: RWBBlock,
    pub cgl: CGLBlock,
}

pub fn bridge_dir() -> PathBuf {
    if cfg!(target_os = "windows") {
        let base = std::env::var("LOCALAPPDATA")
            .or_else(|_| std::env::var("USERPROFILE"))
            .unwrap_or_else(|_| ".".into());
        PathBuf::from(base).join("QRATUM").join("bridge")
    } else {
        let base = std::env::var("XDG_DATA_HOME")
            .or_else(|_| std::env::var("HOME").map(|h| format!("{}/.local/share", h)))
            .unwrap_or_else(|_| ".".into());
        PathBuf::from(base).join("QRATUM").join("bridge")
    }
}

pub fn state_path() -> PathBuf {
    bridge_dir().join("state.json")
}

pub fn boot_log_path() -> PathBuf {
    bridge_dir().join("boot.log")
}

#[tauri::command]
pub async fn qratum_bridge_state() -> Result<BridgeState, String> {
    let p = state_path();
    let raw = std::fs::read_to_string(&p)
        .map_err(|e| format!("read {}: {}", p.display(), e))?;
    serde_json::from_str::<BridgeState>(&raw).map_err(|e| format!("parse: {}", e))
}

#[tauri::command]
pub async fn qratum_bridge_boot_log() -> Result<String, String> {
    let p = boot_log_path();
    std::fs::read_to_string(&p).map_err(|e| format!("read {}: {}", p.display(), e))
}

#[tauri::command]
pub async fn qratum_bridge_path() -> Result<String, String> {
    Ok(state_path().to_string_lossy().to_string())
}

// --------------------------------------------------------------------------
// Boot management — spawn / halt the headless kernel daemon.
// --------------------------------------------------------------------------

#[derive(Default)]
pub struct DaemonHandle(pub Mutex<Option<Child>>);

fn resolve_daemon_script() -> PathBuf {
    if let Ok(env) = std::env::var("QRATUM_KERNEL_DAEMON") {
        return PathBuf::from(env);
    }
    if let Ok(exe) = std::env::current_exe() {
        let mut p = exe;
        for _ in 0..6 {
            if !p.pop() {
                break;
            }
            let candidate = p.join("qratum_kernel").join("qratum_kernel_daemon.py");
            if candidate.exists() {
                return candidate;
            }
        }
    }
    PathBuf::from("qratum_kernel/qratum_kernel_daemon.py")
}

fn resolve_python() -> String {
    std::env::var("QRATUM_PYTHON").unwrap_or_else(|_| {
        if cfg!(target_os = "windows") {
            "python".into()
        } else {
            "python3".into()
        }
    })
}

#[tauri::command]
pub async fn qratum_boot_kernel(
    handle: tauri::State<'_, DaemonHandle>,
) -> Result<String, String> {
    let mut guard = handle.0.lock().map_err(|e| e.to_string())?;
    if let Some(child) = guard.as_mut() {
        match child.try_wait() {
            Ok(None) => return Ok(format!("already running pid={}", child.id())),
            _ => *guard = None,
        }
    }

    let script = resolve_daemon_script();
    if !script.exists() {
        return Err(format!("kernel daemon not found: {}", script.display()));
    }
    let py = resolve_python();
    let child = Command::new(&py)
        .arg(&script)
        .arg("--hz")
        .arg("60")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|e| format!("spawn {} {}: {}", py, script.display(), e))?;
    let pid = child.id();
    *guard = Some(child);
    Ok(format!("kernel booted pid={}", pid))
}

#[tauri::command]
pub async fn qratum_halt_kernel(
    handle: tauri::State<'_, DaemonHandle>,
) -> Result<String, String> {
    let mut guard = handle.0.lock().map_err(|e| e.to_string())?;
    if let Some(mut child) = guard.take() {
        let _ = child.kill();
        let _ = child.wait();
        return Ok("halted".into());
    }
    Ok("not running".into())
}

#[tauri::command]
pub async fn qratum_kernel_status(
    handle: tauri::State<'_, DaemonHandle>,
) -> Result<String, String> {
    let mut guard = handle.0.lock().map_err(|e| e.to_string())?;
    match guard.as_mut() {
        None => Ok("offline".into()),
        Some(child) => match child.try_wait() {
            Ok(None) => Ok(format!("running pid={}", child.id())),
            Ok(Some(status)) => {
                *guard = None;
                Ok(format!("exited code={:?}", status.code()))
            }
            Err(e) => Err(e.to_string()),
        },
    }
}
