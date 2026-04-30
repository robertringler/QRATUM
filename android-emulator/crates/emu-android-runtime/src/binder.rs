//! Binder IPC emulation.
//!
//! Implements Android's Binder inter-process communication mechanism.

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use crate::error::{RuntimeError, RuntimeResult};

/// Unique handle for a Binder object.
pub type BinderHandle = u64;

/// Binder transaction code.
pub type TransactionCode = u32;

/// Standard Binder transaction codes.
pub mod transaction_codes {
    /// First transaction code for app-specific use.
    pub const FIRST_CALL_TRANSACTION: u32 = 0x0000_0001;
    /// Last transaction code for app-specific use.
    pub const LAST_CALL_TRANSACTION: u32 = 0x00FF_FFFF;
    /// Ping transaction.
    pub const PING_TRANSACTION: u32 = (b'_' as u32) << 24 | (b'P' as u32) << 16 | (b'N' as u32) << 8 | b'G' as u32;
    /// Dump transaction.
    pub const DUMP_TRANSACTION: u32 = (b'_' as u32) << 24 | (b'D' as u32) << 16 | (b'M' as u32) << 8 | b'P' as u32;
    /// Shell command transaction.
    pub const SHELL_COMMAND_TRANSACTION: u32 = (b'_' as u32) << 24 | (b'C' as u32) << 16 | (b'M' as u32) << 8 | b'D' as u32;
    /// Interface transaction.
    pub const INTERFACE_TRANSACTION: u32 = (b'_' as u32) << 24 | (b'N' as u32) << 16 | (b'T' as u32) << 8 | b'F' as u32;
}

/// Binder message for IPC.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BinderMessage {
    /// Target handle.
    pub target: BinderHandle,
    /// Transaction code.
    pub code: TransactionCode,
    /// Message flags.
    pub flags: u32,
    /// Message data.
    pub data: Vec<u8>,
    /// File descriptors (handles).
    pub fds: Vec<i32>,
}

impl BinderMessage {
    /// Create a new empty message.
    pub fn new(target: BinderHandle, code: TransactionCode) -> Self {
        Self {
            target,
            code,
            flags: 0,
            data: Vec::new(),
            fds: Vec::new(),
        }
    }

    /// Add data to the message.
    pub fn write_data(&mut self, data: &[u8]) {
        self.data.extend_from_slice(data);
    }

    /// Write a 32-bit integer.
    pub fn write_i32(&mut self, value: i32) {
        self.data.extend_from_slice(&value.to_le_bytes());
    }

    /// Write a 64-bit integer.
    pub fn write_i64(&mut self, value: i64) {
        self.data.extend_from_slice(&value.to_le_bytes());
    }

    /// Write a string (with length prefix).
    pub fn write_string(&mut self, s: &str) {
        let bytes = s.as_bytes();
        self.write_i32(bytes.len() as i32);
        self.data.extend_from_slice(bytes);
        // Align to 4 bytes
        while self.data.len() % 4 != 0 {
            self.data.push(0);
        }
    }
}

/// A Binder service node.
#[derive(Debug, Clone)]
pub struct BinderNode {
    /// Node handle.
    pub handle: BinderHandle,
    /// Service name.
    pub name: String,
    /// Interface name.
    pub interface: String,
    /// Whether the node is alive.
    pub alive: bool,
}

impl BinderNode {
    /// Create a new Binder node.
    pub fn new(handle: BinderHandle, name: impl Into<String>, interface: impl Into<String>) -> Self {
        Self {
            handle,
            name: name.into(),
            interface: interface.into(),
            alive: true,
        }
    }
}

/// Binder service trait.
pub trait BinderService: Send + Sync {
    /// Get the interface descriptor.
    fn interface_descriptor(&self) -> &str;
    
    /// Handle a transaction.
    fn on_transact(&self, code: TransactionCode, data: &[u8], reply: &mut Vec<u8>) -> RuntimeResult<()>;
}

/// Binder driver emulation.
pub struct BinderDriver {
    /// Registered services by name.
    services: RwLock<HashMap<String, Arc<dyn BinderService>>>,
    /// Nodes by handle.
    nodes: RwLock<HashMap<BinderHandle, BinderNode>>,
    /// Next handle to allocate.
    next_handle: AtomicU64,
}

impl BinderDriver {
    /// Create a new Binder driver.
    pub fn new() -> Self {
        Self {
            services: RwLock::new(HashMap::new()),
            nodes: RwLock::new(HashMap::new()),
            next_handle: AtomicU64::new(1),
        }
    }

    /// Register a service.
    pub fn register_service(&self, name: &str, service: Arc<dyn BinderService>) -> RuntimeResult<BinderHandle> {
        let handle = self.next_handle.fetch_add(1, Ordering::SeqCst);
        
        let node = BinderNode::new(handle, name, service.interface_descriptor());
        
        self.nodes.write().insert(handle, node);
        self.services.write().insert(name.to_string(), service);
        
        log::info!("Registered Binder service: {} (handle {})", name, handle);
        Ok(handle)
    }

    /// Look up a service by name.
    pub fn get_service(&self, name: &str) -> Option<BinderHandle> {
        let nodes = self.nodes.read();
        for node in nodes.values() {
            if node.name == name && node.alive {
                return Some(node.handle);
            }
        }
        None
    }

    /// Send a transaction.
    pub fn transact(&self, msg: &BinderMessage) -> RuntimeResult<Vec<u8>> {
        let nodes = self.nodes.read();
        let node = nodes.get(&msg.target).ok_or_else(|| RuntimeError::BinderError {
            reason: format!("unknown handle: {}", msg.target),
        })?;
        
        if !node.alive {
            return Err(RuntimeError::BinderError {
                reason: "service is dead".to_string(),
            });
        }
        
        let services = self.services.read();
        let service = services.get(&node.name).ok_or_else(|| RuntimeError::BinderError {
            reason: format!("service not found: {}", node.name),
        })?;
        
        let mut reply = Vec::new();
        service.on_transact(msg.code, &msg.data, &mut reply)?;
        
        Ok(reply)
    }

    /// List all registered services.
    pub fn list_services(&self) -> Vec<String> {
        self.nodes.read().values().filter(|n| n.alive).map(|n| n.name.clone()).collect()
    }
}

impl Default for BinderDriver {
    fn default() -> Self {
        Self::new()
    }
}

/// Service Manager service.
pub struct ServiceManager {
    driver: Arc<BinderDriver>,
}

impl ServiceManager {
    /// Create a new Service Manager.
    pub fn new(driver: Arc<BinderDriver>) -> Self {
        Self { driver }
    }
}

impl BinderService for ServiceManager {
    fn interface_descriptor(&self) -> &str {
        "android.os.IServiceManager"
    }

    fn on_transact(&self, code: TransactionCode, data: &[u8], reply: &mut Vec<u8>) -> RuntimeResult<()> {
        match code {
            // GET_SERVICE_TRANSACTION = 1
            1 => {
                // Parse service name from data
                if data.len() >= 4 {
                    let name_len = i32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;
                    if data.len() >= 4 + name_len {
                        let name = String::from_utf8_lossy(&data[4..4 + name_len]);
                        if let Some(handle) = self.driver.get_service(&name) {
                            reply.extend_from_slice(&(handle as i64).to_le_bytes());
                            return Ok(());
                        }
                    }
                }
                // Return null if not found
                reply.extend_from_slice(&0i64.to_le_bytes());
                Ok(())
            }
            // LIST_SERVICES_TRANSACTION = 4
            4 => {
                let services = self.driver.list_services();
                reply.extend_from_slice(&(services.len() as i32).to_le_bytes());
                for service in services {
                    let bytes = service.as_bytes();
                    reply.extend_from_slice(&(bytes.len() as i32).to_le_bytes());
                    reply.extend_from_slice(bytes);
                    while reply.len() % 4 != 0 {
                        reply.push(0);
                    }
                }
                Ok(())
            }
            _ => Ok(()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_binder_message() {
        let mut msg = BinderMessage::new(1, transaction_codes::FIRST_CALL_TRANSACTION);
        msg.write_i32(42);
        msg.write_string("test");
        
        assert_eq!(msg.target, 1);
        assert!(!msg.data.is_empty());
    }

    #[test]
    fn test_binder_driver() {
        let driver = Arc::new(BinderDriver::new());
        
        // Register service manager
        let sm = Arc::new(ServiceManager::new(driver.clone()));
        let handle = driver.register_service("android.os.IServiceManager", sm).unwrap();
        
        // Look up the service
        let found = driver.get_service("android.os.IServiceManager");
        assert_eq!(found, Some(handle));
    }
}
