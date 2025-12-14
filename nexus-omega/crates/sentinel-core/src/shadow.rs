//! Shadow identity management

use crate::error::{Result, SentinelError};
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use std::collections::HashMap;

/// Obfuscation level for shadow identities
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ObfuscationLevel {
    /// Minimal obfuscation
    Low,
    /// Standard obfuscation
    Medium,
    /// Maximum obfuscation
    High,
}

/// Shadow identity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShadowIdentity {
    /// Unique identifier
    pub id: Uuid,
    /// Obfuscation level
    pub level: ObfuscationLevel,
    /// Creation timestamp
    pub created_at: u64,
    /// Last rotation timestamp
    pub last_rotation: u64,
    /// Metadata
    pub metadata: HashMap<String, String>,
}

impl ShadowIdentity {
    /// Create a new shadow identity
    pub fn new(level: ObfuscationLevel) -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        Self {
            id: Uuid::new_v4(),
            level,
            created_at: now,
            last_rotation: now,
            metadata: HashMap::new(),
        }
    }

    /// Rotate the identity
    pub fn rotate(&mut self) {
        self.id = Uuid::new_v4();
        self.last_rotation = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
    }
}

/// Shadow identity manager
#[derive(Debug)]
pub struct ShadowManager {
    /// Active identities
    identities: HashMap<Uuid, ShadowIdentity>,
    /// Rotation interval in seconds
    rotation_interval: u64,
}

impl ShadowManager {
    /// Create a new shadow manager
    pub fn new(rotation_interval: u64) -> Self {
        Self {
            identities: HashMap::new(),
            rotation_interval,
        }
    }

    /// Create a new shadow identity
    pub fn create_identity(&mut self, level: ObfuscationLevel) -> Result<Uuid> {
        let identity = ShadowIdentity::new(level);
        let id = identity.id;
        self.identities.insert(id, identity);
        Ok(id)
    }

    /// Get a shadow identity
    pub fn get_identity(&self, id: &Uuid) -> Option<&ShadowIdentity> {
        self.identities.get(id)
    }

    /// Rotate an identity
    pub fn rotate_identity(&mut self, id: &Uuid) -> Result<()> {
        let identity = self
            .identities
            .get_mut(id)
            .ok_or_else(|| SentinelError::ShadowIdentity("Identity not found".to_string()))?;

        identity.rotate();
        Ok(())
    }

    /// Check if rotation is needed for an identity
    pub fn needs_rotation(&self, id: &Uuid) -> bool {
        if let Some(identity) = self.identities.get(id) {
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs();
            now - identity.last_rotation > self.rotation_interval
        } else {
            false
        }
    }

    /// Get all active identities
    pub fn active_identities(&self) -> Vec<Uuid> {
        self.identities.keys().copied().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shadow_identity_creation() {
        let identity = ShadowIdentity::new(ObfuscationLevel::Medium);
        assert_eq!(identity.level, ObfuscationLevel::Medium);
    }

    #[test]
    fn test_shadow_manager() {
        let mut manager = ShadowManager::new(3600);
        let id = manager.create_identity(ObfuscationLevel::High).unwrap();
        assert!(manager.get_identity(&id).is_some());
    }
}
