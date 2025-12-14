//! Pack and unpack operations for Git objects

use crate::error::{GitError, Result};
use crate::objects::GitObject;
use flate2::read::{ZlibDecoder, ZlibEncoder};
use flate2::Compression;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::Read;

/// A packfile containing multiple Git objects
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PackFile {
    pub objects: HashMap<String, PackedObject>,
}

/// A packed Git object with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PackedObject {
    pub sha: String,
    pub object_type: String,
    pub data: Vec<u8>,
    pub compressed: bool,
}

impl PackFile {
    /// Create a new empty packfile
    pub fn new() -> Self {
        Self {
            objects: HashMap::new(),
        }
    }

    /// Add an object to the packfile
    pub fn add_object(&mut self, sha: String, object: GitObject, compress: bool) -> Result<()> {
        let object_type = object.object_type().to_string();
        let data = object.to_bytes();

        let final_data = if compress {
            compress_data(&data)?
        } else {
            data
        };

        self.objects.insert(
            sha.clone(),
            PackedObject {
                sha,
                object_type,
                data: final_data,
                compressed: compress,
            },
        );

        Ok(())
    }

    /// Get an object from the packfile
    pub fn get_object(&self, sha: &str) -> Result<GitObject> {
        let packed = self
            .objects
            .get(sha)
            .ok_or_else(|| GitError::ObjectNotFound(sha.to_string()))?;

        let data = if packed.compressed {
            decompress_data(&packed.data)?
        } else {
            packed.data.clone()
        };

        GitObject::from_bytes(&packed.object_type, &data)
    }

    /// Check if a packfile contains an object
    pub fn has_object(&self, sha: &str) -> bool {
        self.objects.contains_key(sha)
    }

    /// Get the number of objects in the packfile
    pub fn object_count(&self) -> usize {
        self.objects.len()
    }

    /// List all object SHAs in the packfile
    pub fn list_objects(&self) -> Vec<String> {
        self.objects.keys().cloned().collect()
    }

    /// Serialize the packfile to JSON
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string_pretty(self)
            .map_err(|e| GitError::ParseError(format!("Failed to serialize packfile: {}", e)))
    }

    /// Deserialize a packfile from JSON
    pub fn from_json(json: &str) -> Result<Self> {
        serde_json::from_str(json)
            .map_err(|e| GitError::ParseError(format!("Failed to deserialize packfile: {}", e)))
    }

    /// Export all objects (decompressed if needed)
    pub fn export_objects(&self) -> Result<HashMap<String, GitObject>> {
        let mut result = HashMap::new();
        for (sha, _packed) in &self.objects {
            let object = self.get_object(sha)?;
            result.insert(sha.clone(), object);
        }
        Ok(result)
    }
}

impl Default for PackFile {
    fn default() -> Self {
        Self::new()
    }
}

/// Compress data using zlib
fn compress_data(data: &[u8]) -> Result<Vec<u8>> {
    let mut encoder = ZlibEncoder::new(data, Compression::default());
    let mut compressed = Vec::new();
    encoder
        .read_to_end(&mut compressed)
        .map_err(|e| GitError::CompressionError(format!("Compression failed: {}", e)))?;
    Ok(compressed)
}

/// Decompress data using zlib
fn decompress_data(data: &[u8]) -> Result<Vec<u8>> {
    let mut decoder = ZlibDecoder::new(data);
    let mut decompressed = Vec::new();
    decoder
        .read_to_end(&mut decompressed)
        .map_err(|e| GitError::CompressionError(format!("Decompression failed: {}", e)))?;
    Ok(decompressed)
}

/// Create a packfile from a list of objects
pub fn pack_objects(objects: Vec<(String, GitObject)>, compress: bool) -> Result<PackFile> {
    let mut pack = PackFile::new();
    for (sha, object) in objects {
        pack.add_object(sha, object, compress)?;
    }
    Ok(pack)
}

/// Unpack a packfile into a list of objects
pub fn unpack_objects(pack: &PackFile) -> Result<Vec<(String, GitObject)>> {
    let mut result = Vec::new();
    for sha in pack.list_objects() {
        let object = pack.get_object(&sha)?;
        result.push((sha, object));
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::objects::Blob;

    #[test]
    fn test_compress_decompress() {
        let data = b"Hello, World! This is a test of compression.".to_vec();
        let compressed = compress_data(&data).unwrap();
        let decompressed = decompress_data(&compressed).unwrap();
        assert_eq!(data, decompressed);
    }

    #[test]
    fn test_packfile_creation() {
        let pack = PackFile::new();
        assert_eq!(pack.object_count(), 0);
    }

    #[test]
    fn test_add_and_get_object() {
        let mut pack = PackFile::new();
        let blob = GitObject::Blob(Blob::new(b"test data".to_vec()));
        let sha = blob.sha();

        pack.add_object(sha.clone(), blob.clone(), false).unwrap();
        assert!(pack.has_object(&sha));

        let retrieved = pack.get_object(&sha).unwrap();
        match retrieved {
            GitObject::Blob(b) => assert_eq!(b.data, b"test data"),
            _ => panic!("Expected blob"),
        }
    }

    #[test]
    fn test_add_compressed_object() {
        let mut pack = PackFile::new();
        let blob = GitObject::Blob(Blob::new(b"test data for compression".to_vec()));
        let sha = blob.sha();

        pack.add_object(sha.clone(), blob, true).unwrap();
        assert!(pack.has_object(&sha));

        let retrieved = pack.get_object(&sha).unwrap();
        match retrieved {
            GitObject::Blob(b) => assert_eq!(b.data, b"test data for compression"),
            _ => panic!("Expected blob"),
        }
    }

    #[test]
    fn test_list_objects() {
        let mut pack = PackFile::new();
        let blob1 = GitObject::Blob(Blob::new(b"test1".to_vec()));
        let blob2 = GitObject::Blob(Blob::new(b"test2".to_vec()));
        let sha1 = blob1.sha();
        let sha2 = blob2.sha();

        pack.add_object(sha1.clone(), blob1, false).unwrap();
        pack.add_object(sha2.clone(), blob2, false).unwrap();

        let objects = pack.list_objects();
        assert_eq!(objects.len(), 2);
        assert!(objects.contains(&sha1));
        assert!(objects.contains(&sha2));
    }

    #[test]
    fn test_pack_objects() {
        let blob1 = GitObject::Blob(Blob::new(b"test1".to_vec()));
        let blob2 = GitObject::Blob(Blob::new(b"test2".to_vec()));
        let sha1 = blob1.sha();
        let sha2 = blob2.sha();

        let objects = vec![(sha1.clone(), blob1), (sha2.clone(), blob2)];
        let pack = pack_objects(objects, true).unwrap();

        assert_eq!(pack.object_count(), 2);
        assert!(pack.has_object(&sha1));
        assert!(pack.has_object(&sha2));
    }

    #[test]
    fn test_unpack_objects() {
        let blob1 = GitObject::Blob(Blob::new(b"test1".to_vec()));
        let blob2 = GitObject::Blob(Blob::new(b"test2".to_vec()));
        let sha1 = blob1.sha();
        let sha2 = blob2.sha();

        let objects = vec![(sha1.clone(), blob1), (sha2.clone(), blob2)];
        let pack = pack_objects(objects, true).unwrap();
        let unpacked = unpack_objects(&pack).unwrap();

        assert_eq!(unpacked.len(), 2);
    }

    #[test]
    fn test_json_serialization() {
        let mut pack = PackFile::new();
        let blob = GitObject::Blob(Blob::new(b"test".to_vec()));
        let sha = blob.sha();
        pack.add_object(sha.clone(), blob, false).unwrap();

        let json = pack.to_json().unwrap();
        let deserialized = PackFile::from_json(&json).unwrap();

        assert_eq!(pack.object_count(), deserialized.object_count());
        assert!(deserialized.has_object(&sha));
    }

    #[test]
    fn test_export_objects() {
        let mut pack = PackFile::new();
        let blob = GitObject::Blob(Blob::new(b"test".to_vec()));
        let sha = blob.sha();
        pack.add_object(sha.clone(), blob, true).unwrap();

        let exported = pack.export_objects().unwrap();
        assert_eq!(exported.len(), 1);
        assert!(exported.contains_key(&sha));
    }

    #[test]
    fn test_nonexistent_object() {
        let pack = PackFile::new();
        let result = pack.get_object("nonexistent");
        assert!(result.is_err());
    }
}
