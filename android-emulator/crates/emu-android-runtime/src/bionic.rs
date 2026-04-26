//! Bionic libc emulation layer.
//!
//! Provides compatibility with Android's Bionic C library.

use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;

use emu_memory::Mmu;

use crate::error::{RuntimeError, RuntimeResult};

/// Thread-local storage key.
pub type TlsKey = u32;

/// TLS destructor function pointer.
pub type TlsDestructor = Option<fn(u64)>;

/// Bionic libc emulation.
pub struct BionicEmulator {
    /// TLS keys and their values.
    tls_keys: RwLock<HashMap<TlsKey, TlsSlot>>,
    /// Next TLS key to allocate.
    next_tls_key: RwLock<TlsKey>,
    /// Environment variables.
    environ: RwLock<HashMap<String, String>>,
    /// Heap allocator.
    heap: RwLock<HeapState>,
}

struct TlsSlot {
    value: u64,
    destructor: TlsDestructor,
}

struct HeapState {
    /// Start of heap.
    base: u64,
    /// Current break (end of heap).
    brk: u64,
    /// Maximum heap size.
    max_size: u64,
    /// Free list (address, size).
    free_list: Vec<(u64, u64)>,
}

impl BionicEmulator {
    /// Create a new Bionic emulator.
    pub fn new(heap_base: u64, heap_size: u64) -> Self {
        Self {
            tls_keys: RwLock::new(HashMap::new()),
            next_tls_key: RwLock::new(0),
            environ: RwLock::new(Self::default_environ()),
            heap: RwLock::new(HeapState {
                base: heap_base,
                brk: heap_base,
                max_size: heap_size,
                free_list: Vec::new(),
            }),
        }
    }

    fn default_environ() -> HashMap<String, String> {
        let mut env = HashMap::new();
        env.insert("PATH".to_string(), "/system/bin:/system/xbin".to_string());
        env.insert("ANDROID_DATA".to_string(), "/data".to_string());
        env.insert("ANDROID_ROOT".to_string(), "/system".to_string());
        env.insert("HOME".to_string(), "/data/user/0".to_string());
        env.insert("TMPDIR".to_string(), "/data/local/tmp".to_string());
        env
    }

    /// Allocate a TLS key.
    pub fn pthread_key_create(&self, destructor: TlsDestructor) -> RuntimeResult<TlsKey> {
        let mut next = self.next_tls_key.write();
        let key = *next;
        *next += 1;

        self.tls_keys.write().insert(key, TlsSlot {
            value: 0,
            destructor,
        });

        Ok(key)
    }

    /// Delete a TLS key.
    pub fn pthread_key_delete(&self, key: TlsKey) -> RuntimeResult<()> {
        self.tls_keys.write().remove(&key);
        Ok(())
    }

    /// Get TLS value.
    pub fn pthread_getspecific(&self, key: TlsKey) -> Option<u64> {
        self.tls_keys.read().get(&key).map(|s| s.value)
    }

    /// Set TLS value.
    pub fn pthread_setspecific(&self, key: TlsKey, value: u64) -> RuntimeResult<()> {
        if let Some(slot) = self.tls_keys.write().get_mut(&key) {
            slot.value = value;
            Ok(())
        } else {
            Err(RuntimeError::SyscallError {
                syscall: "pthread_setspecific".to_string(),
                errno: 22, // EINVAL
            })
        }
    }

    /// Get environment variable.
    pub fn getenv(&self, name: &str) -> Option<String> {
        self.environ.read().get(name).cloned()
    }

    /// Set environment variable.
    pub fn setenv(&self, name: &str, value: &str, overwrite: bool) -> RuntimeResult<()> {
        let mut env = self.environ.write();
        if !overwrite && env.contains_key(name) {
            return Ok(());
        }
        env.insert(name.to_string(), value.to_string());
        Ok(())
    }

    /// Unset environment variable.
    pub fn unsetenv(&self, name: &str) -> RuntimeResult<()> {
        self.environ.write().remove(name);
        Ok(())
    }

    /// Allocate memory (malloc).
    pub fn malloc(&self, size: u64) -> RuntimeResult<u64> {
        if size == 0 {
            return Ok(0);
        }

        // Align to 16 bytes
        let aligned_size = (size + 15) & !15;
        
        let mut heap = self.heap.write();
        
        // Try to find a suitable block in the free list
        for i in 0..heap.free_list.len() {
            let (addr, block_size) = heap.free_list[i];
            if block_size >= aligned_size {
                heap.free_list.remove(i);
                if block_size > aligned_size + 32 {
                    // Split the block
                    heap.free_list.push((addr + aligned_size, block_size - aligned_size));
                }
                return Ok(addr);
            }
        }

        // Allocate from the top of the heap
        let addr = heap.brk;
        let new_brk = heap.brk + aligned_size;
        
        if new_brk > heap.base + heap.max_size {
            return Err(RuntimeError::OutOfMemory {
                requested: size as usize,
            });
        }
        
        heap.brk = new_brk;
        Ok(addr)
    }

    /// Free memory.
    pub fn free(&self, ptr: u64, size: u64) {
        if ptr == 0 {
            return;
        }
        
        let mut heap = self.heap.write();
        heap.free_list.push((ptr, size));
        
        // TODO: Coalesce adjacent free blocks
    }

    /// Reallocate memory.
    pub fn realloc(&self, ptr: u64, old_size: u64, new_size: u64) -> RuntimeResult<u64> {
        if ptr == 0 {
            return self.malloc(new_size);
        }
        
        if new_size == 0 {
            self.free(ptr, old_size);
            return Ok(0);
        }
        
        if new_size <= old_size {
            return Ok(ptr);
        }
        
        // Allocate new block and copy
        let new_ptr = self.malloc(new_size)?;
        // Note: Actual copying would need MMU access
        self.free(ptr, old_size);
        
        Ok(new_ptr)
    }

    /// Get program break (sbrk).
    pub fn sbrk(&self, increment: i64) -> RuntimeResult<u64> {
        let mut heap = self.heap.write();
        let old_brk = heap.brk;
        
        let new_brk = if increment >= 0 {
            heap.brk.checked_add(increment as u64)
        } else {
            heap.brk.checked_sub((-increment) as u64)
        };
        
        match new_brk {
            Some(brk) if brk >= heap.base && brk <= heap.base + heap.max_size => {
                heap.brk = brk;
                Ok(old_brk)
            }
            _ => Err(RuntimeError::OutOfMemory {
                requested: increment.unsigned_abs() as usize,
            }),
        }
    }

    /// Get current heap break.
    pub fn brk(&self) -> u64 {
        self.heap.read().brk
    }
}

impl Default for BionicEmulator {
    fn default() -> Self {
        // Default heap at 0x20000000, 256MB
        Self::new(0x2000_0000, 256 * 1024 * 1024)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tls() {
        let bionic = BionicEmulator::default();
        
        let key = bionic.pthread_key_create(None).unwrap();
        assert!(bionic.pthread_getspecific(key).unwrap_or(0) == 0);
        
        bionic.pthread_setspecific(key, 42).unwrap();
        assert_eq!(bionic.pthread_getspecific(key), Some(42));
    }

    #[test]
    fn test_environ() {
        let bionic = BionicEmulator::default();
        
        assert!(bionic.getenv("PATH").is_some());
        
        bionic.setenv("TEST_VAR", "hello", true).unwrap();
        assert_eq!(bionic.getenv("TEST_VAR"), Some("hello".to_string()));
        
        bionic.unsetenv("TEST_VAR").unwrap();
        assert!(bionic.getenv("TEST_VAR").is_none());
    }

    #[test]
    fn test_malloc() {
        let bionic = BionicEmulator::new(0x1000, 0x10000);
        
        let ptr1 = bionic.malloc(100).unwrap();
        assert!(ptr1 >= 0x1000);
        
        let ptr2 = bionic.malloc(200).unwrap();
        assert!(ptr2 > ptr1);
        
        bionic.free(ptr1, 100);
        
        // Should reuse freed block
        let ptr3 = bionic.malloc(50).unwrap();
        assert_eq!(ptr3, ptr1);
    }
}
