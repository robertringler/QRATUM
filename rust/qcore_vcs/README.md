# QCore VCS - Universal Version Control System Engine

A foundational Rust engine for multi-VCS (Git, Mercurial, SVN) with CRDT overlays, designed for conflict-free collaboration, cross-repo diff, and high-performance object storage.

## Features

### Phase 1: MVP (Current Implementation)

- ✅ **Git-Compatible Object Store**: Blobs, Trees, Commits, and Tags
- ✅ **SHA-256 Object Hashing**: Content-addressable storage with deduplication
- ✅ **CRDT Timeline**: Conflict-free replicated data type for distributed collaboration
- ✅ **VCS Adapter Interface**: Pluggable architecture for multiple VCS backends
- ✅ **Git Adapter**: Basic Git operations (init, commit, checkout, branch)
- ✅ **Comprehensive Tests**: Unit tests for all core components

### Planned Features

- **Multi-VCS Adapters**: Mercurial and SVN support
- **Advanced CRDT Operations**: Three-way merge, conflict resolution
- **Object Pool Optimization**: LFS deduplication, compression
- **Tree-sitter Integration**: AST-based diff and search
- **Sublinear Performance**: <15ms GitOps operations
- **Live State Streaming**: Real-time collaboration support
- **Firecracker Integration**: Isolated runtime environment

## Architecture

```
qcore_vcs/
├── src/
│   ├── lib.rs           # Public API and error types
│   ├── object_store.rs  # Git object storage (Blob, Tree, Commit, Tag)
│   ├── crdt.rs          # CRDT timeline for conflict-free collaboration
│   ├── vcs.rs           # VCS adapter trait and types
│   └── git.rs           # Git adapter implementation
├── Cargo.toml
└── README.md
```

## Usage

```rust
use qcore_vcs::{
    git::GitAdapter,
    object_store::{Blob, GitObject, ObjectStore},
    vcs::VcsAdapter,
};

// Create a new Git adapter
let mut adapter = GitAdapter::new();
adapter.init().unwrap();

// Commit changes
let commit_id = adapter.commit("Initial commit").unwrap();
println!("Created commit: {}", commit_id);

// Work with object store
let mut store = adapter.store_mut();
let blob = Blob::new(b"Hello, World!".to_vec());
let blob_id = store.store(GitObject::Blob(blob)).unwrap();
```

## CRDT Timeline Example

```rust
use qcore_vcs::{
    crdt::{CrdtTimeline, Operation},
    object_store::ObjectId,
};

let mut timeline = CrdtTimeline::new();

// Add operations from multiple users
let op1 = Operation::new(
    ObjectId::new("commit1".to_string()),
    "user1@example.com".to_string(),
    chrono::Utc::now().timestamp(),
    vec![],
);
timeline.add_operation(op1).unwrap();

// Merge timelines from different replicas
let mut remote_timeline = CrdtTimeline::new();
// ... add operations to remote_timeline ...
timeline.merge(&remote_timeline).unwrap();
```

## Building and Testing

```bash
# Build the library
cargo build

# Run tests
cargo test

# Run tests with verbose output
cargo test -- --nocapture

# Build with optimizations
cargo build --release
```

## Performance Characteristics

- **Object Deduplication**: O(1) lookup and storage using SHA-256 hashing
- **CRDT Merge**: O(n) where n is the number of operations in the remote timeline
- **Topological Sort**: O(n + e) where n is operations and e is dependencies
- **Target**: <15ms for standard GitOps operations (Phase 3)

## Compliance and Certification

QCore VCS is designed to meet:

- **DO-178C Level A**: Aerospace software certification standards
- **NIST 800-53**: Federal security controls
- **CMMC 2.0 Level 2**: Defense contractor cybersecurity
- **Deterministic Reproducibility**: <1μs seed replay drift tolerance

## License

Apache License 2.0 - See LICENSE file for details

## Contributing

Contributions are welcome! Please ensure:

1. All tests pass: `cargo test`
2. Code is formatted: `cargo fmt`
3. No clippy warnings: `cargo clippy`
4. Documentation is updated

## Roadmap

### Phase 2: Multi-VCS Adapters (Q1 2026)
- Mercurial adapter implementation
- SVN adapter implementation
- Config-driven repo ingest

### Phase 3: Performance Optimization (Q2 2026)
- Object pool with compression
- Sublinear diff/search via AST+vector indexing
- <15ms GitOps performance target
- LFS deduplication

### Phase 4: Advanced Features (Q3 2026)
- Tree-sitter integration for AST parsing
- Live state streaming for real-time collaboration
- Firecracker runner integration
- Multi-user conflictless merge

## Contact

For questions, issues, or contributions, please refer to the main QuASIM repository.
