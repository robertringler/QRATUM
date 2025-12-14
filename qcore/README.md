# Q-Core: Rust Git Object Store

Q-Core is a bare object store for Git implemented in Rust. It provides an in-memory implementation of Git's object model with support for pack/unpack operations, ref management, and both CLI and REST API interfaces.

## Features

- **Git Object Types**: Full support for blobs, trees, and commits
- **SHA-1 Hashing**: Proper Git-compatible SHA-1 calculation for all objects
- **In-Memory Store**: Fast object storage and retrieval by SHA
- **Ref Management**: Branches, tags, and HEAD management
- **Pack/Unpack**: Compression and serialization of objects
- **CLI Interface**: Interactive command-line tool for testing
- **REST API**: HTTP server for programmatic access
- **Comprehensive Tests**: Unit tests for all core functionality

## Architecture

```
qcore/
├── src/
│   ├── lib.rs          # Library entry point
│   ├── error.rs        # Error types
│   ├── objects.rs      # Git object types (Blob, Tree, Commit)
│   ├── store.rs        # In-memory object store
│   ├── refs.rs         # Reference management
│   ├── pack.rs         # Pack/unpack operations
│   └── bin/
│       ├── cli.rs      # CLI test harness
│       └── server.rs   # REST API server
└── Cargo.toml          # Rust package configuration
```

## Building

```bash
# Build the library and binaries
cargo build --release

# Run tests
cargo test

# Run with verbose output
cargo test -- --nocapture
```

## CLI Usage

```bash
# Run the CLI
cargo run --bin qcore-cli

# Example session
> blob Hello, World!
Blob created with SHA: 8ab686eafeb1f44702738c8b0f24f2567c36da6d

> tree 100644 file.txt 8ab686eafeb1f44702738c8b0f24f2567c36da6d
Tree created with SHA: <tree-sha>

> commit <tree-sha> Initial commit
Commit created with SHA: <commit-sha>

> branch create main <commit-sha>
Branch created

> tag create v1.0.0 <commit-sha>
Tag created

> list
Total objects: 3
  <blob-sha> (blob)
  <tree-sha> (tree)
  <commit-sha> (commit)

> help
Available commands:
  blob <data>                - Create a blob with the given data
  tree <mode> <name> <sha>   - Create a tree with one entry
  commit <tree> <msg>        - Create a commit
  get <sha>                  - Retrieve an object by SHA
  list                       - List all objects
  branch create <name> <sha> - Create a branch
  branch list                - List all branches
  tag create <name> <sha>    - Create a tag
  tag list                   - List all tags
  head <ref>                 - Set HEAD to a ref
  refs                       - List all refs
  help                       - Show this help
  exit/quit                  - Exit the CLI
```

## REST API Usage

```bash
# Start the server
cargo run --bin qcore-server

# The server listens on http://127.0.0.1:8080
```

### API Endpoints

#### Create a Blob
```bash
curl -X POST http://127.0.0.1:8080/blob \
  -d "Hello, World!"
```

#### Create a Tree
```bash
curl -X POST http://127.0.0.1:8080/tree \
  -H "Content-Type: application/json" \
  -d '{
    "entries": [
      {"mode": "100644", "name": "file.txt", "sha": "<blob-sha>"}
    ]
  }'
```

#### Create a Commit
```bash
curl -X POST http://127.0.0.1:8080/commit \
  -H "Content-Type: application/json" \
  -d '{
    "tree": "<tree-sha>",
    "message": "Initial commit",
    "author": "Author <author@example.com>",
    "committer": "Committer <committer@example.com>",
    "parents": []
  }'
```

#### Get an Object
```bash
curl http://127.0.0.1:8080/object/<sha>
```

#### List All Objects
```bash
curl http://127.0.0.1:8080/objects
```

#### Create a Branch
```bash
curl -X POST http://127.0.0.1:8080/branch \
  -H "Content-Type: application/json" \
  -d '{"name": "main", "sha": "<commit-sha>"}'
```

#### List Branches
```bash
curl http://127.0.0.1:8080/branches
```

#### Create a Tag
```bash
curl -X POST http://127.0.0.1:8080/tag \
  -H "Content-Type: application/json" \
  -d '{"name": "v1.0.0", "sha": "<commit-sha>"}'
```

#### List Tags
```bash
curl http://127.0.0.1:8080/tags
```

#### Get HEAD
```bash
curl http://127.0.0.1:8080/head
```

#### Set HEAD
```bash
curl -X POST http://127.0.0.1:8080/head \
  -H "Content-Type: application/json" \
  -d '{"ref": "refs/heads/main"}'
```

#### List All Refs
```bash
curl http://127.0.0.1:8080/refs
```

## Library Usage

```rust
use qcore::{ObjectStore, Blob, Tree, TreeEntry, Commit};

fn main() {
    // Create a new object store
    let mut store = ObjectStore::new();

    // Store a blob
    let blob_sha = store.store_blob(b"Hello, World!".to_vec());

    // Create and store a tree
    let entry = TreeEntry::new(
        "100644".to_string(),
        "file.txt".to_string(),
        blob_sha.clone(),
    );
    let tree = Tree::new(vec![entry]);
    let tree_sha = store.store_tree(tree);

    // Create and store a commit
    let commit = Commit::new(
        tree_sha,
        vec![],
        "Author <author@example.com>".to_string(),
        "Committer <committer@example.com>".to_string(),
        "Initial commit".to_string(),
    );
    let commit_sha = store.store_commit(commit);

    // Create a branch
    store.refs_mut()
        .create_branch("main".to_string(), commit_sha.clone())
        .unwrap();

    // Set HEAD
    store.refs_mut()
        .set_head("refs/heads/main".to_string())
        .unwrap();

    // Retrieve objects
    let blob = store.get_blob(&blob_sha).unwrap();
    println!("Blob data: {:?}", blob.data);

    let commit = store.get_commit(&commit_sha).unwrap();
    println!("Commit message: {}", commit.message);
}
```

## Testing

The library includes comprehensive unit tests:

```bash
# Run all tests
cargo test

# Run tests with output
cargo test -- --nocapture

# Run specific test module
cargo test objects
cargo test refs
cargo test store
cargo test pack
```

## Performance

Q-Core is designed for in-memory operation with fast object storage and retrieval:

- **O(1)** object lookup by SHA
- **O(1)** ref lookup by name
- Efficient compression using zlib
- Minimal allocations and copying

## Integration with QuASIM

Q-Core is part of the QuASIM ecosystem and provides Git-like object storage capabilities for:

- Provenance tracking via QuNimbus
- Version control of simulation artifacts
- Deterministic state snapshots
- Distributed simulation coordination

## License

Apache-2.0

## Contributing

See the main QuASIM CONTRIBUTING.md for guidelines.
