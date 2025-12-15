//! Basic usage example for QCore VCS

use qcore_vcs::{
    crdt::{CrdtTimeline, Operation},
    git::GitAdapter,
    object_store::{Blob, GitObject, ObjectId, ObjectStore},
    vcs::VcsAdapter,
};

fn main() {
    println!("QCore VCS - Basic Usage Example\n");

    // Git Adapter
    println!("=== Git Adapter ===");
    let mut git = GitAdapter::new();
    git.init().unwrap();
    let commit_id = git.commit("Initial commit").unwrap();
    println!("Created commit: {}", commit_id);

    // Object Store
    println!("\n=== Object Store ===");
    let mut store = ObjectStore::new();
    let blob = Blob::new(b"Hello, QCore VCS!".to_vec());
    let blob_id = store.store(GitObject::Blob(blob)).unwrap();
    println!("Stored blob: {}", blob_id);

    // CRDT Timeline
    println!("\n=== CRDT Timeline ===");
    let mut timeline = CrdtTimeline::new();
    let op1 = Operation::new(
        ObjectId::new("commit1".to_string()),
        "alice@example.com".to_string(),
        chrono::Utc::now().timestamp(),
        vec![],
    );
    timeline.add_operation(op1).unwrap();
    println!("Timeline size: {}", timeline.len());

    println!("\nExample completed successfully!");
}
