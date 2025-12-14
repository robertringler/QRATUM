//! CLI test harness for Q-Core Git Object Store

use qcore::{Commit, GitObject, ObjectStore, Tree, TreeEntry};
use std::io::{self, Write};

fn main() {
    println!("Q-Core Git Object Store CLI");
    println!("===========================\n");

    let mut store = ObjectStore::new();

    loop {
        print!("> ");
        io::stdout().flush().unwrap();

        let mut input = String::new();
        io::stdin().read_line(&mut input).unwrap();
        let input = input.trim();

        if input.is_empty() {
            continue;
        }

        let parts: Vec<&str> = input.split_whitespace().collect();
        let command = parts[0];

        match command {
            "help" => print_help(),
            "exit" | "quit" => break,
            "blob" => handle_blob(&mut store, &parts[1..]),
            "tree" => handle_tree(&mut store, &parts[1..]),
            "commit" => handle_commit(&mut store, &parts[1..]),
            "get" => handle_get(&store, &parts[1..]),
            "list" => handle_list(&store),
            "branch" => handle_branch(&mut store, &parts[1..]),
            "tag" => handle_tag(&mut store, &parts[1..]),
            "head" => handle_head(&mut store, &parts[1..]),
            "refs" => handle_refs(&store),
            _ => println!("Unknown command. Type 'help' for available commands."),
        }
    }

    println!("Goodbye!");
}

fn print_help() {
    println!("Available commands:");
    println!("  blob <data>                - Create a blob with the given data");
    println!("  tree <mode> <name> <sha>   - Create a tree with one entry");
    println!("  commit <tree> <msg>        - Create a commit");
    println!("  get <sha>                  - Retrieve an object by SHA");
    println!("  list                       - List all objects");
    println!("  branch create <name> <sha> - Create a branch");
    println!("  branch list                - List all branches");
    println!("  tag create <name> <sha>    - Create a tag");
    println!("  tag list                   - List all tags");
    println!("  head <ref>                 - Set HEAD to a ref");
    println!("  refs                       - List all refs");
    println!("  help                       - Show this help");
    println!("  exit/quit                  - Exit the CLI");
}

fn handle_blob(store: &mut ObjectStore, args: &[&str]) {
    if args.is_empty() {
        println!("Usage: blob <data>");
        return;
    }

    let data = args.join(" ").into_bytes();
    let sha = store.store_blob(data);
    println!("Blob created with SHA: {}", sha);
}

fn handle_tree(store: &mut ObjectStore, args: &[&str]) {
    if args.len() < 3 {
        println!("Usage: tree <mode> <name> <sha>");
        return;
    }

    let mode = args[0].to_string();
    let name = args[1].to_string();
    let sha = args[2].to_string();

    let entry = TreeEntry::new(mode, name, sha);
    let tree = Tree::new(vec![entry]);
    let tree_sha = store.store_tree(tree);
    println!("Tree created with SHA: {}", tree_sha);
}

fn handle_commit(store: &mut ObjectStore, args: &[&str]) {
    if args.len() < 2 {
        println!("Usage: commit <tree_sha> <message>");
        return;
    }

    let tree = args[0].to_string();
    let message = args[1..].join(" ");

    let commit = Commit::new(
        tree,
        vec![],
        "Author <author@example.com>".to_string(),
        "Committer <committer@example.com>".to_string(),
        message,
    );
    let sha = store.store_commit(commit);
    println!("Commit created with SHA: {}", sha);
}

fn handle_get(store: &ObjectStore, args: &[&str]) {
    if args.is_empty() {
        println!("Usage: get <sha>");
        return;
    }

    let sha = args[0];
    match store.get_object(sha) {
        Ok(obj) => {
            println!("Object type: {}", obj.object_type());
            match obj {
                GitObject::Blob(blob) => {
                    println!("Data length: {} bytes", blob.data.len());
                    if let Ok(s) = String::from_utf8(blob.data.clone()) {
                        println!("Data: {}", s);
                    } else {
                        println!("Data: <binary>");
                    }
                }
                GitObject::Tree(tree) => {
                    println!("Entries: {}", tree.entries.len());
                    for entry in &tree.entries {
                        println!("  {} {} {}", entry.mode, entry.name, entry.sha);
                    }
                }
                GitObject::Commit(commit) => {
                    println!("Tree: {}", commit.tree);
                    println!("Parents: {}", commit.parents.len());
                    println!("Author: {}", commit.author);
                    println!("Committer: {}", commit.committer);
                    println!("Message: {}", commit.message);
                }
            }
        }
        Err(e) => println!("Error: {}", e),
    }
}

fn handle_list(store: &ObjectStore) {
    let objects = store.list_objects();
    println!("Total objects: {}", objects.len());
    for sha in objects {
        if let Ok(obj) = store.get_object(&sha) {
            println!("  {} ({})", sha, obj.object_type());
        }
    }
}

fn handle_branch(store: &mut ObjectStore, args: &[&str]) {
    if args.is_empty() {
        println!("Usage: branch <create|list> [args...]");
        return;
    }

    match args[0] {
        "create" => {
            if args.len() < 3 {
                println!("Usage: branch create <name> <sha>");
                return;
            }
            let name = args[1].to_string();
            let sha = args[2].to_string();
            match store.refs_mut().create_branch(name, sha) {
                Ok(_) => println!("Branch created"),
                Err(e) => println!("Error: {}", e),
            }
        }
        "list" => {
            let branches = store.refs().list_branches();
            println!("Branches: {}", branches.len());
            for branch in branches {
                println!("  {} -> {}", branch.name, branch.sha);
            }
        }
        _ => println!("Unknown subcommand. Use 'create' or 'list'."),
    }
}

fn handle_tag(store: &mut ObjectStore, args: &[&str]) {
    if args.is_empty() {
        println!("Usage: tag <create|list> [args...]");
        return;
    }

    match args[0] {
        "create" => {
            if args.len() < 3 {
                println!("Usage: tag create <name> <sha>");
                return;
            }
            let name = args[1].to_string();
            let sha = args[2].to_string();
            match store.refs_mut().create_tag(name, sha) {
                Ok(_) => println!("Tag created"),
                Err(e) => println!("Error: {}", e),
            }
        }
        "list" => {
            let tags = store.refs().list_tags();
            println!("Tags: {}", tags.len());
            for tag in tags {
                println!("  {} -> {}", tag.name, tag.sha);
            }
        }
        _ => println!("Unknown subcommand. Use 'create' or 'list'."),
    }
}

fn handle_head(store: &mut ObjectStore, args: &[&str]) {
    if args.is_empty() {
        match store.refs().get_head() {
            Some(head) => println!("HEAD: {}", head),
            None => println!("HEAD not set"),
        }
    } else {
        let ref_name = args[0].to_string();
        match store.refs_mut().set_head(ref_name) {
            Ok(_) => println!("HEAD updated"),
            Err(e) => println!("Error: {}", e),
        }
    }
}

fn handle_refs(store: &ObjectStore) {
    let refs = store.refs().list_all_refs();
    println!("Total refs: {}", refs.len());
    for ref_obj in refs {
        println!("  {} -> {}", ref_obj.name, ref_obj.sha);
    }
}
