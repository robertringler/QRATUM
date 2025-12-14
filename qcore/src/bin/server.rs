//! REST API test harness for Q-Core Git Object Store

use qcore::{Commit, ObjectStore, Tree, TreeEntry};
use std::sync::{Arc, Mutex};
use tiny_http::{Method, Request, Response, Server};

fn main() {
    let store = Arc::new(Mutex::new(ObjectStore::new()));
    let server = Server::http("127.0.0.1:8080").unwrap();
    
    println!("Q-Core Git Object Store REST API");
    println!("================================");
    println!("Server listening on http://127.0.0.1:8080");
    println!("\nAvailable endpoints:");
    println!("  POST   /blob             - Create a blob");
    println!("  POST   /tree             - Create a tree");
    println!("  POST   /commit           - Create a commit");
    println!("  GET    /object/<sha>     - Get an object");
    println!("  GET    /objects          - List all objects");
    println!("  POST   /branch           - Create a branch");
    println!("  GET    /branches         - List branches");
    println!("  POST   /tag              - Create a tag");
    println!("  GET    /tags             - List tags");
    println!("  GET    /head             - Get HEAD");
    println!("  POST   /head             - Set HEAD");
    println!("  GET    /refs             - List all refs");

    for request in server.incoming_requests() {
        let store = Arc::clone(&store);
        handle_request(request, store);
    }
}

fn handle_request(mut request: Request, store: Arc<Mutex<ObjectStore>>) {
    let url = request.url().to_string();
    let method = request.method().clone();

    let response = match (method, url.as_str()) {
        (Method::Post, "/blob") => handle_create_blob(&mut request, store),
        (Method::Post, "/tree") => handle_create_tree(&mut request, store),
        (Method::Post, "/commit") => handle_create_commit(&mut request, store),
        (Method::Get, url) if url.starts_with("/object/") => handle_get_object(url, store),
        (Method::Get, "/objects") => handle_list_objects(store),
        (Method::Post, "/branch") => handle_create_branch(&mut request, store),
        (Method::Get, "/branches") => handle_list_branches(store),
        (Method::Post, "/tag") => handle_create_tag(&mut request, store),
        (Method::Get, "/tags") => handle_list_tags(store),
        (Method::Get, "/head") => handle_get_head(store),
        (Method::Post, "/head") => handle_set_head(&mut request, store),
        (Method::Get, "/refs") => handle_list_refs(store),
        _ => Response::from_string("Not Found").with_status_code(404),
    };

    request.respond(response).unwrap();
}

fn handle_create_blob(request: &mut Request, store: Arc<Mutex<ObjectStore>>) -> Response<std::io::Cursor<Vec<u8>>> {
    let mut body = Vec::new();
    if request.as_reader().read_to_end(&mut body).is_err() {
        return Response::from_string("Bad Request").with_status_code(400);
    }

    let mut store = store.lock().unwrap();
    let sha = store.store_blob(body);
    
    let response = serde_json::json!({ "sha": sha });
    Response::from_string(response.to_string())
        .with_header(tiny_http::Header::from_bytes(&b"Content-Type"[..], &b"application/json"[..]).unwrap())
}

fn handle_create_tree(request: &mut Request, store: Arc<Mutex<ObjectStore>>) -> Response<std::io::Cursor<Vec<u8>>> {
    let mut body = String::new();
    if request.as_reader().read_to_string(&mut body).is_err() {
        return Response::from_string("Bad Request").with_status_code(400);
    }

    let data: Result<serde_json::Value, _> = serde_json::from_str(&body);
    if let Ok(data) = data {
        if let Some(entries) = data["entries"].as_array() {
            let mut tree_entries = Vec::new();
            for entry in entries {
                let mode = entry["mode"].as_str().unwrap_or("").to_string();
                let name = entry["name"].as_str().unwrap_or("").to_string();
                let sha = entry["sha"].as_str().unwrap_or("").to_string();
                tree_entries.push(TreeEntry::new(mode, name, sha));
            }
            
            let tree = Tree::new(tree_entries);
            let mut store = store.lock().unwrap();
            let sha = store.store_tree(tree);
            
            let response = serde_json::json!({ "sha": sha });
            return Response::from_string(response.to_string())
                .with_header(tiny_http::Header::from_bytes(&b"Content-Type"[..], &b"application/json"[..]).unwrap());
        }
    }

    Response::from_string("Bad Request").with_status_code(400)
}

fn handle_create_commit(request: &mut Request, store: Arc<Mutex<ObjectStore>>) -> Response<std::io::Cursor<Vec<u8>>> {
    let mut body = String::new();
    if request.as_reader().read_to_string(&mut body).is_err() {
        return Response::from_string("Bad Request").with_status_code(400);
    }

    let data: Result<serde_json::Value, _> = serde_json::from_str(&body);
    if let Ok(data) = data {
        let tree = data["tree"].as_str().unwrap_or("").to_string();
        let message = data["message"].as_str().unwrap_or("").to_string();
        let author = data["author"].as_str().unwrap_or("Author <author@example.com>").to_string();
        let committer = data["committer"].as_str().unwrap_or("Committer <committer@example.com>").to_string();
        
        let parents = if let Some(p) = data["parents"].as_array() {
            p.iter().filter_map(|v| v.as_str().map(String::from)).collect()
        } else {
            Vec::new()
        };

        let commit = Commit::new(tree, parents, author, committer, message);
        let mut store = store.lock().unwrap();
        let sha = store.store_commit(commit);
        
        let response = serde_json::json!({ "sha": sha });
        return Response::from_string(response.to_string())
            .with_header(tiny_http::Header::from_bytes(&b"Content-Type"[..], &b"application/json"[..]).unwrap());
    }

    Response::from_string("Bad Request").with_status_code(400)
}

fn handle_get_object(url: &str, store: Arc<Mutex<ObjectStore>>) -> Response<std::io::Cursor<Vec<u8>>> {
    let sha = url.trim_start_matches("/object/");
    let store = store.lock().unwrap();
    
    match store.get_object(sha) {
        Ok(obj) => {
            let json = serde_json::to_string_pretty(obj).unwrap();
            Response::from_string(json)
                .with_header(tiny_http::Header::from_bytes(&b"Content-Type"[..], &b"application/json"[..]).unwrap())
        }
        Err(_) => Response::from_string("Not Found").with_status_code(404),
    }
}

fn handle_list_objects(store: Arc<Mutex<ObjectStore>>) -> Response<std::io::Cursor<Vec<u8>>> {
    let store = store.lock().unwrap();
    let objects = store.list_objects();
    
    let response = serde_json::json!({ "objects": objects, "count": objects.len() });
    Response::from_string(response.to_string())
        .with_header(tiny_http::Header::from_bytes(&b"Content-Type"[..], &b"application/json"[..]).unwrap())
}

fn handle_create_branch(request: &mut Request, store: Arc<Mutex<ObjectStore>>) -> Response<std::io::Cursor<Vec<u8>>> {
    let mut body = String::new();
    if request.as_reader().read_to_string(&mut body).is_err() {
        return Response::from_string("Bad Request").with_status_code(400);
    }

    let data: Result<serde_json::Value, _> = serde_json::from_str(&body);
    if let Ok(data) = data {
        let name = data["name"].as_str().unwrap_or("").to_string();
        let sha = data["sha"].as_str().unwrap_or("").to_string();
        
        let mut store = store.lock().unwrap();
        match store.refs_mut().create_branch(name, sha) {
            Ok(_) => {
                let response = serde_json::json!({ "status": "created" });
                return Response::from_string(response.to_string())
                    .with_header(tiny_http::Header::from_bytes(&b"Content-Type"[..], &b"application/json"[..]).unwrap());
            }
            Err(e) => {
                let response = serde_json::json!({ "error": e.to_string() });
                return Response::from_string(response.to_string())
                    .with_status_code(400)
                    .with_header(tiny_http::Header::from_bytes(&b"Content-Type"[..], &b"application/json"[..]).unwrap());
            }
        }
    }

    Response::from_string("Bad Request").with_status_code(400)
}

fn handle_list_branches(store: Arc<Mutex<ObjectStore>>) -> Response<std::io::Cursor<Vec<u8>>> {
    let store = store.lock().unwrap();
    let branches: Vec<_> = store.refs().list_branches()
        .iter()
        .map(|r| serde_json::json!({ "name": r.name, "sha": r.sha }))
        .collect();
    
    let response = serde_json::json!({ "branches": branches });
    Response::from_string(response.to_string())
        .with_header(tiny_http::Header::from_bytes(&b"Content-Type"[..], &b"application/json"[..]).unwrap())
}

fn handle_create_tag(request: &mut Request, store: Arc<Mutex<ObjectStore>>) -> Response<std::io::Cursor<Vec<u8>>> {
    let mut body = String::new();
    if request.as_reader().read_to_string(&mut body).is_err() {
        return Response::from_string("Bad Request").with_status_code(400);
    }

    let data: Result<serde_json::Value, _> = serde_json::from_str(&body);
    if let Ok(data) = data {
        let name = data["name"].as_str().unwrap_or("").to_string();
        let sha = data["sha"].as_str().unwrap_or("").to_string();
        
        let mut store = store.lock().unwrap();
        match store.refs_mut().create_tag(name, sha) {
            Ok(_) => {
                let response = serde_json::json!({ "status": "created" });
                return Response::from_string(response.to_string())
                    .with_header(tiny_http::Header::from_bytes(&b"Content-Type"[..], &b"application/json"[..]).unwrap());
            }
            Err(e) => {
                let response = serde_json::json!({ "error": e.to_string() });
                return Response::from_string(response.to_string())
                    .with_status_code(400)
                    .with_header(tiny_http::Header::from_bytes(&b"Content-Type"[..], &b"application/json"[..]).unwrap());
            }
        }
    }

    Response::from_string("Bad Request").with_status_code(400)
}

fn handle_list_tags(store: Arc<Mutex<ObjectStore>>) -> Response<std::io::Cursor<Vec<u8>>> {
    let store = store.lock().unwrap();
    let tags: Vec<_> = store.refs().list_tags()
        .iter()
        .map(|r| serde_json::json!({ "name": r.name, "sha": r.sha }))
        .collect();
    
    let response = serde_json::json!({ "tags": tags });
    Response::from_string(response.to_string())
        .with_header(tiny_http::Header::from_bytes(&b"Content-Type"[..], &b"application/json"[..]).unwrap())
}

fn handle_get_head(store: Arc<Mutex<ObjectStore>>) -> Response<std::io::Cursor<Vec<u8>>> {
    let store = store.lock().unwrap();
    
    match store.refs().get_head() {
        Some(head) => {
            let response = serde_json::json!({ "head": head });
            Response::from_string(response.to_string())
                .with_header(tiny_http::Header::from_bytes(&b"Content-Type"[..], &b"application/json"[..]).unwrap())
        }
        None => {
            let response = serde_json::json!({ "head": null });
            Response::from_string(response.to_string())
                .with_header(tiny_http::Header::from_bytes(&b"Content-Type"[..], &b"application/json"[..]).unwrap())
        }
    }
}

fn handle_set_head(request: &mut Request, store: Arc<Mutex<ObjectStore>>) -> Response<std::io::Cursor<Vec<u8>>> {
    let mut body = String::new();
    if request.as_reader().read_to_string(&mut body).is_err() {
        return Response::from_string("Bad Request").with_status_code(400);
    }

    let data: Result<serde_json::Value, _> = serde_json::from_str(&body);
    if let Ok(data) = data {
        let ref_name = data["ref"].as_str().unwrap_or("").to_string();
        
        let mut store = store.lock().unwrap();
        match store.refs_mut().set_head(ref_name) {
            Ok(_) => {
                let response = serde_json::json!({ "status": "updated" });
                return Response::from_string(response.to_string())
                    .with_header(tiny_http::Header::from_bytes(&b"Content-Type"[..], &b"application/json"[..]).unwrap());
            }
            Err(e) => {
                let response = serde_json::json!({ "error": e.to_string() });
                return Response::from_string(response.to_string())
                    .with_status_code(400)
                    .with_header(tiny_http::Header::from_bytes(&b"Content-Type"[..], &b"application/json"[..]).unwrap());
            }
        }
    }

    Response::from_string("Bad Request").with_status_code(400)
}

fn handle_list_refs(store: Arc<Mutex<ObjectStore>>) -> Response<std::io::Cursor<Vec<u8>>> {
    let store = store.lock().unwrap();
    let refs: Vec<_> = store.refs().list_all_refs()
        .iter()
        .map(|r| serde_json::json!({ "name": r.name, "sha": r.sha, "type": format!("{:?}", r.ref_type) }))
        .collect();
    
    let response = serde_json::json!({ "refs": refs });
    Response::from_string(response.to_string())
        .with_header(tiny_http::Header::from_bytes(&b"Content-Type"[..], &b"application/json"[..]).unwrap())
}
