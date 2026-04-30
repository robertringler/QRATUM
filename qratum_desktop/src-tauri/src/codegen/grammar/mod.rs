// Grammar kernel - compressed LL(k) grammar tables
// Deterministic parsing structures for code generation

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TokenType {
    Keyword,
    Identifier,
    Literal,
    Operator,
    Delimiter,
    Type,
    Comment,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProductionRule {
    pub lhs: String,
    pub rhs: Vec<String>,
    pub action: Option<String>,
}

#[derive(Debug, Clone)]
pub struct Grammar {
    pub language: String,
    pub start_symbol: String,
    pub terminals: Vec<String>,
    pub non_terminals: Vec<String>,
    pub productions: Vec<ProductionRule>,
    pub parse_table: HashMap<(String, String), usize>,
}

impl Grammar {
    pub fn new(language: String) -> Self {
        Grammar {
            language,
            start_symbol: String::from("program"),
            terminals: Vec::new(),
            non_terminals: Vec::new(),
            productions: Vec::new(),
            parse_table: HashMap::new(),
        }
    }

    pub fn load_compressed(data: &[u8]) -> Result<Self, String> {
        // Deserialize compressed grammar table
        bincode::deserialize(data).map_err(|e| format!("Failed to load grammar: {}", e))
    }

    pub fn parse(&self, tokens: &[String]) -> Result<Vec<usize>, String> {
        let mut stack = vec![self.start_symbol.clone()];
        let mut productions_used = Vec::new();
        let mut token_idx = 0;

        while !stack.is_empty() && token_idx < tokens.len() {
            let top = stack.pop().unwrap();
            let current_token = &tokens[token_idx];

            if self.terminals.contains(&top) {
                if &top == current_token {
                    token_idx += 1;
                } else {
                    return Err(format!(
                        "Parse error: expected {}, got {}",
                        top, current_token
                    ));
                }
            } else {
                let key = (top.clone(), current_token.clone());
                if let Some(&prod_idx) = self.parse_table.get(&key) {
                    productions_used.push(prod_idx);
                    let production = &self.productions[prod_idx];
                    for symbol in production.rhs.iter().rev() {
                        stack.push(symbol.clone());
                    }
                } else {
                    return Err(format!(
                        "No production for {} with lookahead {}",
                        top, current_token
                    ));
                }
            }
        }

        Ok(productions_used)
    }
}

// Rust grammar builder
pub fn build_rust_grammar() -> Grammar {
    let mut g = Grammar::new("rust".to_string());

    g.non_terminals = vec![
        "program", "item", "function", "struct", "enum", "impl", "trait", "stmt", "expr",
        "pattern", "type", "block",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();

    g.terminals = vec![
        "fn",
        "struct",
        "enum",
        "impl",
        "trait",
        "let",
        "mut",
        "pub",
        "if",
        "else",
        "while",
        "for",
        "match",
        "return",
        "identifier",
        "literal",
        "->",
        "{",
        "}",
        "(",
        ")",
        ";",
        ",",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();

    // Minimal production rules
    g.productions = vec![
        ProductionRule {
            lhs: "program".to_string(),
            rhs: vec!["item".to_string()],
            action: Some("build_program".to_string()),
        },
        ProductionRule {
            lhs: "item".to_string(),
            rhs: vec!["function".to_string()],
            action: Some("build_item_fn".to_string()),
        },
        ProductionRule {
            lhs: "function".to_string(),
            rhs: vec![
                "fn".to_string(),
                "identifier".to_string(),
                "(",
                ")",
                "block".to_string(),
            ],
            action: Some("build_function".to_string()),
        },
        ProductionRule {
            lhs: "block".to_string(),
            rhs: vec!["{".to_string(), "stmt".to_string(), "}".to_string()],
            action: Some("build_block".to_string()),
        },
        ProductionRule {
            lhs: "stmt".to_string(),
            rhs: vec!["return".to_string(), "expr".to_string(), ";".to_string()],
            action: Some("build_return_stmt".to_string()),
        },
        ProductionRule {
            lhs: "expr".to_string(),
            rhs: vec!["literal".to_string()],
            action: Some("build_literal_expr".to_string()),
        },
    ];

    g
}

// Python grammar builder
pub fn build_python_grammar() -> Grammar {
    let mut g = Grammar::new("python".to_string());

    g.non_terminals = vec![
        "program",
        "stmt",
        "func_def",
        "class_def",
        "expr",
        "assignment",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();

    g.terminals = vec![
        "def",
        "class",
        "return",
        "if",
        "else",
        "for",
        "while",
        "identifier",
        "literal",
        ":",
        "=",
        "(",
        ")",
        ",",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();

    g.productions = vec![
        ProductionRule {
            lhs: "program".to_string(),
            rhs: vec!["stmt".to_string()],
            action: Some("build_program".to_string()),
        },
        ProductionRule {
            lhs: "stmt".to_string(),
            rhs: vec!["func_def".to_string()],
            action: Some("build_stmt_func".to_string()),
        },
        ProductionRule {
            lhs: "func_def".to_string(),
            rhs: vec![
                "def".to_string(),
                "identifier".to_string(),
                "(",
                ")",
                ":".to_string(),
                "stmt".to_string(),
            ],
            action: Some("build_function".to_string()),
        },
    ];

    g
}

// JavaScript grammar builder
pub fn build_js_grammar() -> Grammar {
    let mut g = Grammar::new("javascript".to_string());

    g.non_terminals = vec!["program", "stmt", "func_decl", "expr", "assignment"]
        .iter()
        .map(|s| s.to_string())
        .collect();

    g.terminals = vec![
        "function",
        "const",
        "let",
        "var",
        "return",
        "if",
        "else",
        "identifier",
        "literal",
        "=>",
        "=",
        "(",
        ")",
        "{",
        "}",
        ";",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();

    g.productions = vec![ProductionRule {
        lhs: "program".to_string(),
        rhs: vec!["stmt".to_string()],
        action: Some("build_program".to_string()),
    }];

    g
}

// C grammar builder
pub fn build_c_grammar() -> Grammar {
    let mut g = Grammar::new("c".to_string());

    g.non_terminals = vec!["program", "declaration", "func_def", "stmt", "expr"]
        .iter()
        .map(|s| s.to_string())
        .collect();

    g.terminals = vec![
        "int",
        "void",
        "char",
        "float",
        "return",
        "if",
        "else",
        "while",
        "identifier",
        "literal",
        "(",
        ")",
        "{",
        "}",
        ";",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();

    g.productions = vec![ProductionRule {
        lhs: "program".to_string(),
        rhs: vec!["func_def".to_string()],
        action: Some("build_program".to_string()),
    }];

    g
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rust_grammar_creation() {
        let grammar = build_rust_grammar();
        assert_eq!(grammar.language, "rust");
        assert!(!grammar.productions.is_empty());
        assert!(!grammar.terminals.is_empty());
    }

    #[test]
    fn test_python_grammar_creation() {
        let grammar = build_python_grammar();
        assert_eq!(grammar.language, "python");
    }
}
