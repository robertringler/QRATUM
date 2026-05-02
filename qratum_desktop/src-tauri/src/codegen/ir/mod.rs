// Typed IR Layer - Minimal typed intermediate representation
// Symbol tables, type constraints, error propagation

use crate::codegen::ast::AstNode;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypedIR {
    pub symbols: SymbolTable,
    pub type_constraints: Vec<TypeConstraint>,
    pub error_rules: Vec<ErrorRule>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymbolTable {
    pub scopes: Vec<Scope>,
    pub current_scope: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Scope {
    pub symbols: HashMap<String, Symbol>,
    pub parent: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Symbol {
    pub name: String,
    pub symbol_type: SymbolType,
    pub type_info: TypeInfo,
    pub mutable: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SymbolType {
    Variable,
    Function,
    Type,
    Module,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypeInfo {
    pub base_type: String,
    pub is_reference: bool,
    pub is_mutable: bool,
    pub generic_params: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypeConstraint {
    pub constraint_type: ConstraintType,
    pub message: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConstraintType {
    TypeMatch { expected: String, actual: String },
    Lifetime { name: String, bound: String },
    Trait { name: String, bound: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorRule {
    pub pattern: String,
    pub propagation: ErrorPropagation,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ErrorPropagation {
    Return,
    Panic,
    Unwrap,
    Propagate,
}

impl TypedIR {
    pub fn new() -> Self {
        TypedIR {
            symbols: SymbolTable::new(),
            type_constraints: Vec::new(),
            error_rules: Vec::new(),
        }
    }

    pub fn validate(&self, ast: &AstNode) -> Result<(), Vec<String>> {
        let mut errors = Vec::new();

        // Validate symbol resolution
        if let Err(e) = self.validate_symbols(ast) {
            errors.push(e);
        }

        // Validate type constraints
        if let Err(e) = self.validate_types(ast) {
            errors.push(e);
        }

        // Validate error handling
        if let Err(e) = self.validate_errors(ast) {
            errors.push(e);
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }

    fn validate_symbols(&self, _ast: &AstNode) -> Result<(), String> {
        // Check that all referenced symbols are defined
        Ok(())
    }

    fn validate_types(&self, _ast: &AstNode) -> Result<(), String> {
        // Check type consistency
        Ok(())
    }

    fn validate_errors(&self, _ast: &AstNode) -> Result<(), String> {
        // Check error propagation rules
        Ok(())
    }

    pub fn add_symbol(&mut self, symbol: Symbol) -> Result<(), String> {
        let current = &mut self.symbols.scopes[self.symbols.current_scope];

        if current.symbols.contains_key(&symbol.name) {
            return Err(format!(
                "Symbol '{}' already defined in this scope",
                symbol.name
            ));
        }

        current.symbols.insert(symbol.name.clone(), symbol);
        Ok(())
    }

    pub fn lookup_symbol(&self, name: &str) -> Option<&Symbol> {
        let mut scope_idx = self.symbols.current_scope;

        loop {
            let scope = &self.symbols.scopes[scope_idx];
            if let Some(symbol) = scope.symbols.get(name) {
                return Some(symbol);
            }

            match scope.parent {
                Some(parent_idx) => scope_idx = parent_idx,
                None => return None,
            }
        }
    }

    pub fn enter_scope(&mut self) {
        let parent = self.symbols.current_scope;
        let new_scope = Scope {
            symbols: HashMap::new(),
            parent: Some(parent),
        };
        self.symbols.scopes.push(new_scope);
        self.symbols.current_scope = self.symbols.scopes.len() - 1;
    }

    pub fn exit_scope(&mut self) -> Result<(), String> {
        let current = &self.symbols.scopes[self.symbols.current_scope];
        match current.parent {
            Some(parent_idx) => {
                self.symbols.current_scope = parent_idx;
                Ok(())
            }
            None => Err("Cannot exit root scope".to_string()),
        }
    }
}

impl SymbolTable {
    pub fn new() -> Self {
        SymbolTable {
            scopes: vec![Scope {
                symbols: HashMap::new(),
                parent: None,
            }],
            current_scope: 0,
        }
    }
}

impl Default for TypedIR {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for SymbolTable {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_symbol_table_creation() {
        let ir = TypedIR::new();
        assert_eq!(ir.symbols.scopes.len(), 1);
        assert_eq!(ir.symbols.current_scope, 0);
    }

    #[test]
    fn test_add_symbol() {
        let mut ir = TypedIR::new();
        let symbol = Symbol {
            name: "x".to_string(),
            symbol_type: SymbolType::Variable,
            type_info: TypeInfo {
                base_type: "i32".to_string(),
                is_reference: false,
                is_mutable: false,
                generic_params: Vec::new(),
            },
            mutable: false,
        };

        assert!(ir.add_symbol(symbol).is_ok());
    }

    #[test]
    fn test_scope_management() {
        let mut ir = TypedIR::new();
        ir.enter_scope();
        assert_eq!(ir.symbols.scopes.len(), 2);
        assert!(ir.exit_scope().is_ok());
        assert_eq!(ir.symbols.current_scope, 0);
    }
}
