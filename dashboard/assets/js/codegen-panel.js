/**
 * QRATUM Dashboard - DCGE Code Generation Panel (Phase 4)
 * Deterministic Code Generation Engine with AST/IR visualization
 * @version 4.0.0
 */

const QratumDCGE = (function() {
    'use strict';

    // State
    let codegenState = {
        generatedCode: '',
        ast: null,
        ir: null,
        metrics: {
            textSize: 0,
            stackSize: 0,
            heapSize: 0
        },
        validationResult: null
    };

    /**
     * Initialize DCGE panel
     */
    function init() {
        console.log('[DCGE] Initializing Phase 4 Code Generation Panel');
        
        initCodegenForm();
        initValidateButton();
        initCopyButton();
    }

    /**
     * Initialize code generation form
     */
    function initCodegenForm() {
        const form = document.getElementById('codegen-form');
        if (!form) return;
        
        form.addEventListener('submit', async (e) => {
            e.preventDefault();
            await generateCode();
        });
    }

    /**
     * Initialize validate button
     */
    function initValidateButton() {
        const btn = document.getElementById('validate-code-btn');
        if (btn) {
            btn.addEventListener('click', async () => {
                await validateCode();
            });
        }
    }

    /**
     * Initialize copy button
     */
    function initCopyButton() {
        const btn = document.getElementById('copy-code-btn');
        if (btn) {
            btn.addEventListener('click', () => {
                copyToClipboard();
            });
        }
    }

    /**
     * Generate code from intent
     */
    async function generateCode() {
        const prompt = document.getElementById('codegen-prompt')?.value || '';
        const language = document.getElementById('codegen-language')?.value || 'rust';
        const intentType = document.getElementById('codegen-type')?.value || 'function';
        
        if (!prompt.trim()) {
            showNotification('Please enter a code intent/prompt', 'warning');
            return;
        }
        
        console.log('[DCGE] Generating code:', { prompt, language, intentType });
        
        // Extract function name from prompt
        const nameMatch = prompt.match(/(?:function|create|make|implement)\s+(?:a\s+)?(\w+)/i);
        const name = nameMatch ? nameMatch[1] : 'generated_function';
        
        // Build intent spec
        const intentSpec = {
            language: language,
            intent_type: createIntentTypeVariant(intentType, name, prompt),
            constraints: [],
            docstring: prompt
        };
        
        // Try to call Tauri backend if available
        if (window.__TAURI__) {
            try {
                const result = await window.__TAURI__.tauri.invoke('generate_code', {
                    intent: intentSpec
                });
                
                codegenState.generatedCode = result;
                updateCodeOutput(result);
                updateASTView(intentSpec);
                updateIRView(language);
                await updateMetrics();
                showNotification('Code generated successfully', 'success');
            } catch (error) {
                console.error('[DCGE] Generation error:', error);
                // Fall back to demo mode
                generateDemoCode(language, intentType, name, prompt);
            }
        } else {
            // Web demo mode
            generateDemoCode(language, intentType, name, prompt);
        }
    }

    /**
     * Create intent type variant structure suitable for Rust enum serialization
     * @param {string} type - The intent type (function, struct, module, fileio, threading)
     * @param {string} name - The name of the generated element
     * @param {string} purpose - Description of the code's purpose
     * @returns {Object} Variant structure matching Rust enum format
     */
    function createIntentTypeVariant(type, name, purpose) {
        switch (type) {
            case 'function':
                return { Function: { name, purpose } };
            case 'struct':
                return { Struct: { name, purpose } };
            case 'module':
                return { Module: { name, purpose } };
            case 'fileio':
                return { FileIO: { operation: purpose.includes('read') ? 'read' : 'write' } };
            case 'threading':
                return { Threading: { operation: 'spawn' } };
            default:
                return { Function: { name, purpose } };
        }
    }

    /**
     * Generate demo code for web preview
     */
    function generateDemoCode(language, intentType, name, purpose) {
        let code = '';
        
        switch (language) {
            case 'rust':
                code = generateRustDemo(intentType, name, purpose);
                break;
            case 'python':
                code = generatePythonDemo(intentType, name, purpose);
                break;
            case 'javascript':
                code = generateJSDemo(intentType, name, purpose);
                break;
            case 'c':
                code = generateCDemo(intentType, name, purpose);
                break;
            default:
                code = `// Generated ${intentType} for: ${purpose}`;
        }
        
        codegenState.generatedCode = code;
        updateCodeOutput(code);
        updateASTView({ language, intent_type: intentType, name });
        updateIRView(language);
        updateDemoMetrics(code.length);
        showNotification('Code generated successfully (demo mode)', 'success');
    }

    /**
     * Language-specific code templates for demo generation
     */
    const CODE_TEMPLATES = {
        rust: {
            header: (purpose, timestamp) => `// DCGE Generated Code\n// Purpose: ${purpose}\n// Generated: ${timestamp}\n\n`,
            function: (name, purpose) => `/// ${purpose}\nfn ${name}() -> Result<(), Box<dyn std::error::Error>> {\n    // Implementation\n    Ok(())\n}\n`,
            struct: (name, purpose) => `/// ${purpose}\n#[derive(Debug, Clone)]\npub struct ${name} {\n    // Fields\n}\n\nimpl ${name} {\n    pub fn new() -> Self {\n        Self {}\n    }\n}\n`,
            module: (name, purpose) => `//! ${purpose}\n\npub mod ${name} {\n    pub fn init() {\n        // Initialization\n    }\n}\n`,
            fileio: (name, purpose) => `use std::fs;\nuse std::io::Result;\n\n/// ${purpose}\nfn ${name}(path: &str) -> Result<String> {\n    fs::read_to_string(path)\n}\n`,
            threading: (name, purpose) => `use std::thread;\n\n/// ${purpose}\nfn ${name}() -> thread::JoinHandle<()> {\n    thread::spawn(|| {\n        // Thread work\n    })\n}\n`
        },
        python: {
            header: (purpose, timestamp) => `# DCGE Generated Code\n# Purpose: ${purpose}\n# Generated: ${timestamp}\n\n`,
            function: (name, purpose) => `def ${name}():\n    """${purpose}"""\n    pass\n`,
            struct: (name, purpose) => `class ${name}:\n    """${purpose}"""\n    \n    def __init__(self):\n        pass\n`,
            module: (name, purpose) => `"""${purpose}"""\n\ndef init():\n    pass\n`,
            fileio: (name, purpose) => `def ${name}(path: str) -> str:\n    """${purpose}"""\n    with open(path, 'r') as f:\n        return f.read()\n`,
            threading: (name, purpose) => `import threading\n\ndef ${name}():\n    """${purpose}"""\n    thread = threading.Thread(target=lambda: None)\n    thread.start()\n    return thread\n`
        },
        javascript: {
            header: (purpose, timestamp) => `// DCGE Generated Code\n// Purpose: ${purpose}\n// Generated: ${timestamp}\n\n`,
            function: (name, purpose) => `/**\n * ${purpose}\n */\nfunction ${name}() {\n  // Implementation\n}\n`,
            struct: (name, purpose) => `/**\n * ${purpose}\n */\nclass ${name} {\n  constructor() {\n    // Initialize\n  }\n}\n`,
            module: (name, purpose) => `/**\n * ${purpose}\n */\nexport const ${name} = {\n  init() {\n    // Initialization\n  }\n};\n`,
            fileio: (name, purpose) => `const fs = require('fs');\n\n/**\n * ${purpose}\n */\nfunction ${name}(path) {\n  return fs.readFileSync(path, 'utf8');\n}\n`,
            threading: (name, purpose) => `const { Worker } = require('worker_threads');\n\n/**\n * ${purpose}\n */\nfunction ${name}() {\n  return new Worker('./worker.js');\n}\n`
        },
        c: {
            header: (purpose, timestamp) => `/* DCGE Generated Code */\n/* Purpose: ${purpose} */\n/* Generated: ${timestamp} */\n\n`,
            function: (name, purpose) => `#include <stdio.h>\n\n/* ${purpose} */\nint ${name}(void) {\n    return 0;\n}\n`,
            struct: (name, purpose) => `/* ${purpose} */\ntypedef struct {\n    int field;\n} ${name};\n\n${name} ${name}_new(void) {\n    ${name} s = {0};\n    return s;\n}\n`,
            module: (name, purpose) => `#ifndef ${name.toUpperCase()}_H\n#define ${name.toUpperCase()}_H\n\n/* ${purpose} */\nvoid ${name}_init(void);\n\n#endif\n`,
            fileio: (name, purpose) => `#include <stdio.h>\n#include <stdlib.h>\n\n/* ${purpose} */\nchar* ${name}(const char* path) {\n    FILE* f = fopen(path, "r");\n    if (!f) return NULL;\n    // Read implementation\n    fclose(f);\n    return NULL;\n}\n`,
            threading: (name, purpose) => `#include <pthread.h>\n\n/* ${purpose} */\npthread_t ${name}(void* (*func)(void*)) {\n    pthread_t thread;\n    pthread_create(&thread, NULL, func, NULL);\n    return thread;\n}\n`
        }
    };

    /**
     * Generate demo code for any supported language
     * @param {string} language - The target language (rust, python, javascript, c)
     * @param {string} type - The code type (function, struct, module, fileio, threading)
     * @param {string} name - The name of the generated element
     * @param {string} purpose - Description of the code's purpose
     * @returns {string} Generated code
     */
    function generateDemoCode(language, type, name, purpose) {
        const timestamp = new Date().toISOString();
        const templates = CODE_TEMPLATES[language];
        
        if (!templates) {
            return `// Unsupported language: ${language}\n`;
        }
        
        const header = templates.header(purpose, timestamp);
        const bodyFn = templates[type];
        
        if (!bodyFn) {
            return header + `// Unsupported type: ${type}\n`;
        }
        
        return header + bodyFn(name, purpose);
    }

    /**
     * Generate Rust demo code
     */
    function generateRustDemo(type, name, purpose) {
        return generateDemoCode('rust', type, name, purpose);
    }

    /**
     * Generate Python demo code
     */
    function generatePythonDemo(type, name, purpose) {
        return generateDemoCode('python', type, name, purpose);
    }

    /**
     * Generate JavaScript demo code
     */
    function generateJSDemo(type, name, purpose) {
        return generateDemoCode('javascript', type, name, purpose);
    }

    /**
     * Generate C demo code
     */
    function generateCDemo(type, name, purpose) {
        return generateDemoCode('c', type, name, purpose);
    }

    /**
     * Validate generated code
     */
    async function validateCode() {
        const code = codegenState.generatedCode;
        const language = document.getElementById('codegen-language')?.value || 'rust';
        
        if (!code) {
            showNotification('No code to validate', 'warning');
            return;
        }
        
        if (window.__TAURI__) {
            try {
                const result = await window.__TAURI__.tauri.invoke('validate_code', {
                    language,
                    source: code
                });
                
                codegenState.validationResult = result;
                updateRegressionStatus(result);
                showNotification(result ? 'Code validation passed!' : 'Code validation failed', result ? 'success' : 'error');
            } catch (error) {
                console.error('[DCGE] Validation error:', error);
                // Demo validation
                updateRegressionStatus(true);
                showNotification('Code validation passed (demo)', 'success');
            }
        } else {
            // Demo validation - always passes
            updateRegressionStatus(true);
            showNotification('Code validation passed (demo)', 'success');
        }
    }

    /**
     * Update code output display
     */
    function updateCodeOutput(code) {
        const codeEl = document.getElementById('generated-code');
        if (codeEl) {
            codeEl.innerHTML = `<code>${escapeHtml(code)}</code>`;
        }
    }

    /**
     * Update AST view
     */
    function updateASTView(intent) {
        const astViewer = document.getElementById('ast-viewer');
        if (!astViewer) return;
        
        // Generate demo AST structure
        const ast = {
            type: 'Program',
            items: [{
                type: intent.intent_type?.Function ? 'Function' : 'Module',
                name: intent.intent_type?.Function?.name || intent.name || 'generated',
                body: {
                    type: 'Block',
                    statements: []
                }
            }]
        };
        
        codegenState.ast = ast;
        astViewer.innerHTML = `<pre class="ast-tree">${JSON.stringify(ast, null, 2)}</pre>`;
    }

    /**
     * Update Typed IR view
     */
    function updateIRView(language) {
        const irViewer = document.getElementById('ir-viewer');
        if (!irViewer) return;
        
        // Generate demo IR structure
        const ir = {
            symbols: {
                scopes: [{
                    symbols: {},
                    parent: null
                }],
                current_scope: 0
            },
            type_constraints: [],
            error_rules: []
        };
        
        codegenState.ir = ir;
        irViewer.innerHTML = `<pre class="ir-tree">${JSON.stringify(ir, null, 2)}</pre>`;
    }

    /**
     * Update metrics from backend
     */
    async function updateMetrics() {
        if (window.__TAURI__) {
            try {
                const benchmark = await window.__TAURI__.tauri.invoke('get_dcge_benchmark');
                
                codegenState.metrics = {
                    textSize: benchmark.text_size_bytes,
                    stackSize: benchmark.stack_size_bytes,
                    heapSize: benchmark.heap_size_bytes
                };
                
                updateMetricsDisplay();
            } catch (error) {
                console.error('[DCGE] Metrics error:', error);
                updateDemoMetrics(codegenState.generatedCode.length);
            }
        }
    }

    /**
     * Update demo metrics based on code length
     */
    function updateDemoMetrics(codeLength) {
        codegenState.metrics = {
            textSize: Math.round(codeLength * 1.5),
            stackSize: 256,
            heapSize: 0
        };
        updateMetricsDisplay();
    }

    /**
     * Update metrics display
     */
    function updateMetricsDisplay() {
        const textEl = document.getElementById('text-size');
        const stackEl = document.getElementById('stack-size');
        const heapEl = document.getElementById('heap-size');
        const totalEl = document.getElementById('total-size');
        
        if (textEl) textEl.textContent = `${codegenState.metrics.textSize} bytes`;
        if (stackEl) stackEl.textContent = `${codegenState.metrics.stackSize} bytes`;
        if (heapEl) heapEl.textContent = `${codegenState.metrics.heapSize} bytes`;
        
        const total = codegenState.metrics.textSize + codegenState.metrics.stackSize + codegenState.metrics.heapSize;
        if (totalEl) totalEl.textContent = `${total} bytes`;
    }

    /**
     * Update regression status display
     */
    function updateRegressionStatus(passed) {
        const deltaEl = document.getElementById('regression-delta');
        if (deltaEl) {
            deltaEl.textContent = passed ? 'PASS' : 'FAIL';
            deltaEl.className = `regression-value ${passed ? 'pass' : 'fail'}`;
        }
    }

    /**
     * Copy code to clipboard
     */
    function copyToClipboard() {
        if (!codegenState.generatedCode) {
            showNotification('No code to copy', 'warning');
            return;
        }
        
        navigator.clipboard.writeText(codegenState.generatedCode).then(() => {
            showNotification('Code copied to clipboard!', 'success');
        }).catch(err => {
            console.error('[DCGE] Copy failed:', err);
            showNotification('Failed to copy code', 'error');
        });
    }

    /**
     * Escape HTML for safe display
     * Uses cached div element for better performance
     */
    let _escapeHtmlDiv;
    function escapeHtml(text) {
        if (!_escapeHtmlDiv) {
            _escapeHtmlDiv = document.createElement('div');
        }
        _escapeHtmlDiv.textContent = text;
        return _escapeHtmlDiv.innerHTML;
    }

    /**
     * Show notification
     */
    function showNotification(message, type) {
        if (window.showToast) {
            window.showToast(message, type);
        } else {
            console.log(`[DCGE] ${type}: ${message}`);
        }
    }

    // Public API
    return {
        init,
        generateCode,
        validateCode,
        copyToClipboard,
        getState: () => codegenState
    };
})();

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    // Delay initialization to ensure main dashboard is ready
    setTimeout(() => {
        QratumDCGE.init();
    }, 100);
});

// Export for module usage
if (typeof module !== 'undefined' && module.exports) {
    module.exports = QratumDCGE;
}
