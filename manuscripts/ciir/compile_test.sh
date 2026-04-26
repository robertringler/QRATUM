#!/bin/bash
# Compilation validation script for CIIR Formal System manuscript package
# Constraint-Induced Interface Realism
# This script tests that all LaTeX documents compile successfully

set -e  # Exit on any error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================="
echo "CIIR Formal System Manuscript"
echo "LaTeX Compilation Validation"
echo "========================================="
echo ""

# Check for required tools
echo "Checking for required tools..."
command -v pdflatex >/dev/null 2>&1 || { echo "ERROR: pdflatex not found. Please install TeX Live or MiKTeX."; exit 1; }
command -v bibtex >/dev/null 2>&1 || { echo "ERROR: bibtex not found. Please install TeX Live or MiKTeX."; exit 1; }
echo "✓ pdflatex found: $(pdflatex --version | head -n1)"
echo "✓ bibtex found: $(bibtex --version | head -n1)"
echo ""

# Function to compile LaTeX document
compile_latex() {
    local filename=$1
    local use_bibtex=$2

    echo "----------------------------------------"
    echo "Compiling: $filename"
    echo "----------------------------------------"

    # First pass
    echo "Pass 1: pdflatex..."
    pdflatex -interaction=nonstopmode "$filename" > "${filename%.tex}_compile.log" 2>&1 || {
        echo "ERROR: First pdflatex pass failed"
        echo "Check ${filename%.tex}_compile.log for details"
        return 1
    }

    # BibTeX if requested
    if [ "$use_bibtex" = "true" ]; then
        echo "Running bibtex..."
        bibtex "${filename%.tex}" >> "${filename%.tex}_compile.log" 2>&1 || {
            echo "WARNING: BibTeX had warnings (may be OK if no citations)"
        }

        # Second pass
        echo "Pass 2: pdflatex..."
        pdflatex -interaction=nonstopmode "$filename" >> "${filename%.tex}_compile.log" 2>&1 || {
            echo "ERROR: Second pdflatex pass failed"
            return 1
        }

        # Third pass (for cross-references)
        echo "Pass 3: pdflatex..."
        pdflatex -interaction=nonstopmode "$filename" >> "${filename%.tex}_compile.log" 2>&1 || {
            echo "ERROR: Third pdflatex pass failed"
            return 1
        }
    fi

    # Check for output PDF
    if [ -f "${filename%.tex}.pdf" ]; then
        local pdf_size
        pdf_size=$(du -h "${filename%.tex}.pdf" | cut -f1)
        echo "✓ SUCCESS: ${filename%.tex}.pdf created ($pdf_size)"

        # Check for warnings in log
        local warnings
        warnings=$(grep -i "warning" "${filename%.tex}_compile.log" | wc -l)
        local errors
        errors=$(grep -i "^!" "${filename%.tex}_compile.log" | wc -l)

        if [ "$errors" -gt 0 ]; then
            echo "⚠ WARNING: $errors LaTeX errors found in compilation log"
        fi

        if [ "$warnings" -gt 0 ]; then
            echo "ℹ INFO: $warnings warnings found (may be OK)"
        fi

        return 0
    else
        echo "ERROR: PDF not generated"
        return 1
    fi
}

# Compile cover letter
if [ -f "cover_letter.tex" ]; then
    compile_latex "cover_letter.tex" "false"
    COVER_SUCCESS=$?
else
    echo "WARNING: cover_letter.tex not found"
    COVER_SUCCESS=1
fi
echo ""

# Compile main manuscript
if [ -f "manuscript.tex" ]; then
    compile_latex "manuscript.tex" "true"
    MANUSCRIPT_SUCCESS=$?
else
    echo "ERROR: manuscript.tex not found"
    MANUSCRIPT_SUCCESS=1
fi
echo ""

# Summary
echo "========================================="
echo "COMPILATION SUMMARY"
echo "========================================="

if [ $COVER_SUCCESS -eq 0 ]; then
    echo "✓ Cover Letter: SUCCESS"
else
    echo "✗ Cover Letter: FAILED"
fi

if [ $MANUSCRIPT_SUCCESS -eq 0 ]; then
    echo "✓ Main Manuscript: SUCCESS"
else
    echo "✗ Main Manuscript: FAILED"
fi

echo "========================================="

# Overall result
if [ $COVER_SUCCESS -eq 0 ] && [ $MANUSCRIPT_SUCCESS -eq 0 ]; then
    echo ""
    echo "✓ ALL DOCUMENTS COMPILED SUCCESSFULLY"
    echo ""
    echo "Generated PDFs:"
    ls -lh ./*.pdf 2>/dev/null || echo "  (No PDFs found - check errors above)"
    echo ""
    echo "Compilation logs:"
    ls -lh ./*_compile.log 2>/dev/null || echo "  (No logs found)"
    echo ""
    exit 0
else
    echo ""
    echo "✗ COMPILATION FAILED"
    echo ""
    echo "Please check the compilation logs for details:"
    ls -1 ./*_compile.log 2>/dev/null || echo "  (No logs found)"
    echo ""
    echo "Common issues:"
    echo "  1. Missing LaTeX packages (install texlive-full or similar)"
    echo "  2. Missing RevTeX 4.2 (install texlive-publishers)"
    echo "  3. Missing AMS packages (install texlive-science)"
    echo "  4. Missing mathrsfs/stmaryrd/bbm (install texlive-latex-extra)"
    echo "  5. Bibliography errors (check references.bib)"
    echo ""
    exit 1
fi
