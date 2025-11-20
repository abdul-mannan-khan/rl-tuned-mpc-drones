#!/bin/bash

echo "========================================"
echo "  Compiling AAAI Paper"
echo "========================================"
echo ""

echo "[1/4] First pdflatex pass..."
pdflatex -interaction=nonstopmode main.tex
if [ $? -ne 0 ]; then
    echo "ERROR: First pdflatex pass failed"
    exit 1
fi

echo ""
echo "[2/4] Running bibtex..."
bibtex main
if [ $? -ne 0 ]; then
    echo "WARNING: bibtex had issues, continuing..."
fi

echo ""
echo "[3/4] Second pdflatex pass..."
pdflatex -interaction=nonstopmode main.tex
if [ $? -ne 0 ]; then
    echo "ERROR: Second pdflatex pass failed"
    exit 1
fi

echo ""
echo "[4/4] Final pdflatex pass..."
pdflatex -interaction=nonstopmode main.tex
if [ $? -ne 0 ]; then
    echo "ERROR: Final pdflatex pass failed"
    exit 1
fi

echo ""
echo "========================================"
echo "  Compilation Complete!"
echo "========================================"
echo ""
echo "Output file: main.pdf"
echo ""

# Clean up auxiliary files
echo "Cleaning up auxiliary files..."
rm -f main.aux main.log main.out main.bbl main.blg

echo "Done!"
