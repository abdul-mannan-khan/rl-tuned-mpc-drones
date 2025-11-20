@echo off
echo ========================================
echo   Compiling AAAI Paper
echo ========================================
echo.

echo [1/4] First pdflatex pass...
pdflatex -interaction=nonstopmode main.tex
if errorlevel 1 (
    echo ERROR: First pdflatex pass failed
    pause
    exit /b 1
)

echo.
echo [2/4] Running bibtex...
bibtex main
if errorlevel 1 (
    echo WARNING: bibtex had issues, continuing...
)

echo.
echo [3/4] Second pdflatex pass...
pdflatex -interaction=nonstopmode main.tex
if errorlevel 1 (
    echo ERROR: Second pdflatex pass failed
    pause
    exit /b 1
)

echo.
echo [4/4] Final pdflatex pass...
pdflatex -interaction=nonstopmode main.tex
if errorlevel 1 (
    echo ERROR: Final pdflatex pass failed
    pause
    exit /b 1
)

echo.
echo ========================================
echo   Compilation Complete!
echo ========================================
echo.
echo Output file: main.pdf
echo.

REM Clean up auxiliary files
echo Cleaning up auxiliary files...
del /Q main.aux main.log main.out main.bbl main.blg 2>nul

echo Done!
pause
