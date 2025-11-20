"""
Generate all figures from PlantUML source files
Requires PlantUML JAR file and Java installed

Installation:
1. Install Java: https://www.java.com/download/
2. Download PlantUML JAR: https://plantuml.com/download
3. Place plantuml.jar in this directory OR set PLANTUML_JAR environment variable

Usage:
    python generate_plantuml_figures.py

Output:
    - system_architecture.pdf (Figure 2)
    - transfer_learning_flow.pdf (Figure 3)
    - pybullet_environment.pdf (Figure 4)
"""

import os
import subprocess
import sys

# Configuration
PAPER_DIR = 'D:/rl_tuned_mpc/paper'
FIGURES_DIR = 'D:/rl_tuned_mpc/paper/figures'

# PlantUML JAR location (try environment variable first)
PLANTUML_JAR = os.environ.get('PLANTUML_JAR', 'D:/rl_tuned_mpc/paper/plantuml.jar')

# Figure definitions
FIGURES = [
    {
        'name': 'Figure 2: System Architecture',
        'source': 'figure_2_architecture.puml',
        'output': 'system_architecture'
    },
    {
        'name': 'Figure 3: Transfer Learning Flow',
        'source': 'figure_3_transfer.puml',
        'output': 'transfer_learning_flow'
    },
    {
        'name': 'Figure 4: PyBullet Environment',
        'source': 'figure_4_pybullet.puml',
        'output': 'pybullet_environment'
    }
]

def check_java():
    """Check if Java is installed"""
    try:
        result = subprocess.run(['java', '-version'],
                              capture_output=True,
                              text=True,
                              timeout=5)
        return result.returncode == 0
    except:
        return False

def check_plantuml():
    """Check if PlantUML JAR exists"""
    return os.path.exists(PLANTUML_JAR)

def download_plantuml():
    """Provide instructions to download PlantUML"""
    print("\n" + "="*70)
    print("PlantUML JAR not found!")
    print("="*70)
    print("\nPlease download PlantUML:")
    print("1. Visit: https://plantuml.com/download")
    print("2. Download 'plantuml.jar' (click 'Download PlantUML compiled Jar')")
    print(f"3. Save it to: {PAPER_DIR}\\plantuml.jar")
    print("\nOR set environment variable:")
    print("   set PLANTUML_JAR=C:\\path\\to\\plantuml.jar")
    print("\nAlternatively, use online PlantUML renderer:")
    print("   https://www.plantuml.com/plantuml/uml/")
    print("   (Paste .puml file contents and download as PDF)")
    print("="*70 + "\n")

def generate_figure(source_file, output_name):
    """Generate PDF from PlantUML source"""
    source_path = os.path.join(PAPER_DIR, source_file)

    if not os.path.exists(source_path):
        print(f"[ERROR] Source file not found: {source_path}")
        return False

    # Generate PDF using PlantUML
    cmd = [
        'java',
        '-jar',
        PLANTUML_JAR,
        '-tpdf',  # PDF output
        '-o', FIGURES_DIR,  # Output directory
        source_path
    ]

    try:
        print(f"[*] Generating {output_name}.pdf...")
        result = subprocess.run(cmd,
                              capture_output=True,
                              text=True,
                              timeout=30)

        if result.returncode == 0:
            # PlantUML creates output with same base name as input
            generated_file = os.path.join(FIGURES_DIR,
                                         source_file.replace('.puml', '.pdf'))
            desired_file = os.path.join(FIGURES_DIR, f'{output_name}.pdf')

            # Rename if needed
            if generated_file != desired_file and os.path.exists(generated_file):
                if os.path.exists(desired_file):
                    os.remove(desired_file)
                os.rename(generated_file, desired_file)

            print(f"[OK] Generated: {output_name}.pdf")

            # Also generate PNG for preview
            cmd_png = cmd.copy()
            cmd_png[2] = '-tpng'  # PNG output
            cmd_png[3] = '-tpng'
            subprocess.run(cmd_png, capture_output=True, timeout=30)

            generated_png = os.path.join(FIGURES_DIR,
                                        source_file.replace('.puml', '.png'))
            desired_png = os.path.join(FIGURES_DIR, f'{output_name}.png')
            if generated_png != desired_png and os.path.exists(generated_png):
                if os.path.exists(desired_png):
                    os.remove(desired_png)
                os.rename(generated_png, desired_png)

            print(f"[OK] Generated: {output_name}.png (preview)")
            return True
        else:
            print(f"[ERROR] PlantUML failed:")
            print(result.stderr)
            return False

    except subprocess.TimeoutExpired:
        print(f"[ERROR] PlantUML generation timed out")
        return False
    except Exception as e:
        print(f"[ERROR] {str(e)}")
        return False

def main():
    print("="*70)
    print("PlantUML Figure Generator for AAAI Paper")
    print("="*70)
    print()

    # Ensure figures directory exists
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # Check prerequisites
    print("[*] Checking prerequisites...")

    if not check_java():
        print("[ERROR] Java not found!")
        print("\nPlease install Java:")
        print("  Download from: https://www.java.com/download/")
        print("  Or use: winget install Oracle.JavaRuntimeEnvironment")
        return False
    print("[OK] Java found")

    if not check_plantuml():
        download_plantuml()
        return False
    print(f"[OK] PlantUML found: {PLANTUML_JAR}")
    print()

    # Generate all figures
    success_count = 0
    for fig in FIGURES:
        print(f"{'='*70}")
        print(f"Generating {fig['name']}")
        print(f"{'='*70}")
        if generate_figure(fig['source'], fig['output']):
            success_count += 1
        print()

    # Summary
    print("="*70)
    if success_count == len(FIGURES):
        print(f"[SUCCESS] All {len(FIGURES)} figures generated successfully!")
    else:
        print(f"[PARTIAL] {success_count}/{len(FIGURES)} figures generated")
    print("="*70)
    print()

    if success_count > 0:
        print("Generated files in figures/:")
        for fig in FIGURES:
            pdf_path = os.path.join(FIGURES_DIR, f"{fig['output']}.pdf")
            png_path = os.path.join(FIGURES_DIR, f"{fig['output']}.png")
            if os.path.exists(pdf_path):
                size_mb = os.path.getsize(pdf_path) / 1024
                print(f"  ✓ {fig['output']}.pdf ({size_mb:.1f} KB)")
            if os.path.exists(png_path):
                size_mb = os.path.getsize(png_path) / 1024
                print(f"  ✓ {fig['output']}.png ({size_mb:.1f} KB)")
        print()
        print("Next steps:")
        print("  1. Review generated PDFs in figures/ folder")
        print("  2. Upload to Overleaf along with main.tex")
        print("  3. Compile and verify in paper")

    return success_count == len(FIGURES)

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
