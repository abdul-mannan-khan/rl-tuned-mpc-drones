"""
Generate PlantUML figures using online PlantUML server
No Java or PlantUML installation required!
"""

import requests
import zlib
import base64
import os
import string

PAPER_DIR = 'D:/rl_tuned_mpc/paper'
FIGURES_DIR = 'D:/rl_tuned_mpc/paper/figures'

# PlantUML online server
PLANTUML_SERVER = 'http://www.plantuml.com/plantuml'

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

def plantuml_encode(plantuml_text):
    """
    Encode PlantUML text for URL (PlantUML's encoding scheme)
    """
    # Compress
    zlibbed_str = zlib.compress(plantuml_text.encode('utf-8'))
    compressed_string = zlibbed_str[2:-4]  # Remove zlib header/footer

    # Encode to base64
    b64_encoded = base64.b64encode(compressed_string)

    # Convert to PlantUML's custom encoding
    plantuml_alphabet = string.digits + string.ascii_uppercase + string.ascii_lowercase + '-_'
    standard_alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/'

    # Create translation table
    trans_table = str.maketrans(standard_alphabet, plantuml_alphabet)

    return b64_encoded.decode('utf-8').translate(trans_table)

def generate_figure_online(source_file, output_name):
    """Generate figure using online PlantUML server"""
    source_path = os.path.join(PAPER_DIR, source_file)

    if not os.path.exists(source_path):
        print(f"[ERROR] Source file not found: {source_path}")
        return False

    # Read PlantUML source
    with open(source_path, 'r', encoding='utf-8') as f:
        plantuml_code = f.read()

    print(f"[*] Encoding {source_file}...")

    # Encode for URL
    encoded = plantuml_encode(plantuml_code)

    # Generate SVG (high quality, can convert to PDF later)
    svg_url = f"{PLANTUML_SERVER}/svg/{encoded}"
    png_url = f"{PLANTUML_SERVER}/png/{encoded}"

    try:
        # Download SVG
        print(f"[*] Downloading SVG from PlantUML server...")
        response = requests.get(svg_url, timeout=30)

        if response.status_code == 200:
            svg_path = os.path.join(FIGURES_DIR, f'{output_name}.svg')
            with open(svg_path, 'wb') as f:
                f.write(response.content)
            print(f"[OK] Generated: {output_name}.svg")

            # Also download PNG
            print(f"[*] Downloading PNG from PlantUML server...")
            response_png = requests.get(png_url, timeout=30)
            if response_png.status_code == 200:
                png_path = os.path.join(FIGURES_DIR, f'{output_name}.png')
                with open(png_path, 'wb') as f:
                    f.write(response_png.content)
                print(f"[OK] Generated: {output_name}.png")

            return True
        else:
            print(f"[ERROR] Server returned status code: {response.status_code}")
            return False

    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Network error: {str(e)}")
        return False
    except Exception as e:
        print(f"[ERROR] {str(e)}")
        return False

def convert_svg_to_pdf():
    """Convert SVG to PDF using available tools"""
    print("\n" + "="*70)
    print("SVG to PDF Conversion")
    print("="*70)
    print("\nSVG files have been generated. To convert to PDF:")
    print("\n**Option 1: Online converter (easiest)**")
    print("  1. Go to: https://cloudconvert.com/svg-to-pdf")
    print("  2. Upload each .svg file from figures/ folder")
    print("  3. Download converted PDFs")
    print("  4. Save to figures/ folder")
    print("\n**Option 2: Use Inkscape (if installed)**")
    print("  Download: https://inkscape.org/")
    print("  Command: inkscape input.svg --export-filename=output.pdf")
    print("\n**Option 3: Use Overleaf directly**")
    print("  Overleaf can use .svg files directly in LaTeX!")
    print("  Just upload .svg files instead of .pdf")
    print("  Update main.tex to reference .svg instead of .pdf")
    print("\n**Option 4: Use CairoSVG (Python)**")
    print("  Install: pip install cairosvg")
    print("  Then run: python convert_svg_to_pdf.py")
    print("="*70 + "\n")

def main():
    print("="*70)
    print("Online PlantUML Figure Generator (No Java Required!)")
    print("="*70)
    print()

    # Ensure figures directory exists
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # Check internet connection
    print("[*] Checking internet connection...")
    try:
        requests.get(PLANTUML_SERVER, timeout=5)
        print("[OK] Connected to PlantUML server")
    except:
        print("[ERROR] Cannot connect to PlantUML server")
        print("Please check your internet connection")
        return False

    print()

    # Generate all figures
    success_count = 0
    for fig in FIGURES:
        print(f"{'='*70}")
        print(f"Generating {fig['name']}")
        print(f"{'='*70}")
        if generate_figure_online(fig['source'], fig['output']):
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
            svg_path = os.path.join(FIGURES_DIR, f"{fig['output']}.svg")
            png_path = os.path.join(FIGURES_DIR, f"{fig['output']}.png")
            if os.path.exists(svg_path):
                size_kb = os.path.getsize(svg_path) / 1024
                print(f"  ✓ {fig['output']}.svg ({size_kb:.1f} KB) - Vector format")
            if os.path.exists(png_path):
                size_kb = os.path.getsize(png_path) / 1024
                print(f"  ✓ {fig['output']}.png ({size_kb:.1f} KB) - Raster format")

        # Show conversion options
        convert_svg_to_pdf()

        print("Recommendation:")
        print("  → Use .svg files directly in Overleaf (best quality)")
        print("  → Or convert to PDF using online converter")
        print("  → PNG files are also available for preview")

    return success_count == len(FIGURES)

if __name__ == '__main__':
    import sys
    success = main()
    sys.exit(0 if success else 1)
