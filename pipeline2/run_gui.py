#!/usr/bin/env python3
"""
Pipeline 2 - GUI Launcher

Simple launcher script for the Streamlit GUI.
Automatically opens browser and handles errors gracefully.
"""

import subprocess
import sys
from pathlib import Path


def main():
    """Launch Streamlit GUI"""
    
    print("=" * 70)
    print("🚀 LAUNCHING PIPELINE 2 GUI")
    print("=" * 70)
    print()
    
    # Get path to streamlit app
    gui_path = Path(__file__).parent / "gui" / "streamlit_app.py"
    
    if not gui_path.exists():
        print(f"✗ Error: GUI file not found at {gui_path}")
        print("   Expected: pipeline2/gui/streamlit_app.py")
        return 1
    
    print(f"✓ Found GUI at: {gui_path.name}")
    print()
    print("Starting Streamlit server...")
    print("=" * 70)
    print()
    print("📱 The GUI will open in your browser automatically")
    print("🌐 URL: http://localhost:8501")
    print()
    print("⌨️  Press Ctrl+C to stop the server")
    print()
    print("=" * 70)
    print()
    
    try:
        # Run streamlit
        subprocess.run([
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(gui_path),
            "--server.headless=true"
        ])
    except KeyboardInterrupt:
        print("\n\n" + "=" * 70)
        print("Server stopped by user")
        print("=" * 70)
        return 0
    except Exception as e:
        print(f"\n✗ Error launching GUI: {e}")
        print("\nTroubleshooting:")
        print("  1. Make sure streamlit is installed: pip install streamlit")
        print("  2. Try running directly: streamlit run gui/streamlit_app.py")
        return 1


if __name__ == "__main__":
    sys.exit(main())