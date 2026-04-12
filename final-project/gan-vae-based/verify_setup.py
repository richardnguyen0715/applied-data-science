#!/usr/bin/env python3
"""Verify project setup and dependencies."""

import sys
import importlib.util
from pathlib import Path


def check_module(name: str, display_name: str = None) -> bool:
    """Check if module is installed."""
    display_name = display_name or name
    try:
        spec = importlib.util.find_spec(name)
        if spec is not None:
            print(f"✓ {display_name:<30} installed")
            return True
        else:
            print(f"✗ {display_name:<30} NOT installed")
            return False
    except ImportError:
        print(f"✗ {display_name:<30} NOT installed")
        return False


def check_file_structure() -> bool:
    """Check if required files/folders exist."""
    required_dirs = [
        "src/data",
        "src/models",
        "src/training",
        "src/evaluation",
        "src/pipeline",
        "src/utils",
        "src/cli",
        "configs",
        "docs",
    ]

    required_files = [
        "README.md",
        "requirements.txt",
        "pyproject.toml",
        ".gitignore",
        "configs/default.yaml",
        "configs/gan.yaml",
        "configs/vae.yaml",
        "src/data/dataset.py",
        "src/models/gan.py",
        "src/models/vae.py",
        "src/training/train_gan.py",
        "src/training/train_vae.py",
    ]

    all_ok = True

    print("\nChecking directory structure...")
    for dir_path in required_dirs:
        if Path(dir_path).exists() and Path(dir_path).is_dir():
            print(f"✓ {dir_path}/")
        else:
            print(f"✗ {dir_path}/ NOT FOUND")
            all_ok = False

    print("\nChecking required files...")
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✓ {file_path}")
        else:
            print(f"✗ {file_path} NOT FOUND")
            all_ok = False

    return all_ok


def main():
    """Run all checks."""
    print("=" * 60)
    print("PROJECT SETUP VERIFICATION")
    print("=" * 60)

    print("\nChecking Python version...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 10:
        print(f"✓ Python {version.major}.{version.minor}.{version.micro}")
    else:
        print(f"✗ Python {version.major}.{version.minor} (required >= 3.10)")

    print("\nChecking dependencies...")
    dependencies = [
        ("torch", "PyTorch"),
        ("torchvision", "TorchVision"),
        ("datasets", "HuggingFace Datasets"),
        ("yaml", "PyYAML"),
        ("numpy", "NumPy"),
        ("sklearn", "Scikit-learn"),
        ("matplotlib", "Matplotlib"),
        ("PIL", "Pillow"),
    ]

    all_deps_ok = True
    for module, display in dependencies:
        if not check_module(module, display):
            all_deps_ok = False

    # Check file structure
    structure_ok = check_file_structure()

    print("\n" + "=" * 60)
    if all_deps_ok and structure_ok:
        print("✓ SETUP VERIFICATION PASSED")
        print("\nYou can now run pipelines:")
        print("  python -m src.cli.run_baseline")
        print("  python -m src.cli.run_gan")
        print("  python -m src.cli.run_vae")
        print("\nOr run all at once:")
        print("  bash src/cli/run_all.sh")
        return 0
    else:
        print("✗ SETUP VERIFICATION FAILED")
        if not all_deps_ok:
            print("\nMissing dependencies. Install with:")
            print("  pip install -r requirements.txt")
        if not structure_ok:
            print("\nMissing files or directories. Check project structure.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
