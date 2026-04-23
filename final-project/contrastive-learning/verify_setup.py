"""Quick setup verification script."""

import sys
from pathlib import Path

import torch


def verify_setup():
    """Verify that all dependencies are installed and GPU is available."""
    print("="*50)
    print("Contrastive Learning Setup Verification")
    print("="*50)

    # Check PyTorch
    print("\n1. Checking PyTorch...")
    print(f"   PyTorch version: {torch.__version__}")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   CUDA version: {torch.version.cuda}")
        print(f"   GPU count: {torch.cuda.device_count()}")
        if torch.cuda.device_count() > 0:
            print(f"   GPU 0: {torch.cuda.get_device_name(0)}")

    # Check torchvision
    print("\n2. Checking torchvision...")
    try:
        import torchvision
        print(f"   torchvision version: {torchvision.__version__}")
    except ImportError:
        print("   ERROR: torchvision not installed!")
        return False

    # Check datasets
    print("\n3. Checking datasets library...")
    try:
        from datasets import load_dataset
        print("   datasets library OK")
    except ImportError:
        print("   ERROR: datasets not installed!")
        return False

    # Check other dependencies
    print("\n4. Checking other dependencies...")
    try:
        import numpy
        print(f"   numpy version: {numpy.__version__}")
    except ImportError:
        print("   ERROR: numpy not installed!")
        return False

    try:
        import sklearn
        print(f"   scikit-learn version: {sklearn.__version__}")
    except ImportError:
        print("   ERROR: scikit-learn not installed!")
        return False

    try:
        import matplotlib
        print(f"   matplotlib version: {matplotlib.__version__}")
    except ImportError:
        print("   ERROR: matplotlib not installed!")
        return False

    try:
        import seaborn
        print(f"   seaborn version: {seaborn.__version__}")
    except ImportError:
        print("   ERROR: seaborn not installed!")
        return False

    # Check project structure
    print("\n5. Checking project structure...")
    required_dirs = [
        "src/data",
        "src/models",
        "src/training",
        "src/evaluation",
        "src/pipeline",
        "src/utils",
        "src/cli",
    ]

    all_exist = True
    for dir_name in required_dirs:
        dir_path = Path(dir_name)
        if dir_path.exists():
            print(f"   [OK] {dir_name}")
        else:
            print(f"   [FAILED] {dir_name} MISSING!")
            all_exist = False

    # Check main files
    required_files = [
        "main.py",
        "README.md",
        "requirements.txt",
        "pyproject.toml",
        "src/pipeline/run_contrastive.py",
    ]

    for file_name in required_files:
        file_path = Path(file_name)
        if file_path.exists():
            print(f"   [OK] {file_name}")
        else:
            print(f"   [FAILED] {file_name} MISSING!")
            all_exist = False

    print("\n" + "="*50)
    if all_exist:
        print("[OK] Setup verification passed!")
        print("You can now run: python main.py --help")
        return True
    else:
        print("[FAILED] Setup verification failed!")
        print("Please install missing dependencies: pip install -r requirements.txt")
        return False


if __name__ == "__main__":
    success = verify_setup()
    sys.exit(0 if success else 1)
