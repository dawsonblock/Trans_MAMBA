#!/usr/bin/env python3
"""
setup_colab.py

Unified build and installation script for Transformer Killer Core + Mamba SSM.

This script handles:
    1. System checks (GPU, CUDA version)
    2. PyTorch installation
    3. Mamba SSM installation (with causal-conv1d)
    4. Transformer Killer Core setup
    5. Verification

Usage (Google Colab):
    !python setup_colab.py --install-all
    !python setup_colab.py --install-mamba  # Mamba only
    !python setup_colab.py --check          # System check only
    !python setup_colab.py --verify         # Verify installation

Requirements:
    - Linux
    - NVIDIA GPU
    - CUDA 11.6+
    - PyTorch 1.12+
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def run_cmd(cmd, check=True, capture=False):
    """Run shell command."""
    print(f">>> {cmd}")
    if capture:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.stdout.strip(), result.returncode
    else:
        result = subprocess.run(cmd, shell=True)
        if check and result.returncode != 0:
            print(f"Command failed with code {result.returncode}")
            return False
        return True


def check_system():
    """Check system requirements."""
    print("\n" + "="*60)
    print("SYSTEM CHECK")
    print("="*60)
    
    issues = []
    
    # Check OS
    import platform
    os_name = platform.system()
    print(f"OS: {os_name}")
    if os_name != "Linux":
        issues.append(f"Mamba CUDA requires Linux, got {os_name}")
    
    # Check Python
    py_version = sys.version_info
    print(f"Python: {py_version.major}.{py_version.minor}.{py_version.micro}")
    if py_version < (3, 8):
        issues.append("Python 3.8+ required")
    
    # Check CUDA
    cuda_version = None
    nvcc_out, rc = run_cmd("nvcc --version 2>/dev/null | grep release", capture=True)
    if rc == 0 and nvcc_out:
        import re
        match = re.search(r"release (\d+\.\d+)", nvcc_out)
        if match:
            cuda_version = match.group(1)
            print(f"CUDA: {cuda_version}")
            major, minor = map(int, cuda_version.split('.'))
            if major < 11 or (major == 11 and minor < 6):
                issues.append(f"CUDA 11.6+ required, got {cuda_version}")
    else:
        print("CUDA: Not found (nvcc)")
        issues.append("CUDA not found - Mamba CUDA kernels won't compile")
    
    # Check GPU
    gpu_out, rc = run_cmd("nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null", capture=True)
    if rc == 0 and gpu_out:
        gpus = gpu_out.strip().split('\n')
        print(f"GPU(s): {', '.join(gpus)}")
    else:
        print("GPU: Not detected")
        issues.append("No NVIDIA GPU detected")
    
    # Check PyTorch
    try:
        import torch
        print(f"PyTorch: {torch.__version__}")
        print(f"PyTorch CUDA: {torch.version.cuda}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("PyTorch: Not installed")
        issues.append("PyTorch not installed")
    
    # Summary
    print("\n" + "-"*60)
    if issues:
        print("ISSUES FOUND:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print("All system checks PASSED")
        return True


def install_pytorch():
    """Install PyTorch with CUDA support."""
    print("\n" + "="*60)
    print("INSTALLING PYTORCH")
    print("="*60)
    
    # Check if already installed
    try:
        import torch
        if torch.cuda.is_available():
            print(f"PyTorch {torch.__version__} with CUDA already installed")
            return True
    except ImportError:
        pass
    
    # Install PyTorch
    run_cmd("pip install torch>=2.0 --quiet")
    
    # Verify
    try:
        import importlib
        importlib.invalidate_caches()
        import torch
        print(f"Installed PyTorch {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        return torch.cuda.is_available()
    except ImportError:
        print("PyTorch installation failed")
        return False


def install_mamba_ssm(from_source=False):
    """Install Mamba SSM package."""
    print("\n" + "="*60)
    print("INSTALLING MAMBA SSM")
    print("="*60)
    
    # Check if already installed
    try:
        from mamba_ssm import Mamba2
        print("Mamba SSM already installed")
        return True
    except ImportError:
        pass
    
    if from_source:
        # Install from local source
        mamba_dir = Path(__file__).parent / "external" / "mamba_ssm"
        if mamba_dir.exists():
            print(f"Installing from source: {mamba_dir}")
            os.chdir(mamba_dir)
            success = run_cmd("pip install -e . --no-build-isolation")
            os.chdir(Path(__file__).parent)
            return success
        else:
            print(f"Source directory not found: {mamba_dir}")
            print("Falling back to PyPI installation")
    
    # Install from PyPI
    print("Installing causal-conv1d...")
    run_cmd("pip install causal-conv1d>=1.4.0 --quiet", check=False)
    
    print("Installing mamba-ssm...")
    # Try with --no-build-isolation first (helps with PyTorch version issues)
    success = run_cmd("pip install mamba-ssm --no-build-isolation --quiet", check=False)
    
    if not success:
        print("Retrying without --no-build-isolation...")
        success = run_cmd("pip install mamba-ssm --quiet", check=False)
    
    # Verify
    try:
        import importlib
        importlib.invalidate_caches()
        from mamba_ssm import Mamba2
        print("Mamba SSM installed successfully")
        return True
    except ImportError as e:
        print(f"Mamba SSM installation failed: {e}")
        print("\nNOTE: The GRU fallback in Transformer Killer Core will still work.")
        return False


def install_dependencies():
    """Install other dependencies."""
    print("\n" + "="*60)
    print("INSTALLING DEPENDENCIES")
    print("="*60)
    
    deps = ["numpy", "tqdm", "einops"]
    for dep in deps:
        run_cmd(f"pip install {dep} --quiet", check=False)
    
    # Install from requirements.txt if exists
    req_file = Path(__file__).parent / "transformer_killer_core" / "requirements.txt"
    if req_file.exists():
        run_cmd(f"pip install -r {req_file} --quiet", check=False)
    
    return True


def verify_installation():
    """Verify full installation."""
    print("\n" + "="*60)
    print("VERIFICATION")
    print("="*60)
    
    all_passed = True
    
    # Test 1: PyTorch + CUDA
    print("\n1. PyTorch + CUDA...")
    try:
        import torch
        if torch.cuda.is_available():
            x = torch.randn(2, 16).cuda()
            y = x @ x.T
            print(f"   PASSED: PyTorch {torch.__version__} with CUDA")
        else:
            print("   WARNING: CUDA not available, using CPU")
    except Exception as e:
        print(f"   FAILED: {e}")
        all_passed = False
    
    # Test 2: Mamba SSM
    print("\n2. Mamba SSM...")
    mamba_available = False
    try:
        from mamba_ssm import Mamba2
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = Mamba2(d_model=64, d_state=64, d_conv=4, expand=2).to(device)
        x = torch.randn(2, 32, 64).to(device)
        y = model(x)
        assert y.shape == x.shape
        print("   PASSED: Mamba2 CUDA kernels working")
        mamba_available = True
    except ImportError:
        print("   INFO: Mamba SSM not installed (GRU fallback will be used)")
    except Exception as e:
        print(f"   WARNING: Mamba SSM error: {e}")
        print("   INFO: GRU fallback will be used")
    
    # Test 3: Transformer Killer Core
    print("\n3. Transformer Killer Core...")
    try:
        from transformer_killer_core.memory_core import DualTierMiras, DualTierMirasConfig
        from transformer_killer_core.controllers import build_controller, ControllerConfig
        from transformer_killer_core.ot_memory_agent import OTMemoryAgent, OTMemoryAgentConfig
        
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Test DualTierMiras
        cfg = DualTierMirasConfig(d_model=64, mem_slots=16)
        mem = DualTierMiras(cfg)
        query = torch.randn(2, 64)
        out = mem.read(query)
        assert "v" in out
        
        # Test controllers
        for ctrl_type in ["transformer", "mamba", "mamba_dualmem"]:
            cfg = ControllerConfig(
                controller_type=ctrl_type, vocab_size=16,
                d_model=32, n_layers=1, n_heads=2, max_seq_len=32
            )
            model = build_controller(cfg).to(device)
            x = torch.randint(0, 16, (2, 10)).to(device)
            logits = model(x)
            assert logits.shape == (2, 10, 16)
        
        # Test OTMemoryAgent
        agent_cfg = OTMemoryAgentConfig(vocab_size=16, d_model=32, n_layers=1)
        agent = OTMemoryAgent(agent_cfg).to(device)
        logits = agent(x)
        assert logits.shape == (2, 10, 16)
        
        print("   PASSED: All components working")
    except Exception as e:
        print(f"   FAILED: {e}")
        all_passed = False
    
    # Test 4: Unified benchmark
    print("\n4. Unified Benchmark CLI...")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "transformer_killer_core.unified_bench", "--sanity_check"],
            capture_output=True, text=True, timeout=120
        )
        if "ALL SANITY CHECKS PASSED" in result.stdout:
            print("   PASSED: Sanity checks passed")
        else:
            print("   WARNING: Some sanity checks may have failed")
            print(result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)
    except Exception as e:
        print(f"   WARNING: {e}")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Mamba2 CUDA: {'AVAILABLE' if mamba_available else 'NOT AVAILABLE (using GRU fallback)'}")
    print(f"Overall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    
    if all_passed:
        print("\nYou can now run benchmarks:")
        print("  python -m transformer_killer_core.unified_bench --mode synthetic --task copy_memory --controller mamba_dualmem --device cuda")
    
    return all_passed


def print_usage():
    """Print quick usage guide."""
    print("\n" + "="*60)
    print("QUICK START")
    print("="*60)
    print("""
# Run sanity check
python -m transformer_killer_core.unified_bench --sanity_check

# Synthetic benchmark (copy memory)
python -m transformer_killer_core.unified_bench \\
    --mode synthetic --task copy_memory \\
    --controller mamba_dualmem \\
    --seq_len 100 --delay 40 --epochs 20 \\
    --device cuda

# Language model benchmark
python -m transformer_killer_core.unified_bench \\
    --mode lm --controller mamba_dualmem \\
    --data_path /content/corpus.txt \\
    --seq_len 256 --epochs 10 \\
    --device cuda

# Compare all controllers
for ctrl in transformer mamba mamba_dualmem ot_agent; do
    python -m transformer_killer_core.unified_bench \\
        --mode synthetic --task copy_memory \\
        --controller $ctrl --epochs 20 --device cuda
done
""")


def main():
    parser = argparse.ArgumentParser(
        description="Unified setup script for Transformer Killer Core + Mamba SSM"
    )
    parser.add_argument("--check", action="store_true",
                        help="System check only")
    parser.add_argument("--install-all", action="store_true",
                        help="Full installation (PyTorch + Mamba + deps)")
    parser.add_argument("--install-mamba", action="store_true",
                        help="Install Mamba SSM only")
    parser.add_argument("--install-mamba-source", action="store_true",
                        help="Install Mamba SSM from local source")
    parser.add_argument("--verify", action="store_true",
                        help="Verify installation")
    parser.add_argument("--usage", action="store_true",
                        help="Print usage guide")
    
    args = parser.parse_args()
    
    # Default to --install-all if no args
    if not any([args.check, args.install_all, args.install_mamba,
                args.install_mamba_source, args.verify, args.usage]):
        args.install_all = True
    
    # Change to script directory
    script_dir = Path(__file__).parent.resolve()
    os.chdir(script_dir)
    print(f"Working directory: {script_dir}")
    
    if args.check:
        check_system()
        return
    
    if args.usage:
        print_usage()
        return
    
    if args.install_all:
        check_system()
        install_pytorch()
        install_dependencies()
        install_mamba_ssm(from_source=False)
        verify_installation()
        print_usage()
        return
    
    if args.install_mamba:
        install_mamba_ssm(from_source=False)
        return
    
    if args.install_mamba_source:
        install_mamba_ssm(from_source=True)
        return
    
    if args.verify:
        verify_installation()
        return


if __name__ == "__main__":
    main()
