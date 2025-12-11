#!/bin/bash
# ============================================================
# Unified Setup Script for Transformer Killer Core + Mamba SSM
# ============================================================
#
# Usage:
#   ./setup.sh              # Full installation
#   ./setup.sh --check      # System check only
#   ./setup.sh --mamba      # Install Mamba SSM only
#   ./setup.sh --verify     # Verify installation
#
# Requirements:
#   - Linux with NVIDIA GPU
#   - CUDA 11.6+
#   - Python 3.8+
#
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_header() {
    echo ""
    echo -e "${BLUE}============================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}============================================================${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# ============================================================
# System Check
# ============================================================
check_system() {
    print_header "SYSTEM CHECK"
    
    # Check OS
    OS=$(uname -s)
    echo "OS: $OS"
    if [[ "$OS" != "Linux" ]]; then
        print_warning "Mamba CUDA requires Linux. Got: $OS"
        print_warning "GRU fallback will be used instead."
    fi
    
    # Check Python
    if command -v python3 &> /dev/null; then
        PY_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
        echo "Python: $PY_VERSION"
    else
        print_error "Python3 not found"
        exit 1
    fi
    
    # Check CUDA
    if command -v nvcc &> /dev/null; then
        CUDA_VERSION=$(nvcc --version | grep release | sed 's/.*release //' | sed 's/,.*//')
        echo "CUDA: $CUDA_VERSION"
    else
        print_warning "CUDA (nvcc) not found"
    fi
    
    # Check GPU
    if command -v nvidia-smi &> /dev/null; then
        GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
        echo "GPU: $GPU"
    else
        print_warning "nvidia-smi not found"
    fi
    
    # Check PyTorch
    python3 -c "import torch; print(f'PyTorch: {torch.__version__}')" 2>/dev/null || print_warning "PyTorch not installed"
    python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')" 2>/dev/null || true
    
    echo ""
}

# ============================================================
# Install Dependencies
# ============================================================
install_deps() {
    print_header "INSTALLING DEPENDENCIES"
    
    # Install PyTorch if not present
    python3 -c "import torch" 2>/dev/null || {
        echo "Installing PyTorch..."
        pip install torch>=2.0 --quiet
    }
    
    # Install other deps
    echo "Installing other dependencies..."
    pip install numpy tqdm einops --quiet
    
    # Install from requirements.txt
    if [[ -f "transformer_killer_core/requirements.txt" ]]; then
        pip install -r transformer_killer_core/requirements.txt --quiet
    fi
    
    print_success "Dependencies installed"
}

# ============================================================
# Install Mamba SSM
# ============================================================
install_mamba() {
    print_header "INSTALLING MAMBA SSM"
    
    # Check if already installed
    python3 -c "from mamba_ssm import Mamba2" 2>/dev/null && {
        print_success "Mamba SSM already installed"
        return 0
    }
    
    # Install causal-conv1d first
    echo "Installing causal-conv1d..."
    pip install causal-conv1d>=1.4.0 --quiet || true
    
    # Install mamba-ssm
    echo "Installing mamba-ssm (this may take a few minutes)..."
    pip install mamba-ssm --no-build-isolation --quiet || {
        print_warning "PyPI installation failed, trying without --no-build-isolation..."
        pip install mamba-ssm --quiet || {
            print_warning "Mamba SSM installation failed"
            print_warning "GRU fallback will be used instead"
            return 1
        }
    }
    
    print_success "Mamba SSM installed"
}

# ============================================================
# Install Mamba from Source
# ============================================================
install_mamba_source() {
    print_header "INSTALLING MAMBA SSM FROM SOURCE"
    
    if [[ -d "external/mamba_ssm" ]]; then
        echo "Building from source: external/mamba_ssm"
        cd external/mamba_ssm
        pip install -e . --no-build-isolation
        cd "$SCRIPT_DIR"
        print_success "Mamba SSM installed from source"
    else
        print_error "Source directory not found: external/mamba_ssm"
        print_warning "Falling back to PyPI installation"
        install_mamba
    fi
}

# ============================================================
# Verify Installation
# ============================================================
verify() {
    print_header "VERIFICATION"
    
    echo ""
    echo "Running sanity checks..."
    python3 -m transformer_killer_core.unified_bench --sanity_check
}

# ============================================================
# Print Usage Guide
# ============================================================
print_usage() {
    print_header "QUICK START"
    
    cat << 'EOF'

# Run sanity check
python3 -m transformer_killer_core.unified_bench --sanity_check

# Synthetic benchmark (copy memory)
python3 -m transformer_killer_core.unified_bench \
    --mode synthetic --task copy_memory \
    --controller mamba_dualmem \
    --seq_len 100 --delay 40 --epochs 20 \
    --device cuda

# Language model benchmark
python3 -m transformer_killer_core.unified_bench \
    --mode lm --controller mamba_dualmem \
    --data_path /path/to/corpus.txt \
    --seq_len 256 --epochs 10 \
    --device cuda

# Compare all controllers
for ctrl in transformer mamba mamba_dualmem ot_agent; do
    python3 -m transformer_killer_core.unified_bench \
        --mode synthetic --task copy_memory \
        --controller $ctrl --epochs 20 --device cuda
done

EOF
}

# ============================================================
# Main
# ============================================================
main() {
    case "${1:-}" in
        --check)
            check_system
            ;;
        --mamba)
            install_mamba
            ;;
        --mamba-source)
            install_mamba_source
            ;;
        --verify)
            verify
            ;;
        --usage)
            print_usage
            ;;
        --help|-h)
            echo "Usage: $0 [--check|--mamba|--mamba-source|--verify|--usage]"
            echo ""
            echo "Options:"
            echo "  --check         System check only"
            echo "  --mamba         Install Mamba SSM from PyPI"
            echo "  --mamba-source  Install Mamba SSM from local source"
            echo "  --verify        Verify installation"
            echo "  --usage         Print usage guide"
            echo "  (no args)       Full installation"
            ;;
        *)
            # Full installation
            check_system
            install_deps
            install_mamba
            verify
            print_usage
            ;;
    esac
}

main "$@"
