# OpenFold3 Installation Guide

## Prerequisites
- Python >= 3.10
- CUDA toolkit (NVIDIA GPU) veya ROCm (AMD GPU)
- Conda/Mamba
- Git

## Step-by-Step

### 1. Conda Environment
```bash
conda activate ANM-openfold
```

### 2. Core Dependencies
OpenFold3 pyproject.toml'dan otomatik yüklenir:

| Paket | Amaç |
|-------|------|
| torch | ML framework |
| pytorch-lightning >= 2.1 | Training/inference orchestration |
| deepspeed | Multi-GPU optimization |
| numpy, scipy, pandas | Scientific computing |
| biotite | Biyoinformatik veri yapıları |
| rdkit | Molecular informatics |
| pdbeccdutils | Chemical component dictionary |
| kalign-python | MSA alignment |
| lmdb | Database storage |
| boto3, awscli | Model parameter download |

### 3. Install
```bash
# From PyPI
pip install openfold3

# VEYA source'dan
cd /Users/berat/Projects/ANM-openfold3/openfold3-repo
pip install -e ".[dev]"
```

### 4. Model Parameters
```bash
setup_openfold
# İndirir: ~/.openfold3/openfold3-p2-155k/
```

### 5. Verify
```bash
python -c "import openfold3; print(openfold3.__version__)"
run_openfold predict --help
```

### 6. ANM Dependencies (Gelecek)
```bash
pip install prody mdanalysis biopython
```

## Troubleshooting

### CUDA Version Mismatch
```bash
# Check CUDA
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

### Memory Issues
- `--deepspeed_config_path` ile DeepSpeed kullan
- Küçük batch size
- Activation checkpointing

## Related
- [[conda-setup]] - Conda env detayları
- [[../architecture/01-openfold3-inference-pipeline]] - Pipeline

#setup #installation #dependencies
