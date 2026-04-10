# Conda Environment Setup

## Environment: ANM-openfold

### Oluşturma
```bash
conda create -n ANM-openfold python=3.11 -y
conda activate ANM-openfold
```

### OpenFold3 Kurulumu
```bash
# Option 1: pip install (recommended)
pip install openfold3

# Option 2: Source'dan (development)
cd openfold3-repo
pip install -e .
```

### Model Parametreleri
```bash
setup_openfold  # ~/.openfold3/ altına indirir
```

### Checkpoints
| Model | Açıklama |
|-------|----------|
| `openfold3-p2-155k` | Default, en güncel |
| `openfold3-p2-145k` | Alternatif |
| `openfold3-p1` | Legacy (<0.4) |

### Donanım Gereksinimleri
- **GPU**: NVIDIA CUDA destekli veya AMD ROCm
- **VRAM**: Minimum 16GB (büyük yapılar için 40GB+)
- **RAM**: 32GB+ önerilir
- **Disk**: ~10GB (model parametreleri)

### Test
```bash
run_openfold predict --query_json=examples/example_inference_inputs/query_ubiquitin.json
```

## Related
- [[installation]] - Detaylı kurulum adımları
- [[../architecture/01-openfold3-inference-pipeline]] - Inference pipeline

#setup #conda #installation
