# Diffusion Module

## Kaynak
`openfold3/core/model/structure/diffusion_module.py` (13.5 KB)

## Amaç
3D atom koordinatlarını iteratif diffusion denoising ile tahmin etmek.

### AlphaFold2 vs AlphaFold3/OpenFold3
- **AF2**: IPA (Invariant Point Attention) ile direkt yapı tahmini
- **AF3/OF3**: Diffusion tabanlı iteratif denoising - daha esnek, multi-modal

### İşlem Akışı

```mermaid
flowchart TD
    subgraph FORWARD["Diffusion Sampling"]
        NOISE[Gaussian Noise<br/>x_T ~ N(0, I)] --> STEP1[Denoise Step t=T]
        STEP1 --> STEP2[Denoise Step t=T-1]
        STEP2 --> DOTS[...]
        DOTS --> STEP_FINAL[Denoise Step t=0]
        STEP_FINAL --> COORDS[Final Atom Positions<br/>x_0]
    end

    subgraph DENOISE["Single Denoise Step"]
        X_T[x_noisy + single + pair] --> COND[Diffusion Conditioning]
        COND --> DT[Diffusion Transformer]
        DT --> PRED[Predicted x_0]
        PRED --> UPDATE[x_{t-1} = update(x_t, pred_x0)]
    end

    SR[single_repr] --> COND
    PR[pair_repr] --> COND

    style FORWARD fill:#2d3436,stroke:#e17055,color:#fff
    style DENOISE fill:#2d3436,stroke:#686de0,color:#fff
```

### Diffusion Conditioning
- Noise level (timestep t) bilgisi
- Single ve pair representation conditioning
- Atom-level features

### Diffusion Transformer
- Pair-biased self-attention
- Atom seviyesinde local attention
- Cross-attention between tokens and atoms

### Training vs Inference

| Aspect | Training | Inference |
|--------|----------|-----------|
| Timestep | Random t sampling | Sequential T → 0 |
| Target | Ground truth x_0 | N/A |
| Loss | MSE + bond constraints | N/A |
| Iterations | 1 step per sample | T full steps |

## ANM Entegrasyonu (Gelecek)
Diffusion module'ün çıktısı olan atom koordinatları üzerinde ANM analizi yapılabilir:
- B-factor tahmini vs ANM B-factor karşılaştırması
- pLDDT ile ANM flexibility korelasyonu
- Dynamic domain analizi

## Related
- [[pairformer]] - Önceki aşama (trunk output)
- [[prediction-heads]] - Sonraki aşama (confidence)
- [[../architecture/03-data-flow]] - Veri akışı

#openfold3 #module #diffusion #structure-prediction
