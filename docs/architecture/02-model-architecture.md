# OpenFold3 Model Architecture

## Overview

OpenFold3, AlphaFold3'ün açık kaynak replikasyonudur. Diffusion tabanlı all-atom yapı tahmini yapar.

## Architecture Diagram

```mermaid
flowchart TB
    subgraph INPUT["1. Input Processing"]
        IE[InputEmbedderAllAtom<br/>21.5 KB]
        TE[TemplateEmbedderAllAtom<br/>4.9 KB]
    end

    subgraph TRUNK["2. Trunk - Iterative Refinement"]
        direction TB
        MSA[MSA Module Stack<br/>13.7 KB]
        TM[Template Module<br/>19.9 KB]
        PF[PairFormer Stack<br/>17 KB]

        MSA --> PF
        TM --> PF
    end

    subgraph ROLLOUT["3. Rollout - Structure Generation"]
        DM[Diffusion Module<br/>13.5 KB]
    end

    subgraph HEADS["4. Auxiliary Heads"]
        AH[AuxiliaryHeadsAllAtom<br/>21.3 KB]

        pLDDT[pLDDT<br/>Per-residue confidence]
        PAE[PAE<br/>Predicted Aligned Error]
        pDE[pDE<br/>Predicted Distance Error]
        pTM[pTM<br/>Template Modeling Score]

        AH --> pLDDT
        AH --> PAE
        AH --> pDE
        AH --> pTM
    end

    IE --> TRUNK
    TE --> TRUNK
    TRUNK -->|single_repr + pair_repr| ROLLOUT
    ROLLOUT -->|atom_positions| HEADS

    style INPUT fill:#2d3436,stroke:#00b894,color:#fff
    style TRUNK fill:#2d3436,stroke:#0984e3,color:#fff
    style ROLLOUT fill:#2d3436,stroke:#e17055,color:#fff
    style HEADS fill:#2d3436,stroke:#fdcb6e,color:#fff
```

## Processing Stages

### Stage 1: Input Embedding
- **InputEmbedderAllAtom** - Raw sequence, atom features, residue features
- **TemplateEmbedderAllAtom** - Structural template bilgileri (eğer varsa)
- Çıktı: `single_repr` (N x d_single) + `pair_repr` (N x N x d_pair)

### Stage 2: Trunk (Iterative)
Trunk birden fazla cycle döner:

```mermaid
flowchart LR
    S[single_repr] --> MSA_M[MSA Module]
    P[pair_repr] --> MSA_M
    MSA_M --> TM_M[Template Module]
    TM_M --> PF_M[PairFormer]
    PF_M -->|updated| S2[single_repr']
    PF_M -->|updated| P2[pair_repr']
    S2 -.->|next cycle| MSA_M
    P2 -.->|next cycle| MSA_M
```

### Stage 3: Diffusion Rollout
- İteratif noise → structure denoising
- Her adımda atom pozisyonları refine edilir
- **DiffusionTransformer** + **DiffusionConditioning** katmanları

### Stage 4: Auxiliary Heads
- Confidence metrikleri hesaplanır
- [[../modules/prediction-heads]] detaylı bilgi

## Core Layer Primitives

| Layer | Dosya | Açıklama |
|-------|-------|----------|
| Attention (Pair Bias) | `attention_pair_bias.py` | Pair representation ile bias edilmiş attention |
| Triangular Attention | `triangular_attention.py` | Üçgen güncelleme ile attention |
| Triangular Multiplicative | `triangular_multiplicative_update.py` | Outgoing/incoming triangular update |
| Outer Product Mean | `outer_product_mean.py` | MSA → pair projection |
| Sequence Local Atom Attn | `sequence_local_atom_attention.py` | Atom seviyesinde lokal attention |
| Diffusion Transformer | `diffusion_transformer.py` | Diffusion step transformer |
| Diffusion Conditioning | `diffusion_conditioning.py` | Noise schedule conditioning |
| Transition | `transition.py` | Feed-forward transition layers |

## Primitive Building Blocks

```
core/model/primitives/
├── attention.py       (26 KB)  # Multi-head attention variants
├── linear.py          (5 KB)   # Custom linear layers
├── normalization.py   (4.3 KB) # LayerNorm variants
├── activations.py     (1.9 KB) # SiLU, GELU etc.
├── dropout.py         (2.4 KB) # Structured dropout
└── initialization.py  (2.2 KB) # Weight init schemes
```

## Latent Module Details

```
core/model/latent/
├── base_blocks.py     (17.3 KB) # Temel transformer blokları
├── base_stacks.py     (10.2 KB) # Stack kompozisyonları
├── evoformer.py       (12.8 KB) # Evoformer (sequence evolution)
├── msa_module.py      (13.7 KB) # MSA processing
├── pairformer.py      (17 KB)   # Pair transformations
└── template_module.py (20 KB)   # Template integration
```

## Related
- [[01-openfold3-inference-pipeline]] - Pipeline genel görünüm
- [[03-data-flow]] - Veri akışı detayları
- [[../modules/diffusion-module]] - Diffusion detayları

#openfold3 #architecture #model
