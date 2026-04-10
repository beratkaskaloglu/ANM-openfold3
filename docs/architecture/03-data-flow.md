# OpenFold3 Data Flow

## End-to-End Data Pipeline

```mermaid
flowchart TB
    subgraph PREP["Data Preparation"]
        Q[JSON Query] --> V[Validator<br/>Pydantic]
        V --> MSA_GEN[MSA Generation<br/>ColabFold / JackHMMER]
        V --> TMPL[Template Search]
        MSA_GEN --> FE[Feature Engineering]
        TMPL --> FE
    end

    subgraph EMBED["Embedding"]
        FE --> IE[Input Embedder]
        IE --> SR0[single_repr<br/>N x d_single]
        IE --> PR0[pair_repr<br/>N x N x d_pair]
        FE --> TE[Template Embedder]
        TE --> PR0
    end

    subgraph TRUNK["Iterative Trunk"]
        SR0 --> MSA[MSA Module]
        PR0 --> MSA
        MSA --> PF[PairFormer]
        PF --> SR1[refined single_repr]
        PF --> PR1[refined pair_repr]
    end

    subgraph DIFF["Diffusion Rollout"]
        SR1 --> DC[Diffusion Conditioning]
        PR1 --> DC
        DC --> DT[Diffusion Transformer]
        NOISE[Gaussian Noise<br/>x_noisy] --> DT
        DT --> DENOISE[Denoised Positions]
        DENOISE -->|T iterations| DT
    end

    subgraph OUT["Output"]
        DENOISE --> COORDS[Atom Coordinates<br/>x, y, z per atom]
        SR1 --> HEADS[Auxiliary Heads]
        PR1 --> HEADS
        HEADS --> CONF[Confidence Scores]
        COORDS --> WRITER[OutputWriter]
        CONF --> WRITER
        WRITER --> CIF[mmCIF Files]
        WRITER --> JSON_OUT[Confidence JSON]
    end

    style PREP fill:#1e272e,stroke:#686de0,color:#fff
    style EMBED fill:#1e272e,stroke:#22a6b3,color:#fff
    style TRUNK fill:#1e272e,stroke:#f0932b,color:#fff
    style DIFF fill:#1e272e,stroke:#eb4d4b,color:#fff
    style OUT fill:#1e272e,stroke:#6ab04c,color:#fff
```

## Tensor Shapes (Approximate)

| Tensor | Shape | Açıklama |
|--------|-------|----------|
| `single_repr` | `[B, N_tokens, d_single]` | Per-token representation |
| `pair_repr` | `[B, N_tokens, N_tokens, d_pair]` | Pairwise feature map |
| `msa_repr` | `[B, N_msa, N_tokens, d_msa]` | MSA sequence embeddings |
| `atom_positions` | `[B, N_atoms, 3]` | 3D koordinatlar |
| `plddt_logits` | `[B, N_tokens, n_bins]` | Per-residue confidence |
| `pae_logits` | `[B, N_tokens, N_tokens, n_bins]` | Aligned error map |

## Data Processing Pipeline

```
core/data/
├── framework/     # Temel soyutlamalar
├── io/            # Dosya I/O (mmCIF, PDB parse)
├── pipelines/     # Processing zinciri
├── primitives/    # Veri yapıları (Feature types)
├── resources/     # Harici veri kaynakları (CCD, templates)
└── tools/         # Yardımcı fonksiyonlar
```

## Confidence Score Computation

```mermaid
flowchart LR
    SR[single_repr] --> H1[pLDDT Head]
    PR[pair_repr] --> H2[PAE Head]
    PR --> H3[pDE Head]

    H1 --> pLDDT[pLDDT<br/>0-100 per residue]
    H2 --> PAE_OUT[PAE<br/>NxN error matrix]
    H3 --> pDE_OUT[pDE<br/>Distance error]

    pLDDT --> AGG[Aggregation]
    PAE_OUT --> AGG
    pDE_OUT --> AGG
    AGG --> pTM[pTM score<br/>Global quality]
```

### Confidence Metrikleri

| Metrik | Açıklama | Kullanım |
|--------|----------|----------|
| **pLDDT** | Per-residue local distance difference test | Lokal yapı güvenilirliği |
| **PAE** | Predicted Aligned Error | Domain-domain ilişki güvenilirliği |
| **pDE** | Predicted Distance Error | Mesafe tahmin hatası |
| **pTM** | Predicted Template Modeling | Global yapı kalitesi |

## Memory Optimization

- **Activation Checkpointing**: Trunk iterasyonlarında bellek tasarrufu
- **Chunked Confidence**: Büyük atom sayılarında parçalı hesaplama
- **Selective Offloading**: GPU ↔ CPU transfer
- **DeepSpeed ZeRO**: Multi-GPU bellek optimizasyonu

## Related
- [[01-openfold3-inference-pipeline]] - Pipeline overview
- [[02-model-architecture]] - Model detayları

#openfold3 #data-flow #tensors
