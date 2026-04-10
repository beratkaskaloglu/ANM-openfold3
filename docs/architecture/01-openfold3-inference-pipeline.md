# OpenFold3 Inference Pipeline

## High-Level Flow

```mermaid
flowchart TD
    A[JSON Query Input] --> B[run_openfold.py CLI]
    B --> C[InferenceExperimentRunner]
    C --> D[PyTorch Lightning Trainer]
    D --> E[OpenFold3AllAtom Runner]

    subgraph PREDICT["predict_step()"]
        E --> F[Batch Validation]
        F --> G[Seed Management]
        G --> H[Forward Pass - model.__call__]
        H --> I[Confidence Score Computation]
        I --> J[Output Formatting]
    end

    subgraph MODEL["OpenFold3 Model Forward"]
        H --> M1[Input Embedder]
        M1 --> M2[Template Embedder]
        M2 --> M3[MSA Module Stack]
        M3 --> M4[PairFormer Stack]
        M4 --> M5[Diffusion Module]
        M5 --> M6[Auxiliary Heads]
    end

    J --> K[OF3OutputWriter]
    K --> L1[mmCIF/PDB Files]
    K --> L2[Confidence JSON]
    K --> L3[Full Metrics NPZ]

    style PREDICT fill:#1a1a2e,stroke:#e94560,color:#fff
    style MODEL fill:#0f3460,stroke:#16213e,color:#fff
```

## Entry Points

| Dosya | Boyut | Rol |
|-------|-------|-----|
| `run_openfold.py` | 7.6 KB | CLI giriş noktası (Click) |
| `entry_points/experiment_runner.py` | 32.7 KB | Training/Inference orkestrasyon |
| `entry_points/validator.py` | 18.7 KB | Pydantic tabanlı input validation |
| `of3_all_atom/runner.py` | 36.9 KB | Inference execution |
| `of3_all_atom/model.py` | 29 KB | Ana model mimarisi |

## CLI Commands

```bash
# Inference
run_openfold predict --query_json=query.json

# MSA Alignment
run_openfold align-msa --query_json=query.json

# Training
run_openfold train --config=config.yaml
```

## Distributed Inference

```mermaid
flowchart LR
    CLI[run_openfold] --> DET{Distributed?}
    DET -->|Single GPU| SG[Standard PyTorch]
    DET -->|Multi-GPU| DDP[DDP Strategy]
    DET -->|DeepSpeed| DS[DeepSpeed ZeRO]
    DET -->|MPI| MPI[MPI Launch]

    SG --> TR[Lightning Trainer]
    DDP --> TR
    DS --> TR
    MPI --> TR
```

## Input Format

```json
{
  "queries": {
    "protein_name": {
      "chains": [
        {
          "molecule_type": "protein",
          "chain_ids": ["A"],
          "sequence": "MQIFVKTLTGK..."
        }
      ]
    }
  }
}
```

**Desteklenen Molekül Tipleri:**
- Protein (tek/multi-chain)
- DNA
- RNA
- Small molecules / Ligands
- PTM (post-translational modifications)

## Output Structure

```
output_dir/
├── query_id/
│   ├── seed_0/
│   │   ├── sample_0/
│   │   │   ├── prediction.cif      # mmCIF yapı dosyası
│   │   │   ├── confidence.json     # pLDDT, PAE, pDE, pTM
│   │   │   └── full_confidence.npz # Tam matrisler
│   │   └── sample_1/
│   └── seed_1/
```

## Related
- [[02-model-architecture]] - Model iç mimarisi
- [[03-data-flow]] - Detaylı veri akışı
- [[../modules/prediction-heads]] - Confidence metrikleri

#openfold3 #inference #pipeline
