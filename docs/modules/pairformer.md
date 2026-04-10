# PairFormer

## Kaynak
`openfold3/core/model/latent/pairformer.py` (17 KB)

## Sınıf: PairFormerStack

Pair representation'ı iteratif olarak refine eden transformer stack.

### Neden Önemli
AlphaFold3'ün temel yeniliği: Evoformer yerine PairFormer kullanımı. Pair representation üzerinden yapısal bilgiyi daha etkin işler.

### İşlem Akışı

```mermaid
flowchart TD
    subgraph BLOCK["PairFormer Block (x N_blocks)"]
        TRI_MUL_OUT[Triangular Multiplicative<br/>Outgoing] --> TRI_MUL_IN[Triangular Multiplicative<br/>Incoming]
        TRI_MUL_IN --> TRI_ATT_START[Triangular Attention<br/>Starting Node]
        TRI_ATT_START --> TRI_ATT_END[Triangular Attention<br/>Ending Node]
        TRI_ATT_END --> PAIR_TRANS[Pair Transition<br/>FFN]
        PAIR_TRANS --> SINGLE_ATT[Attention with<br/>Pair Bias]
        SINGLE_ATT --> SINGLE_TRANS[Single Transition<br/>FFN]
    end

    SR[single_repr] --> BLOCK
    PR[pair_repr] --> BLOCK
    BLOCK --> SR_OUT[refined single_repr]
    BLOCK --> PR_OUT[refined pair_repr]
```

### Triangular Updates

```mermaid
flowchart LR
    subgraph TRI["Triangle Geometry"]
        A((i)) -->|"z_ij"| B((j))
        A -->|"z_ik"| C((k))
        C -->|"z_kj"| B
    end
```

- **Outgoing**: i → k → j path üzerinden z_ij güncelleme
- **Incoming**: k → i, k → j path üzerinden z_ij güncelleme
- Üçgen geometri constraint'i yapısal tutarlılık sağlar

## Related
- [[msa-module]] - Önceki aşama
- [[diffusion-module]] - Sonraki aşama
- [[../architecture/02-model-architecture]] - Model overview

#openfold3 #module #pairformer #transformer
