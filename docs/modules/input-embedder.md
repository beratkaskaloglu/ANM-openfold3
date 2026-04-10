# Input Embedder

## Kaynak
`openfold3/core/model/feature_embedders/input_embedders.py` (21.5 KB)

## Sınıf: InputEmbedderAllAtom

Raw input features'ları model representation'larına dönüştürür.

### Girdi
- Sequence features (amino acid, nucleotide types)
- Atom features (element types, charges)
- Residue features (position, chain info)
- Bond features (connectivity)

### Çıktı
- `single_repr` [B, N_tokens, d_single]
- `pair_repr` [B, N_tokens, N_tokens, d_pair]

### İşlem Akışı

```mermaid
flowchart TD
    SEQ[Sequence Features] --> TOK[Token Embedding]
    ATOM[Atom Features] --> ATOM_E[Atom Embedding]
    RES[Residue Features] --> RES_E[Residue Embedding]

    TOK --> S[single_repr]
    ATOM_E --> S
    RES_E --> S

    S --> OP[Outer Product]
    OP --> P[pair_repr]

    REL[Relative Position Encoding] --> P
```

## Related
- [[../architecture/02-model-architecture]] - Model genel bakış
- [[msa-module]] - Sonraki aşama

#openfold3 #module #embedding
