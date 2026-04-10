# Prediction Heads

## Kaynak
- `openfold3/core/model/heads/prediction_heads.py` (21.3 KB)
- `openfold3/core/model/heads/head_modules.py` (9.8 KB)

## Sınıf: AuxiliaryHeadsAllAtom

Yapı kalitesi ve güvenilirlik metriklerini hesaplar.

### Confidence Metrikleri

```mermaid
flowchart LR
    subgraph INPUTS
        SR[single_repr]
        PR[pair_repr]
    end

    subgraph HEADS
        SR --> H1[pLDDT Head]
        PR --> H2[PAE Head]
        PR --> H3[pDE Head]
        H1 & H2 --> H4[pTM Computation]
    end

    subgraph OUTPUTS
        H1 --> O1["pLDDT<br/>0-100 per residue<br/>B-factor olarak CIF'e yazılır"]
        H2 --> O2["PAE<br/>NxN matrix<br/>Domain-domain error"]
        H3 --> O3["pDE<br/>Distance error<br/>Inter-chain kalite"]
        H4 --> O4["pTM<br/>0-1 scalar<br/>Global kalite"]
    end

    style HEADS fill:#2d3436,stroke:#00b894,color:#fff
```

### pLDDT (predicted Local Distance Difference Test)

| Aralık | Yorum |
|--------|-------|
| > 90 | Çok yüksek güven - iyi yapılanmış bölge |
| 70-90 | Güvenilir - genel fold doğru |
| 50-70 | Düşük güven - esnek loop/disordered bölge olabilir |
| < 50 | Çok düşük - muhtemelen disordered |

**ANM ile ilişki**: Düşük pLDDT bölgeleri genelde yüksek ANM flexibility gösterir.

### PAE (Predicted Aligned Error)

NxN matris: residue i'nin residue j'ye göre tahmin hatası.
- Düşük PAE bloğu → aynı rigid domain
- Yüksek PAE → farklı domain veya esnek bağlantı

**ANM ile ilişki**: PAE block yapısı, ANM'den çıkan dynamic domain sınırlarıyla karşılaştırılabilir.

### pDE (Predicted Distance Error)
- Özellikle inter-chain mesafe tahmin hatası
- Ligand binding kalitesi için önemli

### pTM (predicted Template Modeling Score)
- Global yapı kalitesi (0-1)
- > 0.5: iyi fold tahmini
- > 0.8: yüksek kaliteli tahmin

## Output Writer

```mermaid
flowchart TD
    COORDS[Atom Coordinates] --> W[OF3OutputWriter]
    CONF[Confidence Scores] --> W

    W --> CIF["mmCIF File<br/>B-factor = pLDDT"]
    W --> JSON["confidence.json<br/>Aggregated metrics"]
    W --> NPZ["full_confidence.npz<br/>Full matrices"]
```

## Related
- [[diffusion-module]] - Önceki aşama
- [[../architecture/01-openfold3-inference-pipeline]] - Pipeline
- [[../architecture/03-data-flow]] - Data flow

#openfold3 #module #confidence #plddt #pae
