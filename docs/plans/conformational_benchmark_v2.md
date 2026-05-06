# Conformational Benchmark V2 — Multimer Destekli

**Tarih:** 2026-05-06
**Kaynak:** `data/conformational_benchmark.xlsx` (47 protein çifti)
**Referans:** `notebooks/benchmark_open_closed.ipynb` (mevcut monomer benchmark)

---

## 1. Genel Bakış

Mevcut benchmark notebook'u 9 monomer proteini test ediyor. Yeni benchmark:
- **47 protein çifti** (xlsx'ten)
- **Monomer + Multimer desteği** (Homodimer, Heterodimer, Trimer, Tetramer, Heptamer)
- **Multi-chain PDB parsing** (chain "A,B" formatı)
- **Membran proteinleri** (transporter konformasyonları: inward/outward)
- **3 mod:** State1→State2, State2→State1, UniProt Seq

---

## 2. Veri Seti Analizi

### Oligomerik Dağılım

| Oligomer | Sayı | Multi-chain? |
|----------|------|-------------|
| Monomer | 28 | Hayır |
| Homodimer | 8 | 2'si A,B |
| Heterodimer | 3 | Hepsi A,B |
| Dimer | 2 | Hayır |
| Trimer | 1 | Hayır |
| Tetramer | 2 | Hayır |
| Heptamer | 1 | Hayır |

### Multi-Chain Girdiler (5 adet)

| # | Protein | Chain_S1 | Chain_S2 | Oligomer |
|---|---------|----------|----------|----------|
| 26 | ABC transporter TM287/TM288 | A,B | A,B | Heterodimer |
| 32 | WlaB (ABC transporter) | A,B | A,B | Heterodimer |
| 34 | ABC transporter PF0708 | A,B | A,B | Homodimer |
| 37 | ABC transporter TT_C0976 | A,B | A,B | Homodimer |
| 42 | TmrAB (ABC transporter) | A,B | A,B | Heterodimer |

### Kategori Dağılımı

- ABC Transporter: 7
- Binding protein: 5
- MFS Transporter: 3
- P-type ATPase: 3
- Hydrolase: 3
- Kinase: 2
- SLC Transporter: 2
- Transferase: 2
- Diğer: 20 (birer adet)

### Özel Durumlar

| # | Durum | Handling |
|---|-------|---------|
| 31 | MFSD2A: State1 chain=X, State2 chain=A | Her state farklı chain |
| 36 | AAC(3): State2 chain=B | Farklı chain closed form |
| 38-39 | P5A-ATPase: İki State1, ortak State2 | İki ayrı run |
| 42 | TmrAB: İki farklı UniProt (Q72J62/Q72J61) | Heterodimer, iki zincir farklı protein |
| 45 | GPCR: PDB belirtilmemiş | Skip (TBD) |
| 46-47 | SERCA, KRAS: Suggested | Dahil et |

---

## 3. Mimari Kararlar

### 3.1 Multi-Chain Handling

**Monomer (chain = "A"):**
- Mevcut `fetch_ca(pdb_id, "A")` — değişiklik yok

**Multi-chain (chain = "A,B"):**
- Her zinciri ayrı parse et
- CA koordinatlarını birleştir: `cat([ca_A, ca_B], dim=0)`
- Sekansları birleştir: `seq_A + seq_B`
- Zincir sınırını kaydet (downstream alignment için)

```python
def fetch_ca_multimer(pdb_id: str, chain_ids: str) -> tuple[Tensor, str, list[int]]:
    """Multi-chain CA extraction.

    Args:
        chain_ids: "A" or "A,B" format

    Returns:
        coords: [N_total, 3]
        sequence: concatenated sequence
        chain_breaks: [0, N_A, N_A+N_B] — zincir başlangıç indeksleri
    """
```

### 3.2 OF3 Multimer Query

OF3 doğal olarak multimer destekler. Query JSON'u multi-chain olarak oluşturmak yeterli:

```python
# Monomer
{"chains": [{"molecule_type": "protein", "chain_ids": ["A"], "sequence": seq_A}]}

# Homodimer (aynı sekans)
{"chains": [
    {"molecule_type": "protein", "chain_ids": ["A"], "sequence": seq_A},
    {"molecule_type": "protein", "chain_ids": ["B"], "sequence": seq_B}
]}

# Heterodimer (farklı sekans)
{"chains": [
    {"molecule_type": "protein", "chain_ids": ["A"], "sequence": seq_A},
    {"molecule_type": "protein", "chain_ids": ["B"], "sequence": seq_B}
]}
```

**Önemli:** OF3 multimer prediction'da `z_pair` boyutu `[N_A+N_B, N_A+N_B, 128]` olur. ANM Hessian'ı da birleşik koordinatlar üzerinden hesaplanır — zincirler arası etkileşimler doğal olarak dahil.

### 3.3 ANM Multi-Chain

ANM Hessian zaten tüm CA atomları arasındaki mesafeleri kullanır. İki zincir birleştirildiğinde:
- Zincir içi + zincirler arası spring'ler otomatik oluşur (cutoff=10Å)
- Düşük frekanslı modlar zincirler arası hinge motion'ı yakalar
- Ek bir değişiklik gerekmez

### 3.4 Benchmark Tablosu Format

```python
BENCHMARK_PAIRS = [
    {
        "idx": 1,
        "name": "Alpha-2-macroglobulin receptor-associated protein",
        "organism": "Bos taurus",
        "uniprot": "A0A075Q0W3",
        "pdb_state1": "6MKG",
        "chain_s1": "A",
        "state1_label": "Open",
        "pdb_state2": "6MKJ",
        "chain_s2": "A",
        "state2_label": "Closed",
        "category": "Enzyme/Binding",
        "oligomeric": "Monomer",
        "source": "OC23",
    },
    # ...
]
```

---

## 4. Notebook Yapısı

### Cell 0: Başlık + Açıklama
Conformational Benchmark V2 — 47 protein, monomer + multimer

### Cell 1: Environment Setup
Colab ortamı, repo clone, OF3 setup (mevcut ile aynı)

### Cell 2: Benchmark Tablosu
xlsx'ten okunan veya hardcode edilen 47 protein çifti.
`openpyxl` ile xlsx'ten otomatik parse VEYA dict listesi.

```python
# Skip edilecekler
SKIP = {45}  # GPCR — PDB belirtilmemiş
```

### Cell 3: Pipeline Konfigürasyonu
Mevcut config ile aynı — `ModeDriveConfig` parametreleri

### Cell 4: Helper Fonksiyonlar

#### `fetch_ca_multimer(pdb_id, chain_ids)`
- `chain_ids` virgülle ayrılmış olabilir ("A" veya "A,B")
- Her chain'i ayrı parse et
- Birleştir ve chain_breaks döndür

#### `build_multimer_query(chains_data)`
- OF3 multi-chain query JSON oluştur
- Her chain için ayrı entry

#### `run_single_direction()` — Güncellemeler
- `fetch_ca` → `fetch_ca_multimer`
- Multi-chain OF3 query desteği
- Size mismatch'te chain-aware alignment
- `comparison_indices` multi-chain'de doğru çalışmalı

#### `fetch_uniprot_sequence(uniprot_id)` — Mevcut
- Heterodimer durumunda iki farklı UniProt ID'si olabilir (42: Q72J62/Q72J61)
- Bu durumda ikisini de çek ve birleştir

### Cell 5: Converter + OF3 Model Yükleme
Mevcut ile aynı — tek seferlik

### Cell 6: Ana Benchmark Döngüsü

```python
for pair in BENCHMARK_PAIRS:
    if pair["idx"] in SKIP:
        continue

    is_multimer = "," in pair["chain_s1"]

    # [A] State1 -> State2
    # [B] State2 -> State1
    # [C] UniProt Seq (sadece monomer + homodimer için güvenilir)
```

**Multimer'da Mode C (UniProt):**
- Homodimer: Aynı UniProt sekansı × 2 chain
- Heterodimer: İki farklı UniProt sekansı (varsa)
- Trimer+: Monomer UniProt × N chain
- Heterodimer'da iki farklı UniProt bilinmiyorsa Mode C skip

### Cell 7: Summary Table + CSV
Mevcut format + ek sütunlar:
- `Oligomeric`: Monomer/Homodimer/etc.
- `Category`: Enzyme/Transporter/etc.
- `Source`: OC23/TP16/Additional

### Cell 8-13: Visualizations
- TM trajectory per protein (gruplandırılmış)
- Delta TM bar chart (monomer vs multimer ayrımı)
- Scatter: Init TM vs Best TM
- Kategori bazlı ortalama delta TM
- Oligomer bazlı başarı oranı
- Monomer vs Multimer karşılaştırma

---

## 5. Implementasyon Sırası

### Adım 1: `fetch_ca_multimer()` yaz
- Multi-chain parsing
- Chain break tracking
- Test: `fetch_ca_multimer("3FTO", "A")` (monomer) ve `fetch_ca_multimer("3QF4", "A,B")` (heterodimer)

### Adım 2: OF3 multi-chain query builder
- `_prepare_sequence_impl` güncelle: multi-chain query JSON
- Test: Homodimer için trunk çalışır mı?

### Adım 3: Benchmark tablosunu xlsx'ten parse et
- `openpyxl` (Colab'da mevcut) veya hardcode dict

### Adım 4: `run_single_direction` güncelle
- Multimer-aware fetch, query, comparison
- Chain-aware size mismatch handling

### Adım 5: Ana döngü + visualization
- Skip logic (TBD, eksik PDB)
- Kategori/oligomer bazlı analiz

---

## 6. Riskler ve Dikkat Edilecekler

| Risk | Etki | Çözüm |
|------|------|-------|
| OF3 multimer trunk çok yavaş | N²×128 tensör büyük | Küçük multimer'larla başla |
| Membran proteinleri düşük TM | Lipid bilayer ortamı eksik | Beklenen — ayrı raporla |
| ColabFold MSA mapping corruption | Batch invalid | MSA cache temizleme (zaten fix edildi) |
| Heterodimer iki farklı UniProt | Mode C hangi sekans? | İki sekansı birleştir VEYA Mode C skip |
| Heptamer (N×7) çok büyük | GPU OOM | Skip veya tek chain test et |
| NMR yapıları (18: eRF1) | Çoklu model → hangi? | model 0 (varsayılan) |
| Farklı chain ID'leri (31: X vs A) | fetch_ca chain fallback | Her state kendi chain_id'si |

---

## 7. OF3 `_prepare_sequence_impl` Değişiklikleri

Mevcut fonksiyon tek-chain query oluşturuyor. Multi-chain desteği için:

```python
def _prepare_sequence_impl(..., chains_data=None):
    """
    chains_data: None (eski uyumluluk, tek chain)
    veya list[dict]:
        [{"chain_id": "A", "sequence": "MKLT..."},
         {"chain_id": "B", "sequence": "GAVL..."}]
    """
    if chains_data is None:
        # Mevcut tek-chain davranış
        _chains = [{"molecule_type": "protein", "chain_ids": [chain_id], "sequence": sequence}]
    else:
        _chains = [
            {"molecule_type": "protein", "chain_ids": [cd["chain_id"]], "sequence": cd["sequence"]}
            for cd in chains_data
        ]

    _query = {"queries": {query_name: {"chains": _chains}}}
```

---

## 8. Beklenen Çıktı

```
=== Conformational Benchmark V2 ===
47 protein çifti | 28 monomer + 19 multimer
3 mod: State1→State2, State2→State1, UniProt

[1/47] Alpha-2-macroglobulin receptor-associated protein (Monomer)
  [A] 6MKG(Open) -> 6MKJ(Closed)
      init_TM=0.XXXX best_TM=0.XXXX delta=+0.XXXX (XXXs)
  [B] 6MKJ(Closed) -> 6MKG(Open)
      ...
  [C] UniProt A0A075Q0W3
      ...

[26/47] ABC transporter TM287/TM288 (Heterodimer, chains A,B)
  [A] 3QF4(Inward) -> 6QV1(Outward-occluded)
      N_total=XXX (A:XXX + B:XXX), init_TM=0.XXXX
      ...
```

---

## Dosya Referansları

- [[../architecture/pipeline-deep-dive]] — Pipeline teknik detayları
- [[../architecture/configuration-reference]] — Config parametreleri
- [[benchmark_fix_plan]] — Mevcut benchmark fix'leri (MSA cache, token_mask, size mismatch)
- Veri: `data/conformational_benchmark.xlsx`
