# ANM-OpenFold3 Araştırma Yolculuğu

**Tarih:** 2026-05-06
**Durum:** Aktif geliştirme

---

## 1. Proje Motivasyonu

OpenFold3 (OF3), AlphaFold3'un açık kaynak implementasyonudur ve protein yapısı tahmininde son derece başarılıdır. Ancak **tek bir statik yapı** üretir — protein dinamiğini yakalayamaz. Proteinler doğada birden fazla konformasyonda bulunur (açık/kapalı, aktif/inaktif), ve bu konformasyonel çeşitlilik biyolojik fonksiyon için kritiktir.

**Anisotropic Network Model (ANM)**, proteini elastik ağ modeli olarak ele alarak düşük frekanslı titreşim modlarını (hinge bending, domain motion, shear) hesaplar. Bu modlar, büyük ölçekli konformasyonel değişimlerin yönlerini fiziksel olarak anlamlı bir şekilde temsil eder.

**Temel fikir:** ANM'nin ürettiği fiziksel deplasman bilgisini, OF3'ün pair representation (`z_ij`) formatına dönüştürüp diffusion modülüne geri besleyerek, yapıyı **bilinen bir hedef olmadan** konformasyonel uzayda keşfe yönlendirmek. Bu şekilde:

- OF3'ün yapısal bilgisi korunur (trunk inference bir kez hesaplanır)
- ANM fiziksel olarak anlamlı hareket yönleri sağlar
- İteratif döngü ile giderek daha farklı konformasyonlar üretilebilir

---

## 2. Kronolojik Gelişim

### Temel Araştırma: IW-ENM ve Encoder-Decoder

Projenin ilk aşamasında iki temel bileşen geliştirildi:

**IW-ENM (Importance-Weighted Elastic Network Model):** Klasik ANM'nin ötesinde, ağırlıklı spring sabitleriyle Velocity Verlet integratörü kullanan MD simülasyonu. Konformasyonel geçiş noktalarını (turnpoint) tespit ederek erken durdurma kriteri sağlar. Parametre optimizasyonu için MLX tabanlı surrogate MLP yaklaşımı geliştirildi.

**PairContactConverter (Encoder-Decoder):** OF3 trunk'ın ürettiği `z_pair [N,N,128]` ile sigmoid contact map `[N,N]` arasında çift yönlü dönüşüm yapan mimari. Encoder: Linear bottleneck + dot product + sigmoid. Decoder: logit + learned MLP inverse. Üç bileşenli loss (Focal + GNM fizik-bilgili + Reconstruction) ile eğitildi. Test sonuçları: adj_acc=0.9768, bf_r=0.8143.

Detaylar: [[iw-enm-and-converter]]

---

### V1: Basit Uniform Mixing (İlk Tasarım)

**Mimari:** Koordinatlar → ANM modları → deplasman → contact map → inverse head (z_pseudo) → sabit alpha ile blending → OF3 diffusion

```
z_mod = z_trunk + alpha * (z_pseudo - z_trunk)   # tüm pair'lere aynı alpha
```

**Strateji:** Collectivity-tabanlı mod kombinasyonları, df eskalasyonu, RMSD artışını hedefleyen seçim.

**Sorunlar:**
- Kalite kontrolü yok — diffusion çıktısı fiziksel olarak anlamsız olabilir
- Tek diffusion sample — stokastik model farklı çıktılar verebilir
- Kör eskalasyon — yapı bozulduğunda pipeline bunu anlayamaz

Bkz. [[../architecture/09-anm-mode-drive]] ve [[../architecture/10-iterative-refinement]]

---

### V2: Confidence-Guided Adaptive Pipeline

**Tarih:** 2026-04-17 — 2026-04-19
**Motivasyon:** V1'de yapısal drift kontrolsüz; pLDDT iteratif ayarda ayırt edici değil (88-91 arası, RMSD=27A'da bile 89).

**Yenilikler:**
1. **Multi-sample diffusion** (K=5) — en iyi sample ranking score ile seçilir
2. **Confidence metrikleri entegrasyonu** — pTM, pLDDT, PAE, Rg ratio, clash detection
3. **Adaptive fallback ladder** — confidence düşükse parametreleri kademeli düşür (df↓, alpha↓, combo↓)
4. **DiffusionResult genişletme** — PAE, contact_probs, has_clash, consensus_score
5. **Rg filtresi** — patlamış yapıları (Rg > 2.5) hard-reject
6. **Stall prevention** — max_consecutive_rejected + rejected_alpha_decay

**Kritik bulgular:**
- pLDDT iteratif pipeline'da **güvenilir değil** — lokal kaliteyi ölçer ama global fold doğruluğunu yakalayamaz (Thacker 2026: %33.6 FalseVerify oranı)
- pTM kısmen ayırt edici ama tek başına yeterli değil
- Rg ratio ve clash detection basit ama etkili fiziksel filtreler

Bkz. [[confidence_metrics_analysis]], [[../plans/confidence_v2_implementation]]

---

### V3: Composite Confidence + Optimized Search

**Tarih:** 2026-04-24
**Protein:** Adenylate kinase (1AKE → 4AKE, N=214)

**Yenilikler:**
1. **Composite confidence scoring** — ağırlıklı bileşik skor (w_ptm, w_plddt, w_pae, w_rg, w_contact)
2. **4 fazlı grid search:**
   - Phase 1: Weight grid (4 preset × 3 threshold)
   - Phase 2: Alpha schedule (6 strateji)
   - Phase 3: Internal pTM cutoff (4 seviye)
   - Phase 4: Final 30-step run

**Anahtar sonuçlar:**
| Parametre | En İyi Değer | Etki |
|-----------|-------------|------|
| Alpha | 0.3 (sabit, decay yok) | Küçük adım = OF3 daha iyi katlar |
| pTM cutoff | 0.50 | %70 kabul oranı, TM=0.83 |
| Threshold | 0.50 | 0.45 çok gevşek, 0.55 çok sıkı |

**V3 vs V2 karşılaştırması:**
- Accept rate: %20 → %70 (+50pp)
- best_TM: 0.775 → 0.833 (+0.058)
- best_RMSD: 3.35A → 2.62A (-0.73A)
- Drift onset: Step 2 → Step 11 (+9 adım)

**Kritik bulgu:** "Büyük adım atıp sonra küçültmek (decay) yerine, **baştan küçük adım atmak** çok daha etkili."

**RMSD_init kritik eşiği:** ~10A üzerinde OF3 yapıyı doğru katlayamıyor.

Bkz. [[v3_search_analysis]]

---

### Autostop Stratejisi (IW-ENM MD Alternatifi)

**Tarih:** 2026-04-17+

Mode-combo yerine **Interaction-Weighted ENM Molecular Dynamics** kullanan alternatif deplasman kaynağı:

- Kısa MD simülasyonu (velocity-Verlet, 5000 adım)
- Enerji reversal + spring count reversal ile early-stop
- Adaptif fallback merdiveni (L0-L9): cache-replay seviyeleri MD'yi tekrar çalıştırmaz
- Fiziksel olarak daha zengin (non-linear bükülmeler)

**Grid Search V2 bulguları (N=214):**
- `warmup_steps=0` — warmup gereksiz (>=5 felaket)
- `rg_max=2.5` — minimal etki ama güvenli
- `max_consecutive_rejected=8` — en iyi denge
- `rejected_alpha_decay=0.8` — alpha death spiral'ı kırıyor

Bkz. [[grid_search_v2_colab_analysis]], [[enm_early_stopping_criteria]], [[../architecture/09-anm-mode-drive]]

---

### V4: Selective Z-Mixing (Per-Pair Adaptive Alpha)

**Tarih:** 2026-04-25
**Motivasyon:** Uniform alpha, proteinlerdeki heterojen hareketi göz ardı eder — loop/hinge çok hareket ederken helix/core hiç hareket etmez.

**Yenilik:** Her (i,j) residue çifti için "ne kadar değişti?" sorusunu sorup buna göre pair-specific alpha uygula:

```
change_score[N,N] = f(ΔC_topolojik, ΔD_konformasyonel)
alpha_mask[N,N] = g(change_score, cutoff, base, max)
z_mod = z_trunk + alpha_mask * (z_pseudo - z_trunk)
```

**Üç adımlı hesaplama:**
1. **Change Score** — topolojik (ΔC) + konformasyonel (ΔD) değişim birleşimi
2. **Alpha Mask** — cutoff altı sıfırlanır, mapping (linear/sigmoid/step)
3. **Selective Blend** — per-pair alpha ile karıştırma

**Beklenen avantajlar:**
- Core pair'ler korunur (alpha=0) → OF3 stabil
- Loop pair'ler güncellenir (alpha≈0.8) → OF3 hareketi görür
- Daha yüksek alpha_max kullanılabilir (sadece değişen pair'lere)

Bkz. [[../plans/selective_mixing_v1]], [[../architecture/14-selective-mixing-pipeline]]

---

### V5: Resilient Pipeline + Grid Search

**Tarih:** 2026-05-05
**Durum:** Aktif — grid search devam ediyor

**Yenilikler:**
1. **Drive checkpointing** — Colab kesintilerinde sonuçlar korunur
2. **Phase-by-phase grid search** (28 run):
   - Phase A: Uniform vs Selective (2 run)
   - Phase B: change_cutoff sweep [0.05, 0.1, 0.15, 0.2, 0.3] (5 run)
   - Phase C: cutoff × alpha_base × alpha_max (12 run)
   - Phase D: mapping × weights (9 run)
3. **Drift korumaları:** best-so-far rollback, adaptive early stop, alpha decay
4. **Confidence cutoff güncellemesi:** pTM=0.30 (düşürüldü), ranking=0.45 (primary gate)

**Konfigürasyon parametreleri (V5):**
- `selective_mixing: True`
- `selective_change_cutoff: 0.1`
- `selective_alpha_base: 0.0`
- `selective_alpha_max: 1.0`
- `selective_mapping: "linear"`

Bkz. [[../plans/selective_mixing_v5_resilient]], [[../architecture/14-selective-mixing-pipeline]]

---

## 3. Anahtar Bulgular

### Grid Search Sonuçları

| Deney | Optimal Parametre | Bulgu |
|-------|-------------------|-------|
| Alpha schedule | alpha=0.3 sabit | Decay stratejileri gereksiz, baştan küçük adım en iyi |
| pTM cutoff | 0.50 | Pipeline'ın kendi confidence filtresi composite'den daha etkili |
| Warmup | 0 step | warmup>=5 felaket (Rg patlaması) |
| Stall detection | 8 ardışık reject | 8 adım sonra durma, gereksiz GPU'yu önler |
| Alpha decay | 0.8 per reject | Alpha death spiral'ı kırıyor |
| Composite threshold | 0.50 | 0.45 çok gevşek, 0.55 çok sıkı |

### Confidence Metrikleri Hiyerarşisi

| Metrik | Güvenilirlik | Kullanım |
|--------|-------------|----------|
| pLDDT | Düşük (iteratif ayarda) | Tek başına KULLANMA |
| pTM | Orta | Internal cutoff (0.50) |
| Ranking (0.8*pTM + 0.2*pLDDT) | Orta-Yüksek | Primary gate (0.45) |
| Rg ratio | Yüksek (fiziksel) | Hard reject (>2.5) |
| RMSD_init | Yüksek | Hard reject (>10A) |
| Composite score | Yapılandırılabilir | Threshold=0.50 |

### Kritik Eşikler

- **RMSD_init > 10A** → OF3 yapıyı doğru katlayamıyor (ortalama TM < 0.27)
- **Rg > 2.5** → yapı fiziksel olarak anlamsız (patlamış)
- **Alpha > 0.5** → çok agresif, yapı bozulur
- **df > 3.0** → ANM harmonik rejimi dışına çıkılır

---

## 4. Benchmark Sonuçları

### Adenylate Kinase Başarı Hikayesi

**Protein:** Adenylate kinase (N=214), 1AKE (açık) ↔ 4AKE (kapalı)
**Baseline RMSD:** 7.14A, TM=0.45

**En iyi sonuçlar:**
| Yön | TM-score | Açıklama |
|-----|----------|----------|
| Open → Closed | 0.57 → **0.93** | Neredeyse mükemmel hedef tahmini |
| Closed → Open | 0.57 → **0.74** | Anlamlı iyileşme |

ADK, pipeline'ın "poster child" proteinidir — hinge bending hareketi ANM'nin güçlü olduğu domain motion'dır.

### Benchmark Sorunları (Diğer Proteinler)

**Tarih:** 2026-05-06, ilk benchmark çalışması (9 protein × 2 yön = 18 run)

**Başarı oranı:** 2/18 (%11) — sadece ADK çalıştı.

| Sorun | Etkilenen Proteinler | Kök Neden |
|-------|---------------------|-----------|
| `token_mask` KeyError | Maltose-binding, Citrate synthase, Lactoferrin, LAO-binding | OF3 global state corruption (ColabFold MSA cache) |
| Size mismatch | Glutamine-binding (220/223), Src kinase (452/449), GGBP (305/309) | Farklı PDB yapılarında farklı sayıda çözülmüş residue |
| Chain not found | PKA-Calpha | Yanlış chain ID (A yerine E) |

**Planlanan düzeltmeler:**
1. Token_mask fallback + singleton model (en kritik, %80 bloğu kaldırır)
2. Sequence alignment + common core (size mismatch)
3. Chain ID auto-detection

Bkz. [[benchmark_issues_analysis]], [[benchmark_fix_plan]]

---

## 5. Mevcut Durum

### Pipeline Kapasitesi

- **Çalışan protein:** Adenylate kinase (N=214) — TM 0.45 → 0.93
- **GPU:** A100 (Colab) gerekli, ~50-84 dk/run (20-30 step)
- **Strateji:** Collectivity + autostop (IW-ENM MD) alternatifi
- **Confidence:** Composite scoring + hard rejects (Rg, clash, RMSD_init)
- **Mixing:** Selective (per-pair adaptive alpha) — grid search devam ediyor
- **Test coverage:** 360 test, %60 overall

### Bilinen Limitler

1. **Tek protein validasyonu** — sadece ADK'da kanıtlanmış, diğer proteinler benchmark bugları nedeniyle test edilemedi
2. **pLDDT güvenilmezliği** — iteratif ayarda ayırt edici değil
3. **Yapısal drift** — 10+ adım sonra kaçınılmaz (Rg artışı, TM düşüşü)
4. **Hesaplama maliyeti** — trunk 1 kez, ama her step'te diffusion + aux_heads (K sample)
5. **Hedef-bağımsız** — hedef yapı bilinmeden çalışır (avantaj ve limit)

---

## 6. Gelecek Yönelimler

### Kısa Vade (Aktif)

1. **Benchmark fix'leri** — token_mask, size mismatch, chain ID düzeltmeleri → 16-18/18 başarı hedefi
2. **V5 selective mixing grid search tamamlama** — optimal cutoff, alpha_base, alpha_max, mapping bulma
3. **Çoklu protein validasyonu** — benchmark düzeltildikten sonra 9 proteinde TM iyileşmesi ölçme

### Orta Vade

4. **Best-so-far rollback** — en iyi TM'li yapıyı kaydet, drift başladığında geri dön
5. **Adaptive stopping** — 3 ardışık TM düşüşünde pipeline'ı durdur, en iyi yapıyı döndür
6. **PAE-guided mode selection** — PAE matrisinden belirsiz bölgeleri öğren, oraya odaklan
7. **Model caching** — OF3 model'i tek kez yükle, proteinler arası paylaş

### Uzun Vade (Araştırma)

8. **Per-residue alpha (confidence-weighted)** — düşük pLDDT bölgelere daha fazla pertürbasyon
9. **Cosine alpha scheduling** — step bazlı otomatik alpha azaltma
10. **Multi-metric adaptive stopping** — spring survival + RMSD acceleration + energy slope
11. **Farkli protein boyutları** — N=100, N=500 ile parametre kalibrasyonu
12. **ContrastQA / DeepUMQA entegrasyonu** — pLDDT'ye alternatif model kalite tahmini

---

## Dosya İndeksi

### Araştırma
- [[iw-enm-and-converter]] — IW-ENM elastic MD + PairContactConverter encoder-decoder eğitimi
- [[alpha_scheduling_strategies]] — Alpha zamanlama literatür taraması
- [[enm_early_stopping_criteria]] — ENM early-stopping kriterleri
- [[confidence_metrics_analysis]] — Confidence metrikleri analizi
- [[grid_search_v2_colab_analysis]] — Grid search V2 Colab sonuçları
- [[v3_search_analysis]] — V3 composite confidence detaylı analiz
- [[benchmark_issues_analysis]] — Benchmark sorunları analizi
- [[benchmark_fix_plan]] — Benchmark düzeltme planı
- [[code_review_findings]] — Kod inceleme bulguları

### Mimari
- [[../architecture/08-anm-theory]] — ANM matematigi
- [[../architecture/09-anm-mode-drive]] — Mode-Drive pipeline tasarımı
- [[../architecture/10-iterative-refinement]] — İteratif refinement dinamikleri
- [[../architecture/13-confidence-guided-pipeline]] — Confidence sistemi mimarisi
- [[../architecture/14-selective-mixing-pipeline]] — Selective mixing tam mimari

### Planlar
- [[../plans/confidence_v2_implementation]] — Confidence V2 implementasyonu
- [[../plans/optimized_search_v3]] — V3 optimized search planı
- [[../plans/selective_mixing_v1]] — Selective mixing ilk tasarım
- [[../plans/selective_mixing_v5_resilient]] — V5 resilient notebook planı
