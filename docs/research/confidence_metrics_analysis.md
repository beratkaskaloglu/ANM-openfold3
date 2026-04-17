# Confidence Metrikleri Analizi ve Alternatifler

*Tarih: 2026-04-17*

## Bağlam

Mode-Drive pipeline'ında autostop stratejisi kullanılırken şu sorunlar gözlemlendi:
- **pLDDT her zaman yüksek** (~88-91), yapısal kalite ne olursa olsun — **ayırt edici değil**
- **pTM** alpha=1.0'da ~0.45-0.53, alpha=0.25'te ~0.65-0.83 — **kısmen ayırt edici** ama iteratif ayarda güvenilirliği azalıyor
- **L9 forced-accept** kötü yapıyı kabul edince cascade failure başlıyor
- Step 1-2 başarılı (L4 alpha=0.25 geçiyor), step 3+ tamamen çöküyor

---

## 1. AF3/OF3 Confidence Çıktıları

### Mevcut kullandıklarımız

| Metrik | Skala | Nasıl hesaplanıyor | Sorunlar |
|--------|-------|---------------------|----------|
| **pLDDT** | 0-100 | 50-bin softmax → expected value, lokal mesafe tutarlılığı | İteratif ayarda overconfident, 88-91 arası sıkışıyor |
| **pTM** | 0-1 | PAE logit'lerinden TM-score ağırlıklandırması: `1/(1+(d/d0)^2)` | Göreceli olarak daha ayırt edici ama kısa zincirler için çok sıkı |
| **ranking** | 0-1 | `0.8*pTM + 0.2*(pLDDT/100)` | pLDDT baskın olduğu için yanlış sonuç seçiyor (düzeltildi) |

### OF3'ün ürettiği ama kullanmadıklarımız

| Metrik | Skala | Açıklama | Potansiyel |
|--------|-------|----------|------------|
| **PAE** | [N×N] matris | Token i frame'inde token j'nin pozisyon hatası tahmini | Domain-domain ilişkisi, global fold doğruluğu — **çok değerli** |
| **PDE** | [N×N] matris | Mutlak mesafe hatası tahmini | PAE'ye tamamlayıcı |
| **ipTM** | 0-1 | Inter-chain pTM (multi-chain için) | Tek zincirde NaN, ama multi-chain'de kritik |
| **has_clash** | bool | >100 clash çifti veya >%50 atom <1.1Å | İkili — kullanışlı ama kaba |
| **contact_probs** | [N×N] | Distogram'dan contact olasılıkları | **Contact map tutarlılığı** için doğrudan kullanılabilir |

---

## 2. pLDDT Neden İteratif Ayarda Çalışmıyor?

### Kök Neden
Confidence head, eğitim dağılımındaki yapılara karşı LDDT tahmini yapıyor. Pipeline kendi ürettiğini geri beslediğinde, model "gerçek bir yapı gibi görünen" bir girdi görüyor. Confidence head'in bu yapının model tarafından üretildiğini tespit etme mekanizması yok — lokal mesafe tutarlılığını değerlendiriyor, ki global olarak yanlış ama iyi paketlenmiş bir yapı bile bunu sağlayabiliyor.

### Kanıtlar

1. **Thacker (2026)**: Fold-switching proteinlerde **%33.6 FalseVerify oranı** — her 3 yüksek-confidence tahmininden 1'i yanlış fold'a commit olmuş ama yüksek pLDDT gösteriyor. Tam sekonder yapı değişikliğinde oran %80-97'ye çıkıyor.

2. **Oda et al. (2024, AFM-Refine-G)**: İteratif refinement'ta 20 tahmin yüksek confidence (ipTM×0.8+pTM×0.2 > 0.8) göstermiş ama düşük doğruluk (DockQ < 0.23, "Yanlış" sınıfı). Sonuç: "Girdi yapısı model'in iç skor fonksiyonuyla eşleştiğinde yanıltıcı yüksek confidence üretebilir."

3. **Bryant & Noe (2024)**: Alternatif konformasyonlar için pLDDT ile yapısal doğruluk arasında Pearson R = sadece 0.52.

4. **Bizim gözlemimiz**: RMSD_init=27Å (yapı tamamen patlak) olan sonuçta pLDDT=89.4 — **hiçbir uyarı yok**.

### Sonuç
> **pLDDT, iteratif pipeline'da tek başına güvenilir bir stopping kriteri değildir.** Lokal kaliteyi ölçer ama global fold doğruluğunu yakalamaz. Sadece ek metriklerle birlikte kullanılmalı.

---

## 3. Alternatif Confidence Metrikleri

### 3.1 PAE Tabanlı Metrikler (YÜKSEK ÖNCELİK)

OF3 zaten PAE matrisi üretiyor. Bunu çıkarıp kullanabiliriz:

```python
# Mean PAE — global fold kalitesi
mean_pae = pae_matrix.mean()

# Contact-weighted PAE — yakın residue'ler arasındaki hata
contact_mask = (distance_matrix < 8.0)
contact_pae = pae_matrix[contact_mask].mean()

# Domain-domain PAE — farklı domain'ler arası ilişki
```

**Avantaj:** pTM zaten PAE'den türetiliyor ama bilgi kaybıyla. Ham PAE matrisi çok daha zengin.
**Aksiyon:** `of3_diffusion.py`'de PAE çıktısını DiffusionResult'a ekle.

### 3.2 Sample Tutarlılığı (YÜKSEK ÖNCELİK)

K>1 diffusion sample çalıştırıp aralarındaki uyumu ölç:

```python
# Inter-sample RMSD (K sample arasında)
pairwise_rmsd = compute_pairwise_rmsd(all_ca)  # [K, K]
consensus_score = 1.0 / (1.0 + pairwise_rmsd.mean())

# Per-residue RMSF across samples — lokal belirsizlik haritası
sample_rmsf = compute_rmsf_across_samples(all_ca)  # [N]
```

**Avantaj:** Confidence head'den tamamen bağımsız — diffusion model'in sampling belirsizliğini doğrudan ölçer. Model kendi çıktısını yüksek skor verebilir ama farklı sample'lar farklı yapı üretiyorsa bu gerçek belirsizlik.

**Dezavantaj:** K>1 gerekli → compute cost artar. Ama K=3 bile çok bilgilendirici.

### 3.3 Contact Map Tutarlılığı (ORTA ÖNCELİK)

Pipeline ANM-derived contact map'i z_pseudo'ya çeviriyor. OF3'ün distogram çıktısından predicted contact map'i alıp karşılaştır:

```python
# Input contact map vs predicted contact map
input_contacts = coords_to_contact(displaced_ca)  # ANM-derived
pred_contacts = of3_contact_probs  # distogram'dan

consistency = (input_contacts * pred_contacts).sum() / input_contacts.sum()
```

**Avantaj:** Pipeline'ın kendi iç tutarlılığını ölçer. ANM deplasman'dan gelen contact bilgisi ile OF3'ün ürettiği yapının contact'ları uyuşmuyor ise sorun var.

### 3.4 Fiziksel Doğrulama Metrikleri (DÜŞÜK ÖNCELİK)

Target gerektirmeyen, yapısal istatistiklere dayalı metrikler:

| Metrik | Ne ölçer | Araç |
|--------|----------|------|
| **Clash score** | Sterik çakışma / 1000 atom | MolProbity |
| **Ramachandran outlier %** | Backbone açı kalitesi | MolProbity |
| **Radius of gyration oranı** | Rg_obs / Rg_expected (zincir uzunluğuna göre) | Basit hesaplama |
| **Bond geometrisi** | Bağ uzunluk/açı sapması | MolProbity |

**Rg kontrolü** hemen implement edilebilir:
```python
def rg_ratio(coords, N):
    """Rg_observed / Rg_expected for globular proteins."""
    rg_obs = np.sqrt(np.mean(np.sum((coords - coords.mean(0))**2, axis=1)))
    rg_exp = 2.2 * N**0.38  # Flory scaling for globular proteins
    return rg_obs / rg_exp
# ratio >> 1.0 → yapı açılmış (unfolded/exploded)
# ratio ~ 1.0 → normal globular yapı
# ratio << 1.0 → aşırı sıkışmış
```

### 3.5 Gelişmiş Metrikler (ARAŞTIRMA)

| Metrik | Kaynak | Açıklama |
|--------|--------|----------|
| **actifpTM** | Varga et al. 2024 | ipTM'in sadece yapılandırılmış bölgeye odaklanmış versiyonu |
| **Cross-cluster pLDDT varyansı** | Thacker 2026 | Tek pLDDT değil, farklı seed'ler arası pLDDT farkı |
| **ContrastQA** | CASP16 birincisi | Graph contrastive learning ile model kalite tahmini |
| **DeepUMQA-PA** | CASP15/16 | Protein-kompleks kalite tahmini (single-model, consensus'tan iyi) |
| **EQAFold** | NSF 2024 | EGNN tabanlı iyileştirilmiş pLDDT tahmini |

---

## 4. Pipeline'daki Cascade Failure Analizi

### Mevcut Davranış
```
Step 1: L4 alpha=0.25 → pTM=0.828 → PASS → coords güncellenir ✓
Step 2: L4 alpha=0.25 → pTM=0.692 → PASS → coords güncellenir ✓
Step 3: Tüm level'lar FAIL → L9 FORCE → rejected=True → coords GÜNCELLENMİYOR ✓
Step 4: Önceki (step 2) coords'tan devam → ama yine FAIL → L9 FORCE ...
Step 5-8: Aynı döngü, hep L9 FORCE
```

### Sorunlar

1. **rejected=True çalışıyor** — coords güncellenmez, önceki iyi yapıdan devam eder. Ama pipeline aynı IW-ENM MD'yi aynı parametrelerle tekrar çalıştırıyor → aynı sonucu alıyor → sonsuz döngü.

2. **L9 sonrası çeşitlendirme yok** — Rejected step'ten sonra yeni bir IW-ENM çalıştırılıyor ama farklı seed/velocity/parametrelerle değil. Aynı yapıdan aynı MD → aynı displaced coords → aynı z_pseudo → aynı OF3 çıktısı.

3. **pTM neden step 3'te düşüyor?** — Step 2'den sonra coords değişti (RMSD_init=1.71Å). Bu yeni yapıdan başlayan IW-ENM daha farklı bir bölgeye gidiyor. Ama sorun, bu önceki iyi yapının artık "keşfedilecek yeni bölge"sinin confidence cutoff'ları geçememesi.

### Çözüm Önerileri

#### Öneri A: Rejected sonrası parametre mutasyonu
```python
if step_result.rejected:
    # Bir sonraki step'te IW-ENM parametrelerini değiştir
    config.autostop_v_magnitude *= 0.7   # daha az hareket
    config.autostop_n_steps *= 0.5       # daha kısa MD
    config.z_mixing_alpha *= 0.8         # daha az blending
```

#### Öneri B: Step bazlı alpha decay (cosine)
```python
alpha_t = alpha_max * 0.5 * (1 + cos(pi * step / n_steps))
# İlk step'lerde agresif, sonra giderek konservatif
# L4'teki alpha_scale buna ek olarak uygulanır
```

#### Öneri C: Sample tutarlılığı ile L9 filtreleme
```python
# L9'da en yüksek ranking yerine en tutarlı sample'ı seç
# K=3 sample aras inter-RMSD düşükse → yapı güvenilir
# K=3 sample arası inter-RMSD yüksekse → model emin değil, daha konservatif ol
```

#### Öneri D: Max rejected step limiti
```python
max_consecutive_rejected: int = 3  # 3 ardışık rejected → pipeline dur
# Sonsuz L9 döngüsünü engeller
```

#### Öneri E: Rg kontrolü ile fiziksel doğrulama
```python
# L9 seçiminde ranking yanında Rg kontrolü de yap
rg = compute_rg_ratio(new_ca, N)
if rg > 2.0:  # yapı açılmış, fiziksel olarak anlamsız
    skip  # bu sonucu L9 candidate'i olarak bile alma
```

---

## 5. Önerilen Yeni Confidence Stratejisi

### Kısa Vadeli (hemen uygulanabilir)

1. **Rg ratio kontrolü** — L9 candidate'lerinde Rg > 2.0 olanları elele
2. **Max consecutive rejected limiti** (3-5 step)
3. **Rejected sonrası alpha decay** — her rejected step alpha'yı %20 azalt
4. **pTM cutoff'u step'e göre gevşet** — ilk 5 step'te cutoff 0.4, sonra 0.5

### Orta Vadeli (1-2 hafta)

5. **PAE çıktısını of3_diffusion.py'den al** — mean_pae'yi DiffusionResult'a ekle
6. **K=3 sample tutarlılığı** — inter-sample RMSD'yi confidence metriği olarak kullan
7. **Contact map tutarlılığı** — ANM-derived vs OF3-predicted contact uyumu

### Uzun Vadeli (araştırma)

8. **Per-residue alpha** — düşük pLDDT bölgelere daha fazla pertürbasyon
9. **Cosine alpha scheduling** — step bazlı otomatik alpha azaltma
10. **MolProbity entegrasyonu** — clash score + Ramachandran check

---

## 6. Confidence Cutoff Kalibrasyonu

Mevcut pipeline çıktısından gözlemler:

| Alpha | pTM aralığı | pLDDT aralığı | Ranking aralığı | Yapısal kalite |
|-------|-------------|---------------|-----------------|----------------|
| 1.0 | 0.40-0.53 | 88-91 | 0.50-0.60 | Kötü (RMSD>20Å) |
| 0.5 | 0.25-0.54 | 86-88 | 0.37-0.61 | Orta (RMSD 2-15Å) |
| 0.25 | 0.33-0.83 | 85-91 | 0.44-0.84 | İyi-Mükemmel (RMSD 1.5-3Å) |

**Gözlem:** pLDDT hiçbir alpha'da ayırt edici değil (hep 85-91). pTM alpha=0.25'te çok değişken (0.33-0.83) — bazı step'lerde iyi, bazılarında kötü.

**Önerilen cutoff seti:**
```python
# Mevcut (çok sıkı)
confidence_ptm_cutoff: 0.5
confidence_plddt_cutoff: 70.0
confidence_ranking_cutoff: 0.5

# Önerilen (step-adaptive)
# İlk 5 step: gevşek (keşif aşaması)
confidence_ptm_cutoff_warmup: 0.35
confidence_ranking_cutoff_warmup: 0.40
warmup_steps: 5

# Step 6+: sıkı (yakınsama aşaması)
confidence_ptm_cutoff: 0.50
confidence_ranking_cutoff: 0.50
```

---

## Referanslar

- Jumper et al. (2021) — AlphaFold2 pLDDT, pTM, PAE tanımları
- AF3 Supplementary (2024) — N_cycle=4, PAE/PDE, ranking score formülü
- Thacker (2026) bioRxiv:2026.02.19.706878 — %33.6 FalseVerify, fold-switching proteinler
- Oda et al. (2024) bioRxiv:2022.12.27.521991v3 — AFM-Refine-G iteratif refinement uyarısı
- Bryant & Noe (2024) PLoS Comput Biol — pLDDT-doğruluk korelasyonu R=0.52
- Varga et al. (2024) arXiv:2412.15970 — actifpTM
- Schafer & Porter (2025) — Alternatif kalite ölçütleri çağrısı
- Chen et al. (2010) Acta Cryst D66 — MolProbity
- HAL/Structure (2025) — pLDDT esneklik tahmini sınırlamaları
- CASP15/16 EMA sonuçları — ContrastQA, DeepUMQA-PA, GraphGPSM
