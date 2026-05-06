# Konfigürasyon Referansı

`ModeDriveConfig`, ANM Mode-Drive pipeline'ının tüm parametrelerini tek bir `dataclass` içinde toplayan merkezi konfigürasyon yapısıdır. `src/mode_drive_config.py` dosyasında tanımlanır. Her parametre makul bir varsayılan değere sahiptir; yalnızca değiştirmek istediğiniz parametreleri override etmeniz yeterlidir.

---

## ANM Parametreleri

Anisotropic Network Model (ANM) hesaplaması için temel fiziksel parametreler.

| Parametre | Tip | Varsayılan | Açıklama |
|-----------|-----|-----------|----------|
| `anm_cutoff` | `float` | `15.0` | ANM etkileşim mesafesi cutoff (Angstrom). Bu mesafe içindeki Ca atomları arasında yay bağlantısı kurulur. |
| `anm_gamma` | `float` | `1.0` | ANM yay sabiti (force constant). Tüm bağlantılar için uniform gamma. |
| `anm_tau` | `float` | `1.0` | ANM tau parametresi (mesafe ağırlıklandırma üssü). |
| `n_anm_modes` | `int` | `20` | Hesaplanacak ANM normal mod sayısı (ilk N en düşük frekanslı mod). |

---

## Contact Map Parametreleri

Displaced koordinatlardan contact map üretimi için parametreler.

| Parametre | Tip | Varsayılan | Açıklama |
|-----------|-----|-----------|----------|
| `contact_r_cut` | `float` | `10.0` | Contact map mesafe eşiği (Angstrom). |
| `contact_tau` | `float` | `1.5` | Contact map yumuşatma parametresi. |

---

## Displacement (df) Parametreleri

ANM modlarından koordinat deplasmanı oluşturmak için ölçekleme faktörleri.

| Parametre | Tip | Varsayılan | Açıklama |
|-----------|-----|-----------|----------|
| `df` | `float` | `0.6` | Başlangıç global displacement faktörü (Angstrom). |
| `df_min` | `float` | `0.3` | Minimum izin verilen df değeri. |
| `df_max` | `float` | `3.0` | Maksimum izin verilen df değeri. |
| `df_escalation_factor` | `float` | `1.5` | Kombinasyonlar tükendiğinde df'yi çarpan escalation faktörü. |
| `df_scale` | `float` | `2.0` | Random combinator için df ölçek faktörü. |

---

## Selective Mixing (Per-Pair Adaptive Alpha)

z-latent blending sırasında her residue çiftine farklı alpha uygulamak için parametreler. `selective_mixing=True` olduğunda aktif.

| Parametre | Tip | Varsayılan | Açıklama |
|-----------|-----|-----------|----------|
| `selective_mixing` | `bool` | `False` | Selective mixing aktif mi? `False` = uniform alpha davranışı. |
| `z_mixing_alpha` | `float` | `0.3` | Global (uniform) z-mixing alpha değeri. Selective mixing kapalıyken tüm pair'lere bu uygulanır. |
| `z_direction` | `str` | `"plus"` | Delta-z yönü: `"plus"` (ekle) veya `"minus"` (çıkar). |
| `normalize_z` | `bool` | `True` | z-latent normalize edilsin mi? |
| `selective_w_connectivity` | `float` | `0.5` | Connectivity değişimi (Delta-C) ağırlığı. |
| `selective_w_distance` | `float` | `0.5` | Mesafe değişimi (Delta-D) ağırlığı. |
| `selective_change_cutoff` | `float` | `0.15` | Bu eşiğin altındaki change score'a sahip pair'ler `alpha_base` alır. |
| `selective_alpha_base` | `float` | `0.05` | Değişmeyen (korunan) pair'ler için minimum alpha. |
| `selective_alpha_max` | `float` | `0.7` | Maksimum değişim gösteren pair'ler için alpha üst sınırı. |
| `selective_mapping` | `str` | `"sigmoid"` | Change score -> alpha dönüşüm fonksiyonu: `"linear"`, `"sigmoid"`, `"step"`. |
| `selective_distance_mode` | `str` | `"max"` | Mesafe değişimi hesaplama modu: `"max"` veya `"mean"`. |
| `selective_diagonal_band` | `int` | `2` | `|i-j| <= band` olan pair'lerde trunk korunur (alpha_base uygulanır). |

---

## Combination Strategy

Mod kombinasyonu seçim stratejisi ve ilgili parametreler.

| Parametre | Tip | Varsayılan | Açıklama |
|-----------|-----|-----------|----------|
| `combination_strategy` | `CombinationStrategy` | `"collectivity"` | Strateji: `"collectivity"`, `"grid"`, `"random"`, `"targeted"`, `"manual"`, `"autostop"`. |
| `n_combinations` | `int` | `20` | Üretilecek kombinasyon sayısı. |
| `max_combo_size` | `int` | `3` | Bir kombinasyondaki maksimum mod sayısı. |
| `manual_modes` | `tuple[int, ...]` | `()` | Manual strateji için mod indeksleri (ör. `(0, 1, 2)`). |
| `select_modes_range` | `tuple[int, int]` | `(1, 5)` | Random combinator için mod sayısı aralığı. |
| `grid_select_modes` | `int` | `3` | Grid combinator: seçilen mod sayısı. |
| `grid_df_range` | `tuple[float, float]` | `(-2.0, 2.0)` | Grid combinator: df aralığı. |
| `grid_df_steps` | `int` | `5` | Grid combinator: df adım sayısı. |
| `targeted_top_modes` | `int` | `5` | Targeted combinator: en yüksek collectivity'li mod sayısı. |

---

## Confidence Gating V1

OF3 çıktı kalitesini değerlendiren birincil gate mekanizması. Bir step'in kabul/ret kararını belirler.

| Parametre | Tip | Varsayılan | Açıklama |
|-----------|-----|-----------|----------|
| `confidence_ptm_cutoff` | `float` | `0.15` | Minimum pTM eşiği (safety net). OF3 pTM trunk MSA'ya bağlı olduğundan düşük tutulmuştur. |
| `confidence_plddt_cutoff` | `float` | `65.0` | Minimum ortalama pLDDT skoru. |
| `confidence_ranking_cutoff` | `float` | `0.45` | OF3 ranking score gate (`use_composite_gate=False` iken kullanılır). |
| `use_composite_gate` | `bool` | `True` | Data-driven composite gate aktif mi? |
| `composite_gate_threshold` | `float` | `0.55` | Composite gate kabul eşiği. |
| `gate_w_cr` | `float` | `0.45` | Contact reconstruction ağırlığı (en güçlü sinyal, r=+0.598). |
| `gate_w_plddt` | `float` | `0.30` | pLDDT ağırlığı (ikincil sinyal, r=+0.271). |
| `gate_w_rg` | `float` | `0.15` | Rg ratio ağırlığı (fiziksel tutarlılık). |
| `gate_w_ptm` | `float` | `0.10` | pTM ağırlığı (düşük — ters korelasyon gösterir). |

---

## Confidence Gating V2

Ek fiziksel ve istatistiksel kalite metrikleri. `None` = devre dışı.

| Parametre | Tip | Varsayılan | Açıklama |
|-----------|-----|-----------|----------|
| `confidence_mean_pae_cutoff` | `float \| None` | `None` | Maksimum ortalama PAE (düşük = iyi). `None` = kontrol edilmez. |
| `confidence_consensus_cutoff` | `float \| None` | `None` | Minimum consensus score (K>1 diffusion samples arası uyum). |
| `confidence_contact_recon_cutoff` | `float \| None` | `None` | Minimum Pearson r: contact(displaced) vs contact(new_ca). |
| `confidence_contact_of3_cutoff` | `float \| None` | `None` | Minimum Pearson r: contact(displaced) vs OF3 distogram. |
| `confidence_rg_max` | `float` | `2.5` | Maksimum Rg oranı. >2.5 = yapı patlamış (unfolded). |
| `confidence_rg_min` | `float` | `0.3` | Minimum Rg oranı. <0.3 = aşırı sıkışmış (collapsed). |
| `confidence_clash_reject` | `bool` | `True` | OF3 clash algılaması pozitifse step'i reddet. |
| `confidence_rmsd_init_max` | `float` | `10.0` | RMSD-to-initial hard cutoff (Angstrom). >10A = yapı kurtarılamaz. |

---

## Warmup

Pipeline'ın ilk adımlarında confidence cutoff'ları gevşetir.

| Parametre | Tip | Varsayılan | Açıklama |
|-----------|-----|-----------|----------|
| `confidence_warmup_steps` | `int` | `0` | Warmup step sayısı. `0` = devre dışı. |
| `confidence_warmup_ptm_cutoff` | `float` | `0.25` | Warmup sırasında kullanılan pTM cutoff. |
| `confidence_warmup_ranking_cutoff` | `float` | `0.35` | Warmup sırasında kullanılan ranking cutoff. |

---

## Stall Prevention & Rollback

Pipeline'ın sıkışması veya drift'e girmesi durumunda koruma mekanizmaları.

| Parametre | Tip | Varsayılan | Açıklama |
|-----------|-----|-----------|----------|
| `max_consecutive_rejected` | `int` | `0` | Ardışık reddedilen step limiti. `0` = sınırsız. |
| `rejected_alpha_decay` | `float` | `1.0` | Her reddedilen step sonrası alpha çarpanı. `1.0` = decay yok. |
| `enable_best_rollback` | `bool` | `True` | Drift sonrası en iyi yapıya geri dönüş. |
| `best_rollback_tm_drop` | `float` | `0.40` | Mevcut TM, best-so-far'dan bu oranda düştüyse rollback tetiklenir. |

---

## Adaptive Early Stopping

Ardışık TM düşüşü algılandığında pipeline'ı erken durdurma.

| Parametre | Tip | Varsayılan | Açıklama |
|-----------|-----|-----------|----------|
| `enable_adaptive_stop` | `bool` | `True` | Adaptive early stopping aktif mi? |
| `adaptive_stop_window` | `int` | `3` | Bu sayıda ardışık accepted step'te TM düşüyorsa durdur. |

---

## Diffusion Parametreleri

OpenFold3 diffusion sampling kontrolü.

| Parametre | Tip | Varsayılan | Açıklama |
|-----------|-----|-----------|----------|
| `num_diffusion_samples` | `int` | `1` | Diffusion çağrısı başına üretilen sample sayısı (K). |

---

## Pipeline Kontrolü

Genel pipeline akış parametreleri.

| Parametre | Tip | Varsayılan | Açıklama |
|-----------|-----|-----------|----------|
| `n_steps` | `int` | `5` | Sabit iterasyon sayısı (early stop olmadan). |

---

## Adaptive Fallback

Confidence gate'i geçemeyen step'ler için kademeli fallback stratejisi.

| Parametre | Tip | Varsayılan | Açıklama |
|-----------|-----|-----------|----------|
| `enable_confidence_fallback` | `bool` | `False` | Confidence-guided fallback aktif mi? |
| `fallback_combo_tries` | `int` | `3` | Level 1: Collectivity sırasıyla sonraki N combo'yu dene. |
| `fallback_df_factor` | `float` | `0.5` | Level 2: `df *= factor` ile küçült. |
| `fallback_max_combo_size` | `int` | `1` | Level 3: Tek mod'a indir. |
| `fallback_alpha_factor` | `float` | `0.5` | Level 4: `alpha *= factor` ile küçült. |
| `fallback_extended_enabled` | `bool` | `True` | Level 5: Agresif grid search aktif mi? |
| `fallback_extended_combo_count` | `int` | `10` | Level 5: Top N combo. |
| `fallback_extended_df_scales` | `tuple[float, ...]` | `(0.5, 0.25)` | Level 5: df çarpanları. |
| `fallback_extended_alpha_scales` | `tuple[float, ...]` | `(0.5, 0.25)` | Level 5: alpha çarpanları. |

---

## Autostop Strategy

IW-ENM MD simülasyonu ile ANM mode-combo displacement'ın yerini alan strateji. `combination_strategy="autostop"` olduğunda aktif.

### Fiziksel Parametreler

| Parametre | Tip | Varsayılan | Açıklama |
|-----------|-----|-----------|----------|
| `autostop_chain_id` | `str` | `"A"` | Hedef zincir ID'si. |
| `autostop_R_bb` | `float` | `11.0` | Backbone etkileşim cutoff. |
| `autostop_R_sc` | `float` | `2.0` | Side-chain etkileşim cutoff. |
| `autostop_K_0` | `float` | `0.8` | Yay sabiti. |
| `autostop_d_0` | `float` | `3.8` | Referans denge mesafesi. |
| `autostop_n_ref` | `float` | `10.0` | Referans komşu sayısı. |

### Entegrasyon Parametreleri

| Parametre | Tip | Varsayılan | Açıklama |
|-----------|-----|-----------|----------|
| `autostop_dt` | `float` | `0.01` | Zaman adımı. |
| `autostop_mass` | `float` | `1.0` | Parçacık kütlesi. |
| `autostop_damping` | `float` | `0.0` | Sönümleme katsayısı. |
| `autostop_v_mode` | `str` | `"breathing"` | Başlangıç hız modu. |
| `autostop_v_magnitude` | `float` | `1.0` | Başlangıç hız büyüklüğü. |

### Çalıştırma Kontrolü

| Parametre | Tip | Varsayılan | Açıklama |
|-----------|-----|-----------|----------|
| `autostop_n_steps` | `int` | `5000` | MD simülasyon adım sayısı. |
| `autostop_save_every` | `int` | `10` | Her N adımda snapshot kaydet. |
| `autostop_back_off` | `int` | `2` | Turn-point'ten kaç save geri git. |
| `autostop_back_off_fraction` | `float \| None` | `None` | Ayarlanırsa `back_off = int(tk * fraction)`. Sabit back_off'u override eder. |
| `autostop_crash_threshold_distance` | `float` | `0.5` | Crash algılama mesafe eşiği. |
| `autostop_verbose` | `bool` | `True` | Detaylı log çıktısı. |

### Monitor (Early-Stop)

| Parametre | Tip | Varsayılan | Açıklama |
|-----------|-----|-----------|----------|
| `autostop_smooth_w` | `int` | `11` | Smoothing pencere boyutu. |
| `autostop_warmup_frac` | `float` | `0.40` | Warmup fraksiyonu (bu orana kadar early-stop kontrolü yapılmaz). |
| `autostop_patience` | `int` | `3` | Ardışık iyileşme olmayan kontrol sayısı. |
| `autostop_eps_E_rel` | `float` | `0.0002` | Enerji relatif epsilon (convergence kriteri). |
| `autostop_eps_N_rel` | `float` | `0.0005` | Native contacts relatif epsilon. |
| `autostop_crash_window_saves` | `int` | `20` | Crash algılama penceresi (save sayısı). |
| `autostop_crash_threshold` | `int` | `5` | Crash olarak sayılacak minimum olay. |
| `autostop_min_saves_before_check` | `int` | `15` | Early-stop kontrolü başlamadan önce minimum save sayısı. |

### Autostop Fallback Ladder (L0-L9)

Her level, BASELINE konfigürasyona göre bağımsız bir mutasyon uygular (kümülatif değil).

| Parametre | Tip | Varsayılan | Açıklama |
|-----------|-----|-----------|----------|
| `autostop_fallback_levels` | `tuple[int, ...]` | `(0, 1, 4, 9)` | Aktif fallback seviyeleri. L9 = forced-accept safety net. |
| `autostop_fallback_v_scales` | `tuple[float, ...]` | `(1.0, 0.5, 0.25, 0.1)` | Hız magnitude çarpanları. |
| `autostop_fallback_back_off_adds` | `tuple[int, ...]` | `(0, 2, 4, 8)` | back_off ekleme değerleri. |
| `autostop_fallback_pick_fractions` | `tuple[float, ...]` | `(1.0, 0.5, 0.25, 0.125)` | Pick point fraksiyonları: `pk = tk * frac`. |
| `autostop_fallback_eps_E_scales` | `tuple[float, ...]` | `(1.0, 2.0, 0.5, 0.25)` | eps_E çarpanları. |
| `autostop_fallback_eps_N_scales` | `tuple[float, ...]` | `(1.0, 2.0, 0.5, 0.25)` | eps_N çarpanları. |
| `autostop_fallback_patience_deltas` | `tuple[int, ...]` | `(0, -1, 1, 2)` | Patience delta değerleri. |
| `autostop_fallback_smooth_w_deltas` | `tuple[int, ...]` | `(0, 4, -2, 8)` | Smooth window delta değerleri. |
| `autostop_fallback_warmup_frac_scales` | `tuple[float, ...]` | `(1.0, 1.5, 0.5)` | Warmup fraction çarpanları. |
| `autostop_fallback_crash_window_scales` | `tuple[float, ...]` | `(1.0, 2.0, 0.5)` | Crash window çarpanları. |
| `autostop_fallback_crash_threshold_adds` | `tuple[int, ...]` | `(0, 2, -2)` | Crash threshold delta değerleri. |
| `autostop_fallback_alpha_scales` | `tuple[float, ...]` | `(1.0, 0.5, 0.35, 0.25, 0.15, 0.07)` | L4/L7 z_mixing_alpha çarpanları. |
| `autostop_fallback_grid_cap` | `int` | `8` | L7/L8 extended grid maksimum hücre sayısı. |

---

## Notlar

- `CombinationStrategy` tipi bir `Literal` union'dır: `"collectivity" | "grid" | "random" | "targeted" | "manual" | "autostop"`.
- Confidence gate ağırlıkları (`gate_w_*`) 708-step korelasyon analizinden türetilmiştir.
- OF3 pTM, trunk MSA'ya bağlıdır ve yapı değiştikçe sistematik olarak düşer; bu nedenle gate'de düşük ağırlıkla kullanılır.
- `use_composite_gate=True` (varsayılan) olduğunda `confidence_ranking_cutoff` devre dışı kalır; yerine composite score kullanılır.
