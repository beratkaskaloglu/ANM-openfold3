# Agent Task Implementation Plan

Bu dosya 4 paralel sub-agent icin gorev tanimlari icerir. Her agent bagimsiz calisabilir.

---

## Agent 1: Train Pipeline Test Suite

**Amac:** `src/train.py`, `src/model.py`, `src/data.py` icin kapsamli unit + integration testler.

**Mevcut Durum:**
- `tests/test_losses.py` (131 LOC) - loss fonksiyonlari test ediliyor
- `tests/test_contact_head.py` (93 LOC) - ContactProjectionHead test ediliyor
- `tests/test_kirchhoff.py` (80 LOC) - GNM Kirchhoff test ediliyor
- `tests/test_inverse.py` (60 LOC) - PairReprFromCoords test ediliyor
- `tests/test_ground_truth.py` (57 LOC) - ground truth contact test ediliyor
- **EKSIK:** train.py, model.py, data.py icin HICBIR test yok

**Olusturulacak Dosyalar:**

### `tests/test_train.py`
Test sinifi ve senaryolar:

```
TestTrainConfig:
  - test_default_config_values: Default degerler dogru mu
  - test_custom_config: Ozel degerler atanabiliyor mu

TestTrainOneEpoch:
  - test_runs_without_error: Kucuk synthetic data ile 1 epoch calisir
  - test_returns_metrics_dict: Donulen dict'te L_total, L_contact, L_gnm var mi
  - test_loss_decreases: 5 epoch'ta loss azaliyor mu
  - test_gradient_clipping: max_grad_norm=1.0 ile grad norm sinirli mi
  - test_nan_handling: NaN input'ta crash etmiyor mu

TestValidate:
  - test_returns_metrics: val loss, adj_acc, bf_pearson donuyor mu
  - test_no_gradient_in_validation: torch.no_grad icinde calisir mi
  - test_adj_acc_range: adj_acc [0,1] araliginda mi
  - test_bf_pearson_range: bf_pearson [-1,1] araliginda mi

TestFullTrainLoop:
  - test_small_overfit: 1 protein, 50 epoch ile overfit edebiliyor mu (loss < 0.1)
  - test_scheduler_step: OneCycleLR dogru step atiliyor mu
  - test_checkpoint_save: Best model kaydediliyor mu (mock)
```

**Synthetic Data Olusturma:**
```python
def make_synthetic_batch(n_residues=15, c_z=128):
    coords = torch.randn(n_residues, 3) * 10.0
    z = torch.randn(1, n_residues, n_residues, c_z)
    c_gt = compute_gt_probability_matrix(coords)
    return {"pair_repr": z, "coords_ca": coords, "c_gt": c_gt}
```

**Fixture icin mock model:**
```python
@pytest.fixture
def small_model():
    head = ContactProjectionHead(c_z=128, bottleneck_dim=16)
    # OF3 trunk olmadan direkt head test edilecek
    return head
```

### `tests/test_data.py`
```
TestExtractCaCoords:
  - test_returns_tensor: Tensor donuyor mu
  - test_shape_matches_residues: Shape [N,3] mi
  - test_invalid_path_raises: Gecersiz path icin hata veriyor mu

TestProteinContactDataset:
  - test_len: Dataset uzunlugu dogru mu
  - test_getitem_keys: Her item'da coords_ca, c_gt var mi
  - test_c_gt_symmetric: Ground truth simetrik mi
  - test_c_gt_diagonal_zero: Diagonal sifir mi
  - test_c_gt_in_range: Degerler [0,1] araliginda mi

TestShardedPairReprDataset:
  - test_loads_from_npz: .npz shard'dan yukler mi
  - test_len_matches_shard_content: Uzunluk dogru mu
  - test_getitem_shapes: pair_repr ve coords_ca shape'leri dogru mu
  - test_handles_variable_sizes: Farkli N degerli proteinler calisir mi
```

**Gerekli mock:** Gecici .npz shard dosyasi olustur (pytest tmp_path ile).

### `tests/test_model.py`
```
TestGNMContactLearner:
  - test_forward_output_keys: Output dict'te C_pred, pair_repr, pair_repr_recon var mi
  - test_c_pred_shape: [batch, N, N] shape mi
  - test_c_pred_in_range: [0,1] araliginda mi
  - test_c_pred_symmetric: Simetrik mi
  - test_frozen_backbone_no_grad: OF3 parametreleri donmus mu (requires_grad=False)
  - test_head_has_grad: Head parametreleri trainable mi
  - test_trainable_param_count: Sadece ~8K param trainable mi
```

**NOT:** OF3 trunk import edilemez (CUDA gerektirir). Mock trunk kullan:
```python
class MockTrunk(nn.Module):
    def forward(self, batch):
        N = batch["seq_len"]
        return {"pair_repr": torch.randn(1, N, N, 128)}
```

### Calistirma
```bash
cd /Users/berat/Projects/ANM-openfold3
python -m pytest tests/test_train.py tests/test_data.py tests/test_model.py -v
```

### Basari Kriteri
- Tum testler PASS
- Her dosya icin en az 8 test
- Synthetic data ile calisma — gercek PDB veya GPU gerektirmez
- `pytest --tb=short` ile temiz output

---

## Agent 2: Mode-Drive Test Suite

**Amac:** `src/anm.py`, `src/coords_to_contact.py`, `src/mode_combinator.py`, `src/mode_drive.py` icin kapsamli testler.

**Mevcut Durum:** Bu modullerin HICBIR testi yok.

**Olusturulacak Dosyalar:**

### `tests/test_anm.py`
```
TestBuildHessian:
  - test_shape: [3N, 3N] shape mi
  - test_symmetric: H = H^T mi (atol=1e-6)
  - test_row_sum_zero: Satir toplamlari ≈ 0 mi (elastik ag ozelligi)
  - test_positive_semidefinite: Tum eigenvalue'lar >= -1e-6 mi
  - test_cutoff_effect: Buyuk cutoff → daha fazla non-zero element
  - test_gamma_scaling: gamma=2 → H = 2 * H(gamma=1) mi
  - test_single_residue: N=1 icin [3,3] sifir matrisi mi

TestAnmModes:
  - test_eigenvalue_count: n_modes adet eigenvalue donuyor mu
  - test_eigenvector_shape: [N, n_modes, 3] shape mi
  - test_trivial_modes_skipped: Ilk 6 trivial mod atlanmis mi (eigenvalues > 1e-4)
  - test_eigenvalues_positive: Tum eigenvalue'lar pozitif mi
  - test_eigenvalues_ascending: Siralama artan mi
  - test_modes_orthogonal: Modlar birbirine dik mi (v_i · v_j ≈ 0, i≠j)
  - test_n_modes_clamped: n_modes > available durumunda clamp ediliyor mu

TestCollectivity:
  - test_output_shape: [n_modes] shape mi
  - test_range_zero_one: Tum degerler [0,1] araliginda mi
  - test_uniform_mode_max_collectivity: Uniform mod icin κ ≈ 1 mi
  - test_localized_mode_low_collectivity: Tek residue hareketinde κ ≈ 0 mi

TestComboCollectivity:
  - test_single_mode: Tek mod icin collectivity() ile ayni mi
  - test_multi_mode: Birden fazla mod icin deger donuyor mu
  - test_higher_with_collective_modes: Kollektif modlar icin daha yuksek mi

TestAnmBfactors:
  - test_output_shape: [N] shape mi
  - test_all_positive: Tum B-faktorler > 0 mi
  - test_terminal_residues_flexible: Uc residue'ler ortadakilerden daha flexible mi

TestDisplace:
  - test_output_shape: [N, 3] shape mi
  - test_zero_df_no_change: df=0 → coords degismemeli
  - test_displacement_proportional: 2*df → 2*displacement mi
  - test_center_of_mass_preserved: Non-trivial modlarda COM korunuyor mu
```

### `tests/test_coords_to_contact.py`
```
TestCoordsToContact:
  - test_output_shape: [N, N] shape mi
  - test_symmetric: C = C^T mi
  - test_diagonal_zero: C_ii = 0 mi
  - test_values_in_range: [0,1] araliginda mi
  - test_close_residues_high_contact: d < r_cut → C > 0.5 mi
  - test_far_residues_low_contact: d >> r_cut → C ≈ 0 mi
  - test_midpoint_at_rcut: d = r_cut → C ≈ 0.5 mi
  - test_tau_sharpness: Kucuk tau → daha keskin sigmoid mi
```

### `tests/test_mode_combinator.py`
```
TestModeCombo:
  - test_dataclass_fields: mode_indices, dfs, label, collectivity_score var mi

TestCollectivityCombinations:
  - test_returns_list: Liste donuyor mu
  - test_sorted_by_collectivity: Collectivity'ye gore sirali mi (azalan)
  - test_max_combos_respected: Sonuc sayisi <= max_combos mi
  - test_df_normalization: k-boyutlu combo icin df = df_base / sqrt(k) mi
  - test_single_mode_combos_present: Tek modlu kombinasyonlar var mi
  - test_multi_mode_combos_present: Coklu modlu kombinasyonlar var mi

TestGridCombinations:
  - test_returns_list: Liste donuyor mu
  - test_df_values_match_range: df degerleri belirtilen aralikta mi
  - test_max_combos_respected: Sonuc sayisi <= max_combos mi

TestRandomCombinations:
  - test_returns_list: Liste donuyor mu
  - test_correct_count: n_combos adet sonuc var mi
  - test_reproducible_with_seed: Ayni seed → ayni sonuc mu
  - test_mode_indices_valid: Tum mod indisleri [0, n_modes) araliginda mi
  - test_low_freq_bias: Dusuk frekanslı modlar daha sık seciliyor mu (eigenvalue ağırlıklı)

TestTargetedCombinations:
  - test_returns_list: Liste donuyor mu
  - test_projection_alignment: Optimal df'ler hedef yonune hizali mi
  - test_top_modes_selected: En buyuk projeksiyonlu modlar seciliyor mu
```

### `tests/test_mode_drive.py`
```
TestModeDriveConfig:
  - test_default_values: Default degerler dogru mu
  - test_custom_values: Ozel degerler atanabiliyor mu

TestModeDrivePipeline:
  - test_init_with_converter: Converter ile baslatilabilir mi
  - test_run_without_diffusion: diffusion_fn=None ile calisir mi
  - test_run_returns_result: ModeDriveResult donuyor mu
  - test_result_has_trajectory: trajectory listesi var mi ve len = n_steps+1 mi
  - test_result_has_steps: steps listesi var mi ve len = n_steps mi
  - test_rmsd_from_initial: RMSD her adimda initial'e gore mi (artiyor mu)
  - test_df_escalation: df_min'den baslayip gerektiginde ×1.5 artiyor mu
  - test_df_never_exceeds_max: df asla df_max'i gecmiyor mu
  - test_collectivity_strategy: strategy="collectivity" ile collectivity siralamasi mi
  - test_z_mixing: z_mod = α·z_pseudo + (1-α)·z_trunk formulu dogru mu
  - test_contact_map_evolution: Her adimda contact map guncelliyor mu
  - test_step_result_fields: StepResult'ta combo, displaced_ca, new_ca, contact_map, rmsd var mi

TestModeDrivePipelineEdgeCases:
  - test_two_residue_protein: N=2 ile calisir mi (minimum yapi)
  - test_large_protein: N=100 ile calisir mi
  - test_single_step: n_steps=1 ile calisir mi
  - test_zero_alpha: alpha=0 → z_mod = z_trunk mi
  - test_one_alpha: alpha=1 → z_mod = z_pseudo mi
```

**Converter mock:**
```python
class MockConverter:
    def contact_to_z(self, C):
        N = C.shape[0]
        return torch.randn(N, N, 128)
    def z_to_contact(self, z):
        N = z.shape[1]
        c = torch.rand(N, N)
        c = 0.5 * (c + c.T)
        c.fill_diagonal_(0.0)
        return c
```

### Calistirma
```bash
cd /Users/berat/Projects/ANM-openfold3
python -m pytest tests/test_anm.py tests/test_coords_to_contact.py tests/test_mode_combinator.py tests/test_mode_drive.py -v
```

### Basari Kriteri
- Tum testler PASS
- Her dosya icin en az 8 test
- Synthetic Cα koordinatları ile — GPU gerektirmez
- `coords_to_contact.py` testleri mevcut `test_ground_truth.py` ile tutarli

---

## Agent 3: Obsidian Vault Temizlik + Duzenleme

**Amac:** docs/ altindaki Obsidian vault'unu tutarli, linkli ve temiz hale getir.

**Mevcut Sorunlar:**

### 3.1 Duplicate/Cakisan Dosyalar
- `07-implementation-review.md` (18.9K) ve `07-training-plan.md` (16.3K) — iki farkli dosya ayni 07 numarasini kullaniyor
  - **Cozum:** `07-training-plan.md` → `07a-training-plan.md` olarak yeniden numaralandir VEYA icerik incelenerek birlestirilsin

### 3.2 Eksik Linkler
- `00-project-index.md`'de `07-implementation-review` ve `07-training-plan` linkleri YOK
  - **Cozum:** Her iki dosyayi da index'e ekle

### 3.3 Gereksiz/Buyuk Dosyalar
- `AGENT-COMMAND.md` (12K) — eski agent komutu, artik kullanilmiyor olabilir
- `COLAB-NOTEBOOK-COMMAND.md` (13.8K) — eski colab komutu
- `train_of_trunk_vector.md` (15K) — eski notlar
  - **Cozum:** Her birini incele. Aktif referansi yoksa `_archive/` klasorune tasi

### 3.4 Icerik Tutarliligi
- Turkce-Ingilizce karisik kullanim — index Turkce, bazi doklar Ingilizce
  - **Cozum:** Tutarlilik icin tum doklari Turkce yapmaya gerek yok, ama her dosyanin dil tercihi net olmali
- `00-project-index.md`'de `src/` listesinde bazi dosyalar eksik:
  - `data.py`, `ground_truth.py`, `inverse.py`, `model.py`, `train.py` index'te yok
  - **Cozum:** Tam src/ listesini ekle

### 3.5 Obsidian Graph Optimizasyonu
- `modules/` altindaki dosyalarin cogu OF3'un upstream modullerine ait — projeye ozgu degil
  - `diffusion-module.md`, `input-embedder.md`, `msa-module.md`, `pairformer.md`, `prediction-heads.md`
  - **Cozum:** Bunlar referans icin faydali, kalsın ama `anm-mode-drive.md` ile aralarindaki baglanti guclendirilsin

### Yapilacak Isler (sirayla)

1. **Numara cakismasi duzelt:** `07-training-plan.md` → `07a-training-plan.md` VEYA numaralandirmayi tekrar duzelt
2. **00-project-index.md guncelle:**
   - 07-implementation-review ve 07a-training-plan linklerini ekle
   - src/ listesine eksik dosyalari ekle (data.py, ground_truth.py, inverse.py, model.py, train.py)
   - tests/ bolumu ekle
   - scripts/ bolumu ekle
3. **Arsivle:**
   - `docs/_archive/` klasoru olustur
   - Aktif referansi olmayan buyuk dosyalari tasi (AGENT-COMMAND.md, COLAB-NOTEBOOK-COMMAND.md, train_of_trunk_vector.md)
   - Eger baska doklar bunlara referans veriyorsa, referanslari guncelle
4. **Cross-link guclendir:**
   - `08-anm-theory.md` → `06-gnm-math-detail.md` karsilastirma linki
   - `09-anm-mode-drive.md` → `modules/anm-mode-drive.md` referansi
   - `modules/anm-mode-drive.md` → `src/` dosya linkleri
5. **Son kontrol:**
   - Tum `[[...]]` linkleri gecerli dosyalara mi isaret ediyor?
   - Kirik link varsa duzelt
   - Her dosyanin basinda kisa aciklama var mi?

### Basari Kriteri
- `00-project-index.md` tum dosyalari icerir
- Numara cakismasi yok
- Kirik link yok
- Arsivlenen dosyalar `_archive/` altinda
- Obsidian graph'ta tum ana dosyalar bagli

---

## Agent 4: Repo Temizlik + Gereksiz Dosya Silme

**Amac:** Repo'daki gereksiz, eski veya karisikliga yol acan dosyalari temizle.

### 4.1 Git ile Takip Edilmemesi Gereken Dosyalar

`.gitignore` kontrol et ve eksikse ekle:
```
__pycache__/
*.pyc
.pytest_cache/
*.egg-info/
dist/
build/
.DS_Store
*.ipynb_checkpoints/
data/
wandb/
checkpoints/
```

### 4.2 __pycache__ Temizligi
```bash
find . -type d -name __pycache__ -not -path "./openfold3-repo/*" -not -path "./openfold3-mlx/*"
# Bulunan __pycache__'leri sil (sadece proje kok, src/, tests/, scripts/ icinde)
```

### 4.3 Notebook Temizligi
- `notebooks/full_pipeline.ipynb` — icerigini incele:
  - Aktif mi yoksa eski mi?
  - Eger artik kullanilmiyorsa arsivle veya sil
- Notebook output'larini temizle (cell output'lar git'i sisirir)

### 4.4 Empty/Placeholder Dosyalar
- `tests/__init__.py` — bos ama gerekli, kalsin
- `src/__init__.py` — export'lari kontrol et, tum modulleri export ediyor mu
- `data/` klasoru — bossa `.gitkeep` ekle veya tamamen .gitignore'a al

### 4.5 Root Dosyalar
Kontrol et:
- `requirements.txt` var mi? Yoksa olustur (temel bagimliliklar)
- `.python-version` var mi?
- `pyproject.toml` veya `setup.py` var mi? (opsiyonel)

### 4.6 Openfold3 Submodule Kontrolu
- `openfold3-repo/` ve `openfold3-mlx/` git submodule mi yoksa kopya mi?
- Eger kopya ise `.gitignore`'da mi?
- Bu klasorler cok buyuk — git repo boyutunu kontrol et

### Yapilacak Isler (sirayla)

1. `.gitignore` guncelle (eksik pattern'leri ekle)
2. `__pycache__/` dizinlerini sil (proje kok icinde)
3. `notebooks/full_pipeline.ipynb` incele ve karar ver
4. `src/__init__.py` export'larini kontrol et
5. `data/` klasorunu duzelt
6. Root'a `requirements.txt` ekle (yoksa)
7. Git status kontrol et — temiz mi?

### Basari Kriteri
- `.gitignore` kapsamli
- `__pycache__` dizinleri temiz
- Gereksiz notebook arsivlenmis
- `src/__init__.py` tum modulleri export ediyor
- Repo boyutu makul

---

## Calistirma Plani

```
Paralel:
  ├── Agent 1: tests/test_train.py + tests/test_data.py + tests/test_model.py
  ├── Agent 2: tests/test_anm.py + tests/test_coords_to_contact.py + tests/test_mode_combinator.py + tests/test_mode_drive.py
  ├── Agent 3: docs/ temizlik + index guncelleme
  └── Agent 4: repo temizlik + .gitignore + __pycache__

Sira: Her agent bagimsiz calisir, cakisma yok.
```

## Referans: Mevcut Test Stili

Tum yeni testler mevcut stil ile tutarli olmali:
- `pytest` kullan (unittest degil)
- Sinif bazli gruplama: `TestXxx`
- Helper fonksiyonlar modul seviyesinde: `_random_xxx()`
- Fixture'lar `@pytest.fixture` ile
- Assertion'lar `assert` ile (pytest magic)
- Import: `from src.xxx import Yyy`
- Dosya basi: `"""Tests for XXX."""`
