# 10 - Iteratif Refinement Stratejisi

> ANM Mode-Drive pipeline'inin iterasyon dinamikleri, df eskalasyonu ve failure modlari.

## 1. Calisma Prensibi

Pipeline **sabit n_steps** adim calisir (erken durma yok). Her adimda:
1. Mevcut koordinatlardan ANM Hessian + modlar hesaplanir
2. Mod kombinasyonlari collectivity'ye gore siralanir
3. En kollektif combo denenir → RMSD initial'den artiyorsa kabul
4. Artmiyorsa → sonraki combo → artmiyorsa → df eskalasyonu

**RMSD olcumu:** Her zaman **initial structure**'a gore. Yuksek RMSD = daha fazla konformasyonel kesif = iyi.

## 2. df Eskalasyon Mekanizmasi

### 2.1 Neden Eskalasyon?

Bazen mevcut df degeri yeterli deplasman saglamaz:
- Protein cok rijit (yuksek eigenvalue'lar)
- Secilen modlar lokal (az residue hareket ediyor)
- Onceki adimda zaten cok uzaklasilmis, artik kucuk df'ler etkisiz

### 2.2 Eskalasyon Algoritmasi

```
current_df = df_min (0.3)

while current_df <= df_max (3.0):
    combo'lari collectivity sirasinda dene:
        RMSD(initial, new) > prev_rmsd?
            → Evet: KABUL, dur
            → Hayir: sonraki combo

    Hicbir combo basariliysa:
        current_df *= escalation_factor (1.5)
        Combo'lari yeni df ile yeniden uret
        Tekrar dene

En iyi bulunan combo'yu dondur
```

### 2.3 df Degerleri Ornegi

```
df_min = 0.3  →  0.45  →  0.675  →  1.01  →  1.52  →  2.28  →  df_max = 3.0
              ×1.5    ×1.5      ×1.5     ×1.5     ×1.5
```

7 eskalasyon adimi ile 0.3'ten 3.0'a ulasilir.

## 3. Iterasyon Dinamikleri

### Erken Adimlar (1-2)

- prev_rmsd dusuk (0 veya kucuk) → kucuk df yeterli
- En kollektif combo genelde hemen kabul edilir
- Buyuk olcekli hareketler: hinge bending, domain motion
- df_min = 0.3 genelde yeterli

### Orta Adimlar (3-4)

- prev_rmsd artiyor → kabul esigi yukseliyor
- df eskalasyonu gerekebilir
- Yeni Hessian'dan farkli modlar cikar (yapi degisti)
- Orta frekanslı modlar devreye girer

### Gec Adimlar (5+)

- prev_rmsd yuksek → RMSD'yi daha da artirmak zorlasiyor
- df eskalasyonu daha sik
- Diminishing returns baslar
- Pipeline dogal olarak yavaslayan kesif oranina sahip

## 4. z_ij Evrimi

Her adimda hem koordinatlar hem z_ij guncellenir:

```python
# Her adim sonrasi
coords_ca = step_result.new_ca           # yeni koordinatlar
z_current = step_result.z_modified       # yeni z_ij (blend edilmis)
```

Bu **iteratif z_ij evrimi** kritik: sadece koordinatlar degil, diffusion'a giren z_ij de adim adim degisiyor. Boylece her adimda diffusion farkli bir sinyal goruyor.

## 5. Collectivity Dinamikleri

### Neden Collectivity Onemli?

| Mod Tipi | Collectivity | Anlam |
|----------|-------------|-------|
| Hinge bending | κ ~ 0.7-0.9 | Tum protein sallanir → buyuk RMSD |
| Domain shear | κ ~ 0.5-0.7 | Domain kayar → orta RMSD |
| Loop motion | κ ~ 0.1-0.3 | Sadece bir loop → kucuk RMSD |
| Side-chain vibration | κ < 0.1 | Cok lokal → RMSD degismez |

Pipeline en kollektif combo'yu once dener → en buyuk RMSD artisini hedefler.

### Combo Collectivity Hesabi

Tek mod icin standart collectivity kullanilir. Coklu mod icin:
```
combined_i = v_mode1_i + v_mode2_i + ...    # vektorel toplam
κ_combo = collectivity(combined)            # toplam vektorun collectivity'si
```

Bu, modlarin birlikte calisip calismadigini olcer:
- **Yuksek κ_combo:** Modlar ayni residue'lari hareket ettiriyor (constructive)
- **Dusuk κ_combo:** Modlar birbirini gotureyor (destructive)

## 6. Failure Modlari ve Cozumleri

### 6a. z_pseudo Dagilim Uyumsuzlugu

**Belirti:** Diffusion NaN/Inf uretir veya anlamsiz yapilar cikarir.

**Cozum:**
- alpha'yi 0.1'e dusur (`z_mixing_alpha`)
- `normalize_z=True` kontrol et
- z_pseudo ve z_trunk istatistiklerini logla

### 6b. RMSD Artmiyor (Stagnation)

**Belirti:** Her step ayni RMSD, yapi degismiyor.

**Cozum:**
- `df_max`'i artir (3.0 → 5.0)
- `max_combo_size`'i artir (3 → 5)
- Daha fazla mod kullan (`n_anm_modes` 20 → 30)

### 6c. RMSD Cok Hizli Artiyor

**Belirti:** Yapilar fiziksel olarak anlamsizlasiyor.

**Cozum:**
- `df_min` ve `df_max`'i dusur
- `z_mixing_alpha`'yi dusur (trunk bilgisini daha cok koru)
- `n_steps`'i azalt

### 6d. Buyuk Protein Bellek Sorunu

**Belirti:** N > 500 icin Hessian buyuk.

**Cozum:**
- N=500 → Hessian = 1500x1500 = 9 MB (sorun degil)
- N=1000 → Hessian = 3000x3000 = 36 MB (OK)
- N>2000 icin sparse Hessian gerekebilir

### 6e. Combinatorik Patlama

**Belirti:** `max_combo_size=5`, `n_modes=20` → C(20,5) = 15504 aday.

**Cozum:**
- `max_combos` parametresi ile sinirla (default 50)
- `max_combo_size=3` ile basla (1350 aday, hizli)

## 7. Ornek Calisma Akisi

```
Protein: 200 residue, tek zincir
Config: n_steps=5, df_min=0.3, df_max=3.0, max_combo_size=3

Step 1: df=0.3, combo(0,1) κ=0.82  → RMSD=1.2 Å
Step 2: df=0.3, combo(0,2) κ=0.75  → RMSD=2.1 Å
Step 3: df=0.45 (escalation), combo(0) → RMSD=2.8 Å
Step 4: df=0.675 (escalation), combo(1,2) → RMSD=3.3 Å
Step 5: df=0.675, combo(0,1,2) κ=0.69 → RMSD=3.9 Å

Trajectory: 6 yapi (initial + 5 step)
RMSD: 0 → 1.2 → 2.1 → 2.8 → 3.3 → 3.9 Å
```

## Iliskili Dokumanlar

- [[09-anm-mode-drive]] — Pipeline mimarisi, collectivity stratejisi
- [[08-anm-theory]] — ANM matematigi, collectivity formulu
- [[05-gnm-contact-learner]] — ContactProjectionHead inverse path
