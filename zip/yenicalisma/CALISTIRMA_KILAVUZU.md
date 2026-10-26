# Tahmin Pipeline Çalıştırma Kılavuzu

## Klasör yapısı
```
yenicalisma/
├── all_data24.csv                # ham veri (uploads'dan)
├── 01_veri_hazirla.py            # Adım 1
├── 02_model_egitim.py            # Adım 2 (tek dosyada tüm modeller)
├── 03_gelecek_tahmin.py          # Adım 3 (2025-2030 tahmini)
├── hamveriler/                   # klasör
└── temizveriler/                 # klasör
```

## Kurulum (1 defalık)
Anaconda Prompt veya terminalde:
```bash
pip install pandas numpy scikit-learn xgboost catboost statsmodels openpyxl
# Opsiyonel: pip install prophet
```

## Çalıştırma sırası

### 1) Veriyi hazırla
```bash
cd "C:\Users\Bedirhan\Desktop\ARSEK\saglik-personel-yer-secimi\yenicalisma"
python 01_veri_hazirla.py
```
**Süre:** ~10 saniye
**Çıktı:** `model_giris.csv` (1863 satır, 11 kolon)

### 2) Modelleri eğit + metrikleri hesapla
```bash
python 02_model_egitim.py
```
**Süre:** ~15-25 dakika

**Çıktılar:**
- `model_metrikleri_detay.csv` — her model × hedef × fold için tüm metrikler
- `model_metrikleri_fold.csv` — fold ortalamaları
- `model_metrikleri_genel.csv` — **GENEL MODEL KARŞILAŞTIRMA TABLOSU** (makaleye gidecek)
- `en_iyi_modeller.csv` — her hedef için Fold-5'te en iyi model
- `tahminler_fold5_2021_2024.csv` — son fold test tahminleri

### 3) 2025-2030 gelecek tahmini
```bash
python 03_gelecek_tahmin.py
```
**Süre:** ~1-2 dakika
**Çıktı:** `tahmin_2025_2030.xlsx` ve `.csv`

## K-fold yapısı (5 fold, rolling expanding)

| Fold | Train     | Test      |
|------|-----------|-----------|
| 1    | 2002-2016 | 2017-2020 |
| 2    | 2002-2017 | 2018-2021 |
| 3    | 2002-2018 | 2019-2022 |
| 4    | 2002-2019 | 2020-2023 |
| **5** | **2002-2020 (19 yıl)** | **2021-2024 (4 yıl)** ← ANA FOLD |

## Modeller (toplam 12)
1. LinearRegression
2. RandomForest
3. ExtraTrees
4. GradientBoosting
5. XGBoost
6. CatBoost
7. SVR
8. KNN
9. Bagging
10. Voting (RF + XGB + GB)
11. Stacking (RF + XGB → Linear meta-learner)
12. ARIMA (univariate, her il için ayrı eğitilir)
13. Prophet — opsiyonel (script içinde `PROPHET_ACTIVE = True` yap, ~30-60 dk ekler)

## Hesaplanan metrikler
- **Accuracy** = 100 − MAPE (önceki çalışmadaki tanım korunmuştur)
- **MAE, RMSE, R²** — regresyon standartı
- **Error Rate** = MAPE
- **Precision, Recall, F1** — Tahminler ülke medyanına göre "yüksek/düşük" sınıflandırılır, metrikler bu ikili sınıflandırma üzerinden hesaplanır. Yöntem bölümünde belirtilmelidir.

## Sorun çıkarsa
- **CatBoost yüklenmiyorsa:** Script otomatik atlar (`HAVE_CAT = False` uyarısı verir).
- **Statsmodels yoksa:** ARIMA atlanır.
- **Bellek yetmezse:** Script içinden `Voting` ve `Stacking` modellerini `yeni_modeller()` fonksiyonundan çıkar. Süre yarıya iner.
- **Çok yavaşsa:** `n_estimators` parametrelerini 100'e indir (model fabrikası içinde).

## Akış özeti
```
all_data24.csv
    ↓ (01_veri_hazirla.py)
model_giris.csv
    ↓ (02_model_egitim.py)
model_metrikleri_genel.csv  +  en_iyi_modeller.csv
    ↓ (03_gelecek_tahmin.py)
tahmin_2025_2030.xlsx
    ↓
[bir sonraki aşama: TOPSIS analizi]
```
