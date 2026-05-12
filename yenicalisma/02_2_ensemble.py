# =====================================================================
# 02_2 - PER-TARGET WEIGHTED ENSEMBLE
#
# Mantık:
#   Her hedef için Fold-5 Accuracy'ye göre TOP-K model seçilir.
#   Ağırlıklar w_i = Accuracy_i / sum(Accuracy_top) — normalleştirilir.
#   Ensemble tahmin = sum(w_i * tahmin_i)
#
# Girdiler:
#   - tahminler_fold5_2021_2024.csv   (her modelin Fold-5 tahminleri)
#   - model_metrikleri_fold.csv        (model × hedef × fold metrikleri)
#
# Çıktılar:
#   - ensemble_secimleri.csv           (her hedef için top-K + ağırlıklar)
#   - tahminler_ensemble_fold5.csv     (ensemble Fold-5 tahminleri)
#   - ensemble_metrikleri.csv          (ensemble metrikleri)
#   - karsilastirma_ensemble_vs_tekli.csv  (ensemble vs en iyi tek model)
# =====================================================================

# %% [1] KÜTÜPHANELER VE PARAMETRELER ================================
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                             r2_score, precision_score, recall_score, f1_score)

TOP_K = 3         # her hedef için kaç model birleştirilecek
AGIRLIK_TIPI = "accuracy"   # "accuracy" → Acc'a oranlı,  "inv_mae" → 1/MAE'ye oranlı

print(f"TOP_K = {TOP_K}, ağırlık tipi = {AGIRLIK_TIPI}")

# %% [2] VERİLERİ YÜKLE ==============================================
tahminler = pd.read_csv("tahminler_fold5_2021_2024.csv")
metrikler_df = pd.read_csv("model_metrikleri_fold.csv")

print("Tahminler:", tahminler.shape, "| model sayısı:", tahminler["model"].nunique())
print("Modeller:", sorted(tahminler["model"].unique()))
print("Hedefler:", sorted(tahminler["hedef"].unique()))

# %% [3] HER HEDEF İÇİN TOP-K MODEL + AĞIRLIK ========================
ana_fold = metrikler_df[metrikler_df["Fold"] == 5].copy()

secim_satirlari = []
agirlik_map = {}   # {hedef: {model: weight}}

for hedef in ana_fold["Hedef"].unique():
    bu = ana_fold[ana_fold["Hedef"] == hedef].sort_values("Accuracy", ascending=False)
    bu = bu[bu["Model"].isin(tahminler["model"].unique())]   # tahmini olanlar
    top = bu.head(TOP_K).copy()
    if AGIRLIK_TIPI == "accuracy":
        w_raw = top["Accuracy"].values
    elif AGIRLIK_TIPI == "inv_mae":
        w_raw = 1.0 / (top["MAE"].values + 1e-9)
    else:
        raise ValueError("AGIRLIK_TIPI 'accuracy' veya 'inv_mae' olmalı")
    w = w_raw / w_raw.sum()
    top["Agirlik"] = w
    secim_satirlari.append(top[["Hedef", "Model", "Accuracy", "MAE", "RMSE", "Agirlik"]])
    agirlik_map[hedef] = dict(zip(top["Model"].values, w))

secimler = pd.concat(secim_satirlari, ignore_index=True)
secimler.to_csv("ensemble_secimleri.csv", index=False, encoding="utf-8-sig")
print("\n========== HER HEDEF İÇİN TOP-K MODEL VE AĞIRLIKLAR ==========")
print(secimler.to_string(index=False))

# %% [4] ENSEMBLE TAHMİNİ HESAPLA ====================================
# Her (il, yıl, hedef) için top-K model tahminlerinin ağırlıklı ortalaması.
ens_satirlari = []
for hedef, agirlik in agirlik_map.items():
    alt = tahminler[(tahminler["hedef"] == hedef) &
                    (tahminler["model"].isin(agirlik.keys()))].copy()
    if alt.empty: continue
    # pivot: (il_adı, yıl) × model
    piv = alt.pivot_table(index=["il_adı", "yıl"], columns="model",
                          values="tahmin", aggfunc="mean")
    # ağırlıklı ortalama
    weights = np.array([agirlik.get(m, 0) for m in piv.columns])
    weights = weights / weights.sum()
    # eksik (NaN) tahmini olan modeli o satırda saymadan ortalama:
    mask = ~piv.isna().values
    agirlik_matrisi = np.tile(weights, (len(piv), 1)) * mask
    # satır toplamlarını sıfırdan koru
    satir_top = agirlik_matrisi.sum(axis=1, keepdims=True)
    satir_top[satir_top == 0] = 1
    agirlik_matrisi = agirlik_matrisi / satir_top
    ens_pred = np.nansum(piv.values * agirlik_matrisi, axis=1)
    # gerçek değerleri ekle (model fark etmez, ilk modelden çek)
    gercek_piv = alt.pivot_table(index=["il_adı", "yıl"], columns="model",
                                 values="gercek", aggfunc="first")
    gercek = gercek_piv.bfill(axis=1).iloc[:, 0].values
    for (il, yr), g, p in zip(piv.index, gercek, ens_pred):
        ens_satirlari.append(dict(il_adı=il, yıl=int(yr), hedef=hedef,
                                  model="Ensemble",
                                  gercek=float(g), tahmin=float(p)))

ensemble_df = pd.DataFrame(ens_satirlari)
ensemble_df.to_csv("tahminler_ensemble_fold5.csv", index=False, encoding="utf-8-sig")
print(f"\nEnsemble tahminleri: {len(ensemble_df)} satır")
print(ensemble_df.head())

# %% [5] ENSEMBLE METRİKLERİ =========================================
def metrikler_hesapla(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2   = r2_score(y_true, y_pred) if len(np.unique(y_true)) > 1 else np.nan
    mask = y_true != 0
    mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100) if mask.any() else np.nan
    acc  = max(0.0, 100.0 - mape) if not np.isnan(mape) else np.nan
    med  = float(np.median(y_true))
    yt_cls = (y_true >= med).astype(int)
    yp_cls = (y_pred >= med).astype(int)
    prec = precision_score(yt_cls, yp_cls, zero_division=0)
    rec  = recall_score(yt_cls, yp_cls, zero_division=0)
    f1   = f1_score(yt_cls, yp_cls, zero_division=0)
    return dict(Accuracy=acc, MAE=mae, RMSE=rmse, R2=r2,
                ErrorRate=mape, Precision=prec, Recall=rec, F1=f1)

ens_metrikleri = []
for hedef in ensemble_df["hedef"].unique():
    alt = ensemble_df[ensemble_df["hedef"] == hedef]
    m = metrikler_hesapla(alt["gercek"].values, alt["tahmin"].values)
    m.update(dict(Hedef=hedef, Model="Ensemble"))
    ens_metrikleri.append(m)

ens_metrikleri_df = pd.DataFrame(ens_metrikleri)
ens_metrikleri_df.to_csv("ensemble_metrikleri.csv", index=False, encoding="utf-8-sig")
print("\n========== ENSEMBLE METRİKLERİ (HEDEF BAZINDA) ==========")
print(ens_metrikleri_df[["Hedef", "Accuracy", "MAE", "RMSE", "R2",
                         "Precision", "Recall", "F1"]].to_string(index=False))

# %% [6] KARŞILAŞTIRMA: ENSEMBLE vs EN İYİ TEK MODEL =================
en_iyi_tekli = (ana_fold.sort_values("Accuracy", ascending=False)
                .groupby("Hedef").head(1)
                [["Hedef", "Model", "Accuracy", "MAE", "RMSE", "R2"]])

ens_kolonlar = ens_metrikleri_df[["Hedef", "Accuracy", "MAE", "RMSE", "R2"]].copy()
ens_kolonlar.columns = ["Hedef", "Ens_Accuracy", "Ens_MAE", "Ens_RMSE", "Ens_R2"]
en_iyi_tekli.columns = ["Hedef", "EnIyi_Model", "Tekli_Accuracy", "Tekli_MAE",
                        "Tekli_RMSE", "Tekli_R2"]

kars = en_iyi_tekli.merge(ens_kolonlar, on="Hedef")
kars["Kazanan"] = np.where(kars["Ens_Accuracy"] > kars["Tekli_Accuracy"],
                            "Ensemble", "Tekli")
kars["Acc_Fark"] = (kars["Ens_Accuracy"] - kars["Tekli_Accuracy"]).round(3)

kars.to_csv("karsilastirma_ensemble_vs_tekli.csv", index=False, encoding="utf-8-sig")
print("\n========== ENSEMBLE vs EN İYİ TEK MODEL ==========")
print(kars.to_string(index=False))

# %% [7] ÖZET ========================================================
ens_genel_acc = ens_metrikleri_df["Accuracy"].mean()
tekli_genel_acc = en_iyi_tekli["Tekli_Accuracy"].mean()

print(f"\n──────────────────────────────────────────────────")
print(f"Genel ortalama Accuracy:")
print(f"  Ensemble (top-{TOP_K}):  {ens_genel_acc:.3f}%")
print(f"  Tek başına en iyi:     {tekli_genel_acc:.3f}%")
print(f"  Fark:                  {ens_genel_acc - tekli_genel_acc:+.3f} puan")
print(f"──────────────────────────────────────────────────")

kazanan_sayi = (kars["Kazanan"] == "Ensemble").sum()
print(f"Ensemble {kazanan_sayi}/{len(kars)} hedefte daha iyi")
if ens_genel_acc > tekli_genel_acc:
    print("→ ÖNERİ: Ensemble yaklaşımını kullan (03_gelecek_tahmin.py için)")
else:
    print("→ ÖNERİ: Tek başına en iyi modeli kullan (per-target selection)")
