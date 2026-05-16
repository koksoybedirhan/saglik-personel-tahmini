# =====================================================================
# 09 - 2025-2030 TOPSIS ÖNEM / İHTİYAÇ SIRALAMASI
# Shannon Entropi tabanlı ağırlıklandırma
# Referans: Shannon (1948); Wang & Lee (2009, ESWA)
# =====================================================================

# %% [1] KÜTÜPHANELER VE SABİTLER ====================================
import os
import re
import numpy as np
import pandas as pd

TOPSIS_VERI_PATH = "2025_2030_tum_hedefler_final_tahmin.csv"
ENSEMBLE_PATH    = "tahmin_2025_2030_ensemble_sabit.csv"
GELECEK_YILLAR   = list(range(2025, 2031))


# %% [2] KRİTER TANIMLARI VE ENTROPİ FONKSİYONU ======================
CRITERIA_TYPES = {
    "oran_65_plus":  "benefit",   # yüksek değer = yüksek ihtiyaç
    "oran_0_14":     "benefit",
    "hekim_1000":    "cost",      # yüksek değer = düşük ihtiyaç
    "yatak_100000":  "cost",
}
CRITERIA_COLS = list(CRITERIA_TYPES.keys())


def shannon_entropi_agirlik(df, kriterler):
    """
    Shannon entropisi ile veriden objektif ağırlık hesaplar.
      1) Her kriterin kolonunu normalize et (toplam = 1)
      2) Her kriter için entropi: E_j = -k * Σ p_ij * ln(p_ij)
      3) Bilgi içeriği: d_j = 1 - E_j
      4) Ağırlık: w_j = d_j / Σ d_k  (toplam 1)
    """
    X = df[kriterler].values.astype(float)
    X = np.where(X <= 0, 1e-9, X)
    X_norm = X / X.sum(axis=0, keepdims=True)
    k = 1.0 / np.log(len(df))
    E = -k * np.sum(X_norm * np.log(X_norm), axis=0)
    d = 1.0 - E
    w = d / d.sum()
    return dict(zip(kriterler, w))


# %% [3] YARDIMCI FONKSİYONLAR =======================================
TR_MAP = str.maketrans({
    "İ": "i", "I": "i", "ı": "i",
    "Ş": "s", "ş": "s",
    "Ğ": "g", "ğ": "g",
    "Ü": "u", "ü": "u",
    "Ö": "o", "ö": "o",
    "Ç": "c", "ç": "c",
})

def normalize_city_name(x):
    if pd.isna(x):
        return np.nan
    x = str(x).strip()
    x = re.sub(r"-\d+$", "", x)
    x = x.translate(TR_MAP).lower()
    x = re.sub(r"[^a-z0-9]+", "_", x)
    x = re.sub(r"_+", "_", x).strip("_")
    return x


def assign_priority_group(rank):
    """81 il için yaklaşık öncelik grubu."""
    if rank <= 20:   return "Çok yüksek öncelik"
    elif rank <= 40: return "Yüksek öncelik"
    elif rank <= 60: return "Orta öncelik"
    else:            return "Düşük öncelik"


# %% [4] VERİLERİ OKU =================================================
topsis_raw = pd.read_csv(TOPSIS_VERI_PATH)
ensemble   = pd.read_csv(ENSEMBLE_PATH)

topsis_raw.columns = [c.strip() for c in topsis_raw.columns]
ensemble.columns   = [c.strip() for c in ensemble.columns]

topsis_raw["yıl"] = pd.to_numeric(topsis_raw["yıl"], errors="coerce").astype(int)
ensemble["yıl"]   = pd.to_numeric(ensemble["yıl"],   errors="coerce").astype(int)

if "il_key" not in topsis_raw.columns:
    topsis_raw["il_key"] = topsis_raw["il_adı"].apply(normalize_city_name)
if "il_key" not in ensemble.columns:
    ensemble["il_key"] = ensemble["il_adı"].apply(normalize_city_name)

topsis_raw = topsis_raw[topsis_raw["yıl"].isin(GELECEK_YILLAR)].copy()
ensemble   = ensemble[ensemble["yıl"].isin(GELECEK_YILLAR)].copy()

print("TOPSIS veri:",   topsis_raw.shape)
print("Ensemble veri:", ensemble.shape)


# %% [5] TOPSIS KRİTERLERİNİ HAZIRLA (rename) ========================
topsis_criteria = topsis_raw.copy()

rename_map = {
    "yas_65_plus":  "oran_65_plus",
    "yas_0_14":     "oran_0_14",
    "hekim_oran":   "hekim_1000",
    "yatak_oran":   "yatak_100000",
}
topsis_criteria = topsis_criteria.rename(columns=rename_map)

required_topsis_cols = [
    "il_adı", "il_key", "yıl", "nufus",
    "oran_65_plus", "oran_0_14", "hekim_1000", "yatak_100000"
]
missing_cols = [c for c in required_topsis_cols if c not in topsis_criteria.columns]
if missing_cols:
    raise ValueError(f"TOPSIS kriter veri setinde eksik kolonlar var: {missing_cols}")

topsis_criteria = topsis_criteria[required_topsis_cols].copy()

for col in ["nufus"] + CRITERIA_COLS:
    topsis_criteria[col] = pd.to_numeric(topsis_criteria[col], errors="coerce")

print("TOPSIS kriter matrisi:", topsis_criteria.shape)


# %% [6] ENSEMBLE İLE BİRLEŞTİR ======================================
ensemble_cols = ["il_key", "yıl", "doktor", "ebe", "diger", "hemsire", "yatak", "hastane"]
available_ensemble_cols = [c for c in ensemble_cols if c in ensemble.columns]

future = topsis_criteria.merge(
    ensemble[available_ensemble_cols],
    on=["il_key", "yıl"],
    how="left"
)
print("Birleşmiş TOPSIS + ensemble veri:", future.shape)


# %% [7] ★ ENTROPİ TABANLI AĞIRLIK HESAPLA ★ ========================
# Referans yıl: 2025 (gerçek yaş verisi olan en güncel yıl)
ref_df = future[future["yıl"] == 2025].dropna(subset=CRITERIA_COLS)
WEIGHTS = shannon_entropi_agirlik(ref_df, CRITERIA_COLS)

print("\n" + "="*60)
print("ENTROPİ TABANLI TOPSIS AĞIRLIKLARI (2025 referansı)")
print("="*60)
for k_, v_ in WEIGHTS.items():
    print(f"  {k_:<16} {v_:.4f}")
print(f"  {'TOPLAM':<16} {sum(WEIGHTS.values()):.4f}")
print("="*60 + "\n")


# %% [8] AĞIRLIKLI HAM KRİTERLER (sadece görselleştirme amaçlı) =====
for col, w in WEIGHTS.items():
    future[f"{col}_agirlikli_ham"] = future[col] * w


# %% [9] TOPSIS FONKSİYONU ============================================
def calculate_topsis_for_one_year(df_year, criteria_cols, weights, criteria_types):
    """Tek bir yıl için iller arası TOPSIS skoru hesaplar."""
    d = df_year.copy().reset_index(drop=True)

    # Eksik değerleri yıl medyanı ile doldur
    for col in criteria_cols:
        d[col] = pd.to_numeric(d[col], errors="coerce")
        d[col] = d[col].fillna(d[col].median())

    X = d[criteria_cols].astype(float).values

    # 1) Vektör normalizasyonu
    denom = np.sqrt((X ** 2).sum(axis=0))
    denom = np.where(denom == 0, 1e-12, denom)
    X_norm = X / denom

    # 2) Ağırlıklandırma
    weight_vector = np.array([weights[col] for col in criteria_cols], dtype=float)
    X_weighted = X_norm * weight_vector

    for idx, col in enumerate(criteria_cols):
        d[f"{col}_norm"]      = X_norm[:, idx]
        d[f"{col}_agirlikli"] = X_weighted[:, idx]

    # 3) İdeal iyi/kötü
    ideal_best, ideal_worst = [], []
    for idx, col in enumerate(criteria_cols):
        if criteria_types[col] == "benefit":
            ideal_best.append(X_weighted[:, idx].max())
            ideal_worst.append(X_weighted[:, idx].min())
        elif criteria_types[col] == "cost":
            ideal_best.append(X_weighted[:, idx].min())
            ideal_worst.append(X_weighted[:, idx].max())
        else:
            raise ValueError(f"Bilinmeyen kriter tipi: {col}")

    ideal_best  = np.array(ideal_best)
    ideal_worst = np.array(ideal_worst)

    # 4) Uzaklıklar
    d_plus  = np.sqrt(((X_weighted - ideal_best)  ** 2).sum(axis=1))
    d_minus = np.sqrt(((X_weighted - ideal_worst) ** 2).sum(axis=1))

    # 5) Skor
    topsis_score = d_minus / (d_plus + d_minus + 1e-12)

    d["topsis_skor"]        = topsis_score
    d["ideal_iyi_uzaklik"]  = d_plus
    d["ideal_kotu_uzaklik"] = d_minus

    # 6) Sıralama + öncelik grubu
    d["topsis_sira"]   = d["topsis_skor"].rank(ascending=False, method="dense").astype(int)
    d["oncelik_grubu"] = d["topsis_sira"].apply(assign_priority_group)
    return d


# %% [10] HER YIL İÇİN TOPSIS ÇALIŞTIR ===============================
topsis_results = []
for year in sorted(future["yıl"].dropna().unique()):
    df_year = future[future["yıl"] == year].copy()
    result_year = calculate_topsis_for_one_year(
        df_year=df_year,
        criteria_cols=CRITERIA_COLS,
        weights=WEIGHTS,
        criteria_types=CRITERIA_TYPES
    )
    topsis_results.append(result_year)

topsis_all = pd.concat(topsis_results, ignore_index=True)
topsis_all = topsis_all.sort_values(["yıl", "topsis_sira"]).reset_index(drop=True)

print("TOPSIS sonuçları:", topsis_all.shape)


# %% [11] İHTİYAÇ PAYI (yıl içinde normalize) ========================
topsis_all["ihtiyac_payi"] = topsis_all["topsis_skor"]
topsis_all["ihtiyac_payi_norm"] = (
    topsis_all.groupby("yıl")["ihtiyac_payi"]
    .transform(lambda x: x / x.sum() if x.sum() != 0 else 1 / len(x))
)


# %% [12] ÇIKTI TABLOLARINI KAYDET ===================================
score_cols = [
    "il_adı", "il_key", "yıl", "nufus",
    "oran_65_plus", "oran_0_14", "hekim_1000", "yatak_100000",
    "oran_65_plus_agirlikli_ham", "oran_0_14_agirlikli_ham",
    "hekim_1000_agirlikli_ham",   "yatak_100000_agirlikli_ham",
    "oran_65_plus_norm", "oran_0_14_norm", "hekim_1000_norm", "yatak_100000_norm",
    "oran_65_plus_agirlikli", "oran_0_14_agirlikli",
    "hekim_1000_agirlikli",   "yatak_100000_agirlikli",
    "ideal_iyi_uzaklik", "ideal_kotu_uzaklik",
    "topsis_skor", "topsis_sira", "oncelik_grubu", "ihtiyac_payi_norm",
]
for col in ["doktor", "ebe", "diger", "hemsire", "yatak", "hastane"]:
    if col in topsis_all.columns:
        score_cols.append(col)

scores_out = topsis_all[score_cols].sort_values(["yıl", "topsis_sira"]).reset_index(drop=True)

weighted_cols = [
    "il_adı", "il_key", "yıl",
    "oran_65_plus", "oran_65_plus_agirlikli_ham", "oran_65_plus_agirlikli",
    "oran_0_14",    "oran_0_14_agirlikli_ham",    "oran_0_14_agirlikli",
    "hekim_1000",   "hekim_1000_agirlikli_ham",   "hekim_1000_agirlikli",
    "yatak_100000", "yatak_100000_agirlikli_ham", "yatak_100000_agirlikli",
    "topsis_skor", "topsis_sira", "oncelik_grubu"
]
weighted_out = topsis_all[weighted_cols].sort_values(["yıl", "topsis_sira"]).reset_index(drop=True)

scores_out.to_csv("09_topsis_onem_siralamasi_2025_2030.csv",
                  index=False, encoding="utf-8-sig")
weighted_out.to_csv("09_topsis_agirlikli_kriterler_2025_2030.csv",
                    index=False, encoding="utf-8-sig")

with pd.ExcelWriter("09_topsis_tek_dosya_2025_2030.xlsx") as writer:
    scores_out.to_excel(writer,   sheet_name="TOPSIS_Onem_Siralamasi", index=False)
    weighted_out.to_excel(writer, sheet_name="Agirlikli_Kriterler",    index=False)
    pd.DataFrame({
        "Kriter":  list(WEIGHTS.keys()),
        "Ağırlık": list(WEIGHTS.values()),
        "Tip":     [CRITERIA_TYPES[k] for k in WEIGHTS.keys()],
        "Yöntem":  ["Shannon Entropi"] * len(WEIGHTS),
        "Yorum": [
            "65+ oranı arttıkça ihtiyaç artar",
            "0-14 oranı arttıkça ihtiyaç artar",
            "1000 kişi başına hekim azaldıkça ihtiyaç artar",
            "100 bin kişi başına yatak azaldıkça ihtiyaç artar",
        ]
    }).to_excel(writer, sheet_name="Kriter_Agirliklari", index=False)

print("\n[OK] Çıktılar kaydedildi:")
print("  - 09_topsis_onem_siralamasi_2025_2030.csv")
print("  - 09_topsis_agirlikli_kriterler_2025_2030.csv")
print("  - 09_topsis_tek_dosya_2025_2030.xlsx")


# %% [13] 2025 İLK 20 ÖNCELİK ========================================
print("\n========== 2025 İÇİN İLK 20 ÖNCELİKLİ İL ==========")
ilk20 = topsis_all[topsis_all["yıl"] == 2025].head(20)[
    ["topsis_sira", "il_adı", "topsis_skor",
     "oran_65_plus", "oran_0_14", "hekim_1000", "yatak_100000"]
]
print(ilk20.round(3).to_string(index=False))
