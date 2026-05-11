# =====================================================================
# 03 - 2025-2030 İTERATİF (RECURSIVE) GELECEK TAHMİNİ
#
# Mantık:
#  * Her hedef değişken için 02 scriptinde Fold-5'te EN İYİ MODEL seçildi.
#  * Bu modeli 2002-2024 tüm veriyle yeniden eğitiyoruz.
#  * 2025'i tahmin ederken 2024 değerleri girdi.
#  * 2026'yı tahmin ederken 2025 TAHMİNİ girdi (recursive forecast).
#  * ...2030'a kadar.
#  * Nüfus için TÜİK'in 2025-2030 projeksiyonu kullanılır (varsa);
#    yoksa lineer trend ile tahmin edilir.
#
# Çıktı: tahmin_2025_2030.xlsx
# =====================================================================

import warnings, os
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.ensemble     import (RandomForestRegressor, ExtraTreesRegressor,
                                  GradientBoostingRegressor, BaggingRegressor,
                                  VotingRegressor, StackingRegressor)
from sklearn.svm          import SVR
from sklearn.neighbors    import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from xgboost              import XGBRegressor
try:
    from catboost import CatBoostRegressor
    HAVE_CAT = True
except ImportError:
    HAVE_CAT = False

# ----------- 1) VERİYİ YÜKLE -----------
df = pd.read_csv("model_giris.csv").sort_values(["il_adı", "yıl"]).reset_index(drop=True)
en_iyi = pd.read_csv("en_iyi_modeller.csv")
print("En iyi modeller:\n", en_iyi)

feature_map = {
    "uzman":     ["yıl", "nufus", "hastane", "yatak", "asistan", "pratisyen", "hemsire", "ebe", "diger"],
    "pratisyen": ["yıl", "nufus", "hastane", "yatak", "asistan", "uzman",     "hemsire", "ebe", "diger"],
    "asistan":   ["yıl", "nufus", "hastane", "yatak", "uzman",   "pratisyen", "hemsire", "ebe", "diger"],
    "ebe":       ["yıl", "nufus", "hastane", "yatak", "asistan", "uzman",     "hemsire", "pratisyen", "diger"],
    "diger":     ["yıl", "nufus", "hastane", "yatak", "asistan", "pratisyen", "hemsire", "ebe", "uzman"],
    "hemsire":   ["yıl", "nufus", "hastane", "yatak", "asistan", "uzman",     "pratisyen", "ebe", "diger"],
    "yatak":     ["yıl", "nufus", "hastane", "hemsire", "asistan", "uzman",   "pratisyen", "ebe", "diger"],
    "hastane":   ["yıl", "nufus", "yatak",   "hemsire", "asistan", "uzman",   "pratisyen", "ebe", "diger"],
}

# ----------- 2) MODEL FABRİKASI (en_iyiler.csv'deki ada göre) -----------
def model_olustur(ad):
    if ad == "LinearRegression":   return LinearRegression()
    if ad == "RandomForest":       return RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    if ad == "ExtraTrees":         return ExtraTreesRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    if ad == "GradientBoosting":   return GradientBoostingRegressor(n_estimators=300, random_state=42)
    if ad == "XGBoost":            return XGBRegressor(n_estimators=400, random_state=42, verbosity=0,
                                                       objective="reg:squarederror")
    if ad == "CatBoost" and HAVE_CAT: return CatBoostRegressor(iterations=400, depth=6, verbose=0, random_seed=42)
    if ad == "SVR":                return SVR(C=10, epsilon=0.1)
    if ad == "KNN":                return KNeighborsRegressor(n_neighbors=5)
    if ad == "Bagging":            return BaggingRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    if ad == "Voting":
        return VotingRegressor([("rf",  RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)),
                                ("xgb", XGBRegressor(n_estimators=300, random_state=42, verbosity=0,
                                                     objective="reg:squarederror")),
                                ("gb",  GradientBoostingRegressor(n_estimators=300, random_state=42))])
    if ad == "Stacking":
        return StackingRegressor(
            estimators=[("rf",  RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)),
                        ("xgb", XGBRegressor(n_estimators=300, random_state=42, verbosity=0,
                                             objective="reg:squarederror"))],
            final_estimator=LinearRegression(), n_jobs=-1)
    # ARIMA/Prophet için ayrıca işlem yapılacak - default XGB döndür
    return XGBRegressor(n_estimators=400, random_state=42, verbosity=0,
                        objective="reg:squarederror")

# ----------- 3) NÜFUS PROJEKSİYONU 2025-2030 -----------
def nufus_projeksiyonu(df):
    """TÜİK projeksiyonu yoksa il-bazlı lineer ekstrapolasyon."""
    proj_yol = "tuik26-30hesaplanmis/Turkiye_Nufus_Projeksiyon_Guncellenmis.xlsx"
    sonuc = {}
    if os.path.exists(proj_yol):
        print("TÜİK projeksiyonu kullanılıyor:", proj_yol)
        tp = pd.read_excel(proj_yol)
        # Tüm il ve yıllar varsa burada uygun şekilde çevir
        for _, r in tp.iterrows():
            il = r.get("il_adı") or r.iloc[0]
            for y in range(2025, 2031):
                if y in tp.columns:
                    sonuc[(il, y)] = float(r[y])
        if sonuc:
            return sonuc
        print("[UYARI] Projeksiyon dosyası beklenen formatta değil — lineer ekstrapolasyona düşülüyor.")

    print("Lineer trend ile nüfus ekstrapolasyonu yapılıyor.")
    for il in df["il_adı"].unique():
        d = df[df["il_adı"] == il].dropna(subset=["nufus"])
        if len(d) < 3:
            continue
        # Son 10 yılın trendiyle
        d = d.sort_values("yıl").tail(10)
        coef = np.polyfit(d["yıl"], d["nufus"], 1)
        for y in range(2025, 2031):
            sonuc[(il, y)] = float(np.polyval(coef, y))
    return sonuc

nufus_geleceği = nufus_projeksiyonu(df)

# ----------- 4) HER HEDEF İÇİN MODEL EĞİT (2002-2024 tüm veri) -----------
modeller = {}
scalers  = {}
for _, r in en_iyi.iterrows():
    h, ad = r["Hedef"], r["Model"]
    if h not in feature_map: continue
    feats = feature_map[h]
    train = df[["il_adı", "yıl"] + feats + [h]].dropna()
    X = train[feats].values
    y = train[h].values
    mdl = model_olustur(ad)
    sc  = StandardScaler().fit(X) if ad in ("SVR", "KNN") else None
    mdl.fit(sc.transform(X) if sc is not None else X, y)
    modeller[h] = mdl
    scalers[h]  = sc
    print(f"[OK] {h:<10} → {ad}")

# ----------- 5) RECURSIVE 2025-2030 TAHMİNİ -----------
hedefler = list(feature_map.keys())
gelecek  = []

# 2024 son durumdan başla
durum = df[df["yıl"] == 2024].set_index("il_adı").copy()

for yil in range(2025, 2031):
    yeni = {}
    for il in df["il_adı"].unique():
        if il not in durum.index: continue
        baz = durum.loc[il].copy()
        baz["yıl"] = yil
        if (il, yil) in nufus_geleceği:
            baz["nufus"] = nufus_geleceği[(il, yil)]
        yeni[il] = baz

    # Her hedef için tahmin yap (girdi: yeni dict + diğer hedeflerin önceki yıl değeri)
    pred_dict = {}
    for h in hedefler:
        feats = feature_map[h]
        X_row = []
        iller_sirali = []
        for il, row in yeni.items():
            try:
                X_row.append([row[f] for f in feats])
                iller_sirali.append(il)
            except KeyError:
                continue
        X_row = np.asarray(X_row, dtype=float)
        sc = scalers.get(h)
        pred = modeller[h].predict(sc.transform(X_row) if sc is not None else X_row)
        # Negatif tahminleri 0'a kıs
        pred = np.clip(pred, 0, None)
        pred_dict[h] = dict(zip(iller_sirali, pred))

    # Yeni durumu güncelle (recursive)
    for il, row in yeni.items():
        for h in hedefler:
            if il in pred_dict[h]:
                row[h] = pred_dict[h][il]
        gelecek.append({"il_adı": il, "yıl": yil,
                        **{k: row[k] for k in ["nufus"] + hedefler if k in row}})
    # Bir sonraki iterasyon için durum güncelle
    durum = pd.DataFrame(yeni).T

gelecek_df = pd.DataFrame(gelecek).sort_values(["il_adı", "yıl"]).reset_index(drop=True)
gelecek_df.to_excel("tahmin_2025_2030.xlsx", index=False)
gelecek_df.to_csv("tahmin_2025_2030.csv", index=False, encoding="utf-8-sig")
print("\n[OK] tahmin_2025_2030.xlsx ve .csv kaydedildi.")
print(gelecek_df.head(12))
