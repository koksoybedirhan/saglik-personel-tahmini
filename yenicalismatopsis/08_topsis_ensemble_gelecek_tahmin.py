# =====================================================================
# 08 - TOPSIS HEDEFLERİ İÇİN ENSEMBLE GELECEK TAHMİNİ
#
# 1) topsis_metrikleri_detay.csv'den her hedef için Fold-5 top-3 modeli al
# 2) Her modeli FULL data ile yeniden eğit
# 3) Recursive olarak gelecek yılları tahmin et:
#       - yas_0_14, yas_65_plus   → 2026-2030 (2025 zaten gerçek)
#       - hekim_oran, yatak_oran  → 2025-2030
# 4) Top-3 ağırlıklı ensemble birleştir
#
# Çıktılar:
#   - topsis_gelecek_tahmin.xlsx + .csv  (TOPSIS analizinde kullanılacak)
#   - topsis_ensemble_secimleri.csv
# =====================================================================

# %% [1] KÜTÜPHANELER ===============================================
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

from sklearn.linear_model     import LinearRegression, Ridge, ElasticNet
from sklearn.ensemble         import (RandomForestRegressor, ExtraTreesRegressor,
                                      GradientBoostingRegressor, BaggingRegressor,
                                      VotingRegressor, StackingRegressor,
                                      HistGradientBoostingRegressor)
from sklearn.svm              import SVR
from sklearn.neighbors        import KNeighborsRegressor
from sklearn.preprocessing    import StandardScaler
from xgboost import XGBRegressor
try:
    from catboost import CatBoostRegressor
    HAVE_CAT = True
except ImportError:
    HAVE_CAT = False

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model  import ARIMA
try:
    from prophet import Prophet
    HAVE_PROPHET = True
except ImportError:
    HAVE_PROPHET = False

TOP_K = 3

# %% [2] VERİ + ÖNCEKİ ÇIKTILAR + FEATURE ENGINEERING ===============
import os
df = pd.read_csv("topsis_veri.csv").sort_values(["il_adı", "yıl"]).reset_index(drop=True)
hedefler_meta = pd.read_csv("topsis_hedef_yillar.csv")
metr = pd.read_csv("topsis_metrikleri_detay.csv")

HEDEFLER = ["yas_0_14", "yas_65_plus", "hekim_oran", "yatak_oran"]

# Aynı feature engineering — 07 ile birebir aynı olmalı
for yol in ["../yenicalisma/model_giris.csv", "model_giris.csv", "../model_giris.csv"]:
    if os.path.exists(yol):
        nf = pd.read_csv(yol)[["il_adı", "yıl", "nufus"]]
        df = df.merge(nf, on=["il_adı", "yıl"], how="left")
        break
else:
    df["nufus"] = np.nan

il_id_map = {il: i for i, il in enumerate(sorted(df["il_adı"].unique()))}
df["il_id"] = df["il_adı"].map(il_id_map)

df = df.sort_values(["il_adı", "yıl"]).reset_index(drop=True)
for h in HEDEFLER:
    df[f"{h}_lag1"] = df.groupby("il_adı")[h].shift(1)
    df[f"{h}_lag2"] = df.groupby("il_adı")[h].shift(2)

FEATURE_MAP = {
    "yas_0_14":    ["yıl", "il_id", "nufus",
                    "yas_0_14_lag1", "yas_0_14_lag2",
                    "yas_65_plus_lag1", "hekim_oran_lag1", "yatak_oran_lag1"],
    "yas_65_plus": ["yıl", "il_id", "nufus",
                    "yas_65_plus_lag1", "yas_65_plus_lag2",
                    "yas_0_14_lag1", "hekim_oran_lag1", "yatak_oran_lag1"],
    "hekim_oran":  ["yıl", "il_id", "nufus",
                    "hekim_oran_lag1", "hekim_oran_lag2",
                    "yas_0_14_lag1", "yas_65_plus_lag1", "yatak_oran_lag1"],
    "yatak_oran":  ["yıl", "il_id", "nufus",
                    "yatak_oran_lag1", "yatak_oran_lag2",
                    "yas_0_14_lag1", "yas_65_plus_lag1", "hekim_oran_lag1"],
}

# %% [3] TOP-K SEÇİMİ VE AĞIRLIKLAR ================================
ana = metr[metr["Fold"] == 5]
secim_satirlari = []
agirlik_map = {}
for hedef in HEDEFLER:
    bu = ana[ana["Hedef"] == hedef].sort_values("Accuracy", ascending=False).head(TOP_K)
    if bu.empty: continue
    w = bu["Accuracy"].values
    w = w / w.sum()
    bu = bu[["Hedef","Model","Accuracy","MAE","RMSE"]].copy()
    bu["Agirlik"] = w
    secim_satirlari.append(bu)
    agirlik_map[hedef] = dict(zip(bu["Model"].values, w))

secimler = pd.concat(secim_satirlari, ignore_index=True)
secimler.to_csv("topsis_ensemble_secimleri.csv", index=False, encoding="utf-8-sig")
print("Ensemble seçimleri:")
print(secimler.to_string(index=False))

# %% [4] MODEL FABRİKASI (full-data eğitim için) ===================
def make_ml(ad):
    if ad == "LinearRegression":     return LinearRegression(), False
    if ad == "Ridge":                return Ridge(alpha=1.0, random_state=42), False
    if ad == "ElasticNet":           return ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000, random_state=42), False
    if ad == "RandomForest":         return RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1), False
    if ad == "ExtraTrees":           return ExtraTreesRegressor(n_estimators=300, random_state=42, n_jobs=-1), False
    if ad == "GradientBoosting":     return GradientBoostingRegressor(n_estimators=300, random_state=42), False
    if ad == "HistGradientBoosting": return HistGradientBoostingRegressor(random_state=42), False
    if ad == "XGBoost":              return XGBRegressor(n_estimators=400, random_state=42, verbosity=0, objective="reg:squarederror"), False
    if ad == "CatBoost" and HAVE_CAT:return CatBoostRegressor(iterations=400, depth=6, verbose=0, random_seed=42), False
    if ad == "SVR":                  return SVR(C=10, epsilon=0.1), True
    if ad == "KNN":                  return KNeighborsRegressor(n_neighbors=5), True
    if ad == "Bagging":              return BaggingRegressor(n_estimators=100, random_state=42, n_jobs=-1), False
    if ad == "Voting":
        return VotingRegressor([
            ("rf",  RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)),
            ("xgb", XGBRegressor(n_estimators=200, random_state=42, verbosity=0, objective="reg:squarederror")),
            ("gb",  GradientBoostingRegressor(n_estimators=200, random_state=42))]), False
    if ad == "Stacking":
        return StackingRegressor(
            estimators=[
                ("rf",  RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)),
                ("xgb", XGBRegressor(n_estimators=200, random_state=42, verbosity=0, objective="reg:squarederror"))],
            final_estimator=LinearRegression(), n_jobs=-1), False
    return None, False

UNI = {"HoltWinters", "ARIMA", "Prophet"}

def uni_forecast(ad, yrs, vals, h):
    s = pd.Series(vals).astype(float)
    if len(s) < 4: return np.repeat(s.iloc[-1], h)
    try:
        if ad == "HoltWinters":
            m = ExponentialSmoothing(s, trend="add", seasonal=None).fit(optimized=True)
            return np.clip(m.forecast(steps=h).values, 0, None)
        if ad == "ARIMA":
            best, ba = None, np.inf
            for o in [(1,1,1),(2,1,1),(0,1,1),(1,1,2)]:
                try:
                    m = ARIMA(s.values, order=o).fit()
                    if m.aic < ba: best, ba = m, m.aic
                except: pass
            if best is None: return np.repeat(s.iloc[-1], h)
            return np.clip(np.asarray(best.forecast(steps=h)), 0, None)
        if ad == "Prophet" and HAVE_PROPHET:
            dp = pd.DataFrame({"ds": pd.to_datetime([f"{y}-01-01" for y in yrs]),
                               "y": s.values})
            m = Prophet(yearly_seasonality=False, daily_seasonality=False, weekly_seasonality=False)
            m.fit(dp)
            son = max(yrs)
            future = pd.DataFrame({"ds": pd.to_datetime([f"{son+i}-01-01" for i in range(1, h+1)])})
            return np.clip(m.predict(future)["yhat"].values, 0, None)
    except Exception:
        pass
    return np.repeat(s.iloc[-1], h)

# %% [5] HEDEF BAŞINA TAHMİN YILLARI ===============================
TAHMIN_YIL = {}
for h in HEDEFLER:
    ys = int(hedefler_meta[hedefler_meta["hedef"] == h]["max_yil"].iloc[0])
    TAHMIN_YIL[h] = list(range(ys + 1, 2031))  # max_yil+1 → 2030
    print(f"  {h:<15} tahmin yılları: {TAHMIN_YIL[h]}")

# %% [6] HER HEDEF × HER MODEL İÇİN GELECEK TAHMİNİ ================
# Önce univariate'leri hesapla — diğer hedefler için feature olarak gerekirler
tahmin_db = {h: {} for h in HEDEFLER}

print("\n========== Univariate tahminler ==========")
for hedef in HEDEFLER:
    yrs_tahmin = TAHMIN_YIL[hedef]
    if not yrs_tahmin: continue
    h_n = len(yrs_tahmin)
    for model_ad in agirlik_map.get(hedef, {}):
        if model_ad not in UNI: continue
        kayit = []
        for il in df["il_adı"].dropna().unique():
            seri = df[(df["il_adı"] == il)].sort_values("yıl").dropna(subset=[hedef])
            if seri.empty: continue
            yrs = seri["yıl"].values
            vals = seri[hedef].values
            pred = uni_forecast(model_ad, yrs, vals, h_n)
            for i, yr in enumerate(yrs_tahmin):
                kayit.append({"il_adı": il, "yıl": yr, "tahmin": float(pred[i])})
        tahmin_db[hedef][model_ad] = pd.DataFrame(kayit)
        print(f"  {hedef:<15} {model_ad:<18} bitti")

# %% [7] MULTIVARIATE FEATURE BAZI ÜRET ============================
# Her tahmin yılı için feature olarak diğer hedeflerin ortalama univariate
# tahminini kullan. Bunu için her hedefin univariate ort. tahminini sözlüğe koy.
def feature_baz_olustur():
    """Tüm 4 hedef için (il, yıl) → ortalama univariate tahmin dict döndürür."""
    db = {}
    for hedef in HEDEFLER:
        univ = [m for m in tahmin_db[hedef] if m in UNI]
        if not univ: continue
        cat = pd.concat([tahmin_db[hedef][m] for m in univ])
        ort = cat.groupby(["il_adı", "yıl"])["tahmin"].mean().reset_index()
        for _, r in ort.iterrows():
            db[(hedef, r["il_adı"], int(r["yıl"]))] = float(r["tahmin"])
    return db

baz_dict = feature_baz_olustur()

# Genişletilmiş tablo oluştur — geçmiş + univariate ortalaması ile doldurulmuş gelecek
df_genis = df.copy()
tum_iller = df["il_adı"].dropna().unique()
tum_yillar = sorted(set([y for h in HEDEFLER for y in TAHMIN_YIL[h]]))
for yr in tum_yillar:
    for il in tum_iller:
        idx = df_genis.index[(df_genis["il_adı"] == il) & (df_genis["yıl"] == yr)]
        if not len(idx):
            yeni = {"il_adı": il, "yıl": yr}
            for h in HEDEFLER:
                yeni[h] = baz_dict.get((h, il, yr), np.nan)
            df_genis = pd.concat([df_genis, pd.DataFrame([yeni])], ignore_index=True)
        else:
            for h in HEDEFLER:
                df_genis.loc[idx[0], h] = baz_dict.get((h, il, yr), np.nan)
df_genis = df_genis.sort_values(["il_adı", "yıl"]).reset_index(drop=True)

# Önemli: lag özelliklerini ve il_id'yi df_genis'te YENİDEN hesapla
df_genis["il_id"] = df_genis["il_adı"].map(il_id_map)
if "nufus" not in df_genis.columns:
    df_genis["nufus"] = np.nan
# Nüfusu lineer extrapolate et (gelecek yıllar için)
for il in tum_iller:
    mask_g = (df_genis["il_adı"] == il) & (df_genis["nufus"].isna())
    mask_var = (df_genis["il_adı"] == il) & (df_genis["nufus"].notna())
    if mask_var.sum() < 3 or mask_g.sum() == 0: continue
    s = df_genis[mask_var].tail(10)
    coef = np.polyfit(s["yıl"], s["nufus"], 1)
    yillar_eksik = df_genis.loc[mask_g, "yıl"].values
    df_genis.loc[mask_g, "nufus"] = np.polyval(coef, yillar_eksik)

# Lag özelliklerini yeniden hesapla — geçmiş + univariate-doldurulmuş gelecek üzerinden
for h in HEDEFLER:
    df_genis[f"{h}_lag1"] = df_genis.groupby("il_adı")[h].shift(1)
    df_genis[f"{h}_lag2"] = df_genis.groupby("il_adı")[h].shift(2)

# %% [8] MULTIVARIATE MODELLERİ EĞİT VE TAHMİN ÜRET ================
print("\n========== Multivariate tahminler ==========")
for hedef in HEDEFLER:
    yrs_tahmin = TAHMIN_YIL[hedef]
    if not yrs_tahmin: continue
    feats = FEATURE_MAP[hedef]
    for model_ad in agirlik_map.get(hedef, {}):
        if model_ad in UNI: continue
        mdl, scale = make_ml(model_ad)
        if mdl is None: continue
        # Eğitim verisi: tarihsel hedef tüm satırlar (NaN olmayanlar)
        train = df_genis[df_genis[hedef].notna() & df_genis["yıl"].between(
            int(hedefler_meta[hedefler_meta["hedef"]==hedef]["min_yil"].iloc[0]),
            int(hedefler_meta[hedefler_meta["hedef"]==hedef]["max_yil"].iloc[0])
        )][feats + [hedef]].dropna()
        if train.empty: continue
        X_tr, y_tr = train[feats].values, train[hedef].values
        sc = StandardScaler().fit(X_tr) if scale else None
        mdl.fit(sc.transform(X_tr) if sc is not None else X_tr, y_tr)

        kayit = []
        for yr in yrs_tahmin:
            sat = df_genis[df_genis["yıl"] == yr]
            mask = ~sat[feats].isna().any(axis=1)
            kul = sat[mask]
            if kul.empty: continue
            X = kul[feats].values
            pred = mdl.predict(sc.transform(X) if sc is not None else X)
            pred = np.clip(pred, 0, None)
            for il, p in zip(kul["il_adı"].values, pred):
                kayit.append({"il_adı": il, "yıl": yr, "tahmin": float(p)})
        tahmin_db[hedef][model_ad] = pd.DataFrame(kayit)
        print(f"  {hedef:<15} {model_ad:<22} bitti")

# %% [9] AĞIRLIKLI ENSEMBLE BİRLEŞTİRME ============================
print("\n========== Ağırlıklı ensemble ==========")
sonuc_satirlari = []
for hedef in HEDEFLER:
    if hedef not in agirlik_map: continue
    pivolar, weights = [], []
    for m, w in agirlik_map[hedef].items():
        if m not in tahmin_db[hedef] or tahmin_db[hedef][m].empty: continue
        s = tahmin_db[hedef][m].set_index(["il_adı","yıl"])["tahmin"].rename(m)
        pivolar.append(s); weights.append(w)
    if not pivolar: continue
    piv = pd.concat(pivolar, axis=1)
    w = np.array(weights)
    mask = ~piv.isna().values
    am = np.tile(w, (len(piv), 1)) * mask
    sat_top = am.sum(axis=1, keepdims=True); sat_top[sat_top == 0] = 1
    am = am / sat_top
    ens = np.nansum(piv.values * am, axis=1)
    for (il, yr), p in zip(piv.index, ens):
        sonuc_satirlari.append({"il_adı": il, "yıl": int(yr), "hedef": hedef,
                                "tahmin": float(p)})

sonuc_long = pd.DataFrame(sonuc_satirlari)

# 2025 yaş verisi gerçek ise tahminin yerine koy (sadece yaş hedefleri için)
gercek_2025 = df[df["yıl"] == 2025][["il_adı", "yıl"] +
                                     [h for h in ["yas_0_14","yas_65_plus"] if h in df.columns]]
if not gercek_2025.empty:
    for _, r in gercek_2025.iterrows():
        for hedef in ["yas_0_14", "yas_65_plus"]:
            if hedef not in r or pd.isna(r[hedef]): continue
            idx = sonuc_long.index[(sonuc_long["il_adı"]==r["il_adı"]) &
                                   (sonuc_long["yıl"]==2025) &
                                   (sonuc_long["hedef"]==hedef)]
            if len(idx):
                sonuc_long.loc[idx[0], "tahmin"] = float(r[hedef])

# Wide formata getir
sonuc_wide = sonuc_long.pivot_table(index=["il_adı", "yıl"], columns="hedef",
                                     values="tahmin").reset_index()
sonuc_wide = sonuc_wide[["il_adı", "yıl"] + [h for h in HEDEFLER if h in sonuc_wide.columns]]

# %% [10] ÇIKTI ====================================================
sonuc_wide.to_csv("topsis_gelecek_tahmin.csv", index=False, encoding="utf-8-sig")
sonuc_wide.to_excel("topsis_gelecek_tahmin.xlsx", index=False)
print(f"\n[OK] topsis_gelecek_tahmin.xlsx ({len(sonuc_wide)} satır)")
print(sonuc_wide.head(15).to_string(index=False))
