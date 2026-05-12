# =====================================================================
# 04 - TEKİL MODELLERLE 2025-2030 GELECEK TAHMİNİ
#
# Her hedef için en_iyi_modeller.csv'deki yöntem AYNEN kullanılır.
# (ARIMA dahil — override yok.)
#
# Strateji:
#   1) Univariate model olan hedefler (HoltWinters/Prophet/ARIMA) için
#      her il × hedef ayrı eğitilir ve 2025-2030 doğrudan tahmin edilir.
#   2) Multivariate model olan hedefler (sklearn) için feature_map kullanılır;
#      diğer hedeflerin tahminleri girdi olur (yani önce univariateler bitmeli).
#
# İsteğe bağlı override:
#   Eğer ileride bir hedefin modelini değiştirmek istersen MODEL_OVERRIDE
#   sözlüğüne ekle (default boş).
#
# Çıktı: tahmin_2025_2030_tekil.xlsx + .csv
# =====================================================================

# %% [1] KÜTÜPHANELER ===============================================
import warnings, time
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

from sklearn.linear_model      import LinearRegression
from sklearn.ensemble          import (RandomForestRegressor, ExtraTreesRegressor,
                                       GradientBoostingRegressor, BaggingRegressor,
                                       VotingRegressor, StackingRegressor,
                                       HistGradientBoostingRegressor)
from sklearn.svm               import SVR
from sklearn.neighbors         import KNeighborsRegressor
from sklearn.preprocessing     import StandardScaler
from xgboost                   import XGBRegressor

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

# %% [2] VERİ + EN İYİ MODELLER ====================================
baz = pd.read_csv("gelecek_baz.csv").sort_values(["il_adı", "yıl"]).reset_index(drop=True)
en_iyi = pd.read_csv("en_iyi_modeller.csv")
print("Baseline:", baz.shape)
print("\nEn iyi modeller (orijinal):")
print(en_iyi.to_string(index=False))

# --- Model seçimi (override default'ta boş — ARIMA dahil her şey kalıyor) ---
MODEL_OVERRIDE = {
    # Örnek: "hastane": "HoltWinters",   # istersen aç
}
secim = {}
for _, r in en_iyi.iterrows():
    h = r["Hedef"]
    secim[h] = MODEL_OVERRIDE.get(h, r["Model"])
print("\nKullanılacak modeller:")
for h, m in secim.items():
    print(f"  {h:<10}  → {m}")

HEDEFLER = ["doktor", "ebe", "diger", "hemsire", "yatak", "hastane"]
FEATURE_MAP = {
    "doktor":  ["yıl", "nufus", "hastane", "yatak", "hemsire", "ebe", "diger"],
    "ebe":     ["yıl", "nufus", "hastane", "yatak", "doktor", "hemsire", "diger"],
    "diger":   ["yıl", "nufus", "hastane", "yatak", "doktor", "hemsire", "ebe"],
    "hemsire": ["yıl", "nufus", "hastane", "yatak", "doktor", "ebe", "diger"],
    "yatak":   ["yıl", "nufus", "hastane", "doktor", "hemsire", "ebe", "diger"],
    "hastane": ["yıl", "nufus", "yatak",   "doktor", "hemsire", "ebe", "diger"],
}

# %% [3] UNIVARIATE TAHMİN FONKSİYONLARI ============================
def fc_holtwinters(seri, h):
    seri = pd.Series(seri).astype(float)
    if len(seri) < 4 or seri.std() == 0:
        return np.repeat(seri.iloc[-1], h)
    try:
        m = ExponentialSmoothing(seri, trend="add", seasonal=None).fit(optimized=True)
        return np.clip(m.forecast(steps=h).values, 0, None)
    except Exception:
        return np.repeat(seri.iloc[-1], h)

def fc_arima(seri, h):
    seri = np.asarray(seri, dtype=float)
    if len(seri) < 6:
        return np.repeat(seri[-1], h)
    best, best_aic = None, np.inf
    for order in [(1,1,1), (2,1,1), (0,1,1), (1,1,2)]:
        try:
            m = ARIMA(seri, order=order).fit()
            if m.aic < best_aic:
                best, best_aic = m, m.aic
        except Exception:
            continue
    if best is None:
        return np.repeat(seri[-1], h)
    return np.clip(np.asarray(best.forecast(steps=h)), 0, None)

def fc_prophet(seri_yil, seri_y, h):
    if not HAVE_PROPHET or len(seri_y) < 4:
        return np.repeat(seri_y[-1], h)
    try:
        dp = pd.DataFrame({"ds": pd.to_datetime([f"{y}-01-01" for y in seri_yil]),
                           "y": seri_y})
        m = Prophet(yearly_seasonality=False, daily_seasonality=False,
                    weekly_seasonality=False)
        m.fit(dp)
        son = max(seri_yil)
        future = pd.DataFrame({"ds": pd.to_datetime([f"{son+i}-01-01" for i in range(1, h+1)])})
        return np.clip(m.predict(future)["yhat"].values, 0, None)
    except Exception:
        return np.repeat(seri_y[-1], h)

def univariate_forecast(model_ad, seri_yil, seri_y, h=6):
    if model_ad == "HoltWinters": return fc_holtwinters(seri_y, h)
    if model_ad == "ARIMA":       return fc_arima(seri_y, h)
    if model_ad == "Prophet":     return fc_prophet(list(seri_yil), list(seri_y), h)
    raise ValueError(f"Bilinmeyen univariate model: {model_ad}")

# %% [4] MULTIVARIATE MODEL FABRİKASI ===============================
def make_multivariate(model_ad):
    if model_ad == "LinearRegression":
        return LinearRegression(), False
    if model_ad == "RandomForest":
        return RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1), False
    if model_ad == "ExtraTrees":
        return ExtraTreesRegressor(n_estimators=300, random_state=42, n_jobs=-1), False
    if model_ad == "GradientBoosting":
        return GradientBoostingRegressor(n_estimators=300, random_state=42), False
    if model_ad == "HistGradientBoosting":
        return HistGradientBoostingRegressor(random_state=42), False
    if model_ad == "XGBoost":
        return XGBRegressor(n_estimators=400, random_state=42, verbosity=0,
                            objective="reg:squarederror"), False
    if model_ad == "CatBoost" and HAVE_CAT:
        return CatBoostRegressor(iterations=400, depth=6, verbose=0, random_seed=42), False
    if model_ad == "SVR":  return SVR(C=10, epsilon=0.1), True
    if model_ad == "KNN":  return KNeighborsRegressor(n_neighbors=5), True
    if model_ad == "Bagging":
        return BaggingRegressor(n_estimators=100, random_state=42, n_jobs=-1), False
    if model_ad == "Voting":
        return VotingRegressor([
            ("rf",  RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)),
            ("xgb", XGBRegressor(n_estimators=200, random_state=42, verbosity=0,
                                 objective="reg:squarederror")),
            ("gb",  GradientBoostingRegressor(n_estimators=200, random_state=42)),
        ]), False
    raise ValueError(f"Bilinmeyen multivariate model: {model_ad}")

# %% [5] HANGİ HEDEF UNIVARIATE / MULTIVARIATE ======================
UNIVARIATE_LIST = {"HoltWinters", "ARIMA", "Prophet"}
univariate_hedefler  = [h for h in HEDEFLER if secim[h] in UNIVARIATE_LIST]
multivariate_hedefler = [h for h in HEDEFLER if secim[h] not in UNIVARIATE_LIST]
print(f"Univariate hedefler ({len(univariate_hedefler)}):   {univariate_hedefler}")
print(f"Multivariate hedefler ({len(multivariate_hedefler)}): {multivariate_hedefler}")

# %% [6] UNIVARIATE TAHMİNLERİ HESAPLA ==============================
print("\n========== Univariate tahminler 2025-2030 ==========")
t0 = time.time()
for il in baz["il_adı"].unique():
    seri_il = baz[(baz["il_adı"] == il) & (baz["yıl"] <= 2024)].sort_values("yıl")
    for h in univariate_hedefler:
        s = seri_il[h].dropna()
        if s.empty: continue
        yrs = seri_il.loc[s.index, "yıl"].values
        pred = univariate_forecast(secim[h], yrs, s.values, h=6)
        for i, yr in enumerate(range(2025, 2031)):
            idx = baz.index[(baz["il_adı"] == il) & (baz["yıl"] == yr)]
            if len(idx):
                baz.loc[idx[0], h] = float(pred[i])
print(f"Univariate süre: {(time.time()-t0)/60:.1f} dk")

# %% [7] MULTIVARIATE MODELLERİ EĞİT VE TAHMİN ÜRET =================
# Multivariate hedefler için tüm geçmiş veriyle (2002-2024) eğit,
# 2025-2030 için feature olarak univariate tahminlerini + nüfusu kullan.
print("\n========== Multivariate tahminler 2025-2030 ==========")
for hedef in multivariate_hedefler:
    feats = FEATURE_MAP[hedef]
    model_ad = secim[hedef]
    train = baz[(baz["yıl"] <= 2024)][["il_adı", "yıl"] + feats + [hedef]].dropna()
    X_tr = train[feats].values
    y_tr = train[hedef].values
    mdl, scale = make_multivariate(model_ad)
    sc = StandardScaler().fit(X_tr) if scale else None
    mdl.fit(sc.transform(X_tr) if sc is not None else X_tr, y_tr)

    # Tahmin
    for yr in range(2025, 2031):
        satir = baz[baz["yıl"] == yr].copy()
        eksik = satir[feats].isna().any(axis=1)
        kullan = satir[~eksik]
        if kullan.empty: continue
        X = kullan[feats].values
        pred = mdl.predict(sc.transform(X) if sc is not None else X)
        pred = np.clip(pred, 0, None)
        baz.loc[kullan.index, hedef] = pred
    print(f"  {hedef:<10} ({model_ad}) bitti")

# %% [8] HASTANE/YATAK İÇİN TAM SAYIYA YUVARLAMA ====================
# Hastane sayısı tam sayı olmalı (gerçekçi raporlama için)
for col in ["hastane"]:
    baz.loc[baz["yıl"] >= 2025, col] = baz.loc[baz["yıl"] >= 2025, col].round().astype("Int64")

# %% [9] ÇIKTI =====================================================
sonuc = (baz[baz["yıl"] >= 2025]
         [["il_adı", "yıl", "nufus"] + HEDEFLER]
         .sort_values(["il_adı", "yıl"]).reset_index(drop=True))

sonuc.to_csv("tahmin_2025_2030_tekil.csv", index=False, encoding="utf-8-sig")
sonuc.to_excel("tahmin_2025_2030_tekil.xlsx", index=False)
print(f"\n[OK] tahmin_2025_2030_tekil.xlsx ({len(sonuc)} satır = 81 il × 6 yıl)")
print(sonuc.head(12).to_string(index=False))
