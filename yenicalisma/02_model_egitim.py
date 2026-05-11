# =====================================================================
# 02 - MODEL EĞİTİM + DEĞERLENDİRME
#
# Bu dosya Jupyter Notebook hücrelerine bölünmüştür (# %% işaretçileri).
# VS Code / Jupyter / Spyder her bir `# %%` bloğunu ayrı hücre olarak görür.
#
# Yapı:
#   1)  Veriyi yükle
#   2)  Feature map
#   3)  K-fold
#   4)  Model fabrikası (4.1, 4.2, ... — her model ayrı hücre)
#   5)  Metrik fonksiyonu
#   6)  Tek model eğitim fonksiyonu (egit_model)
#   7)  TÜM modelleri çalıştır
#   8)  ARIMA (univariate, order arama dahil)
#   9)  Prophet (opsiyonel)
#  10)  Özet tablolar
# =====================================================================

# %% [1] VERİYİ YÜKLE ================================================
import warnings, time
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

df = pd.read_csv("model_giris.csv").sort_values(["il_adı", "yıl"]).reset_index(drop=True)

# Toplam doktor/hekim değişkeni: uzman + pratisyen + asistan
df["doktor"] = df["uzman"] + df["pratisyen"] + df["asistan"]

print("Veri:", df.shape, "İl:", df["il_adı"].nunique(),
      "Yıl aralığı:", df["yıl"].min(), "-", df["yıl"].max())

# %% [2] FEATURE MAP =================================================
# Not: Asistan verisi şehir bazında çok dengesiz/sıçramalı olduğu için
# uzman + pratisyen + asistan toplamı "doktor" hedefi olarak ele alındı.
# Böylece model, toplam hekim ihtiyacını daha stabil şekilde tahmin eder.
feature_map = {
    "doktor":  ["yıl", "nufus", "hastane", "yatak", "hemsire", "ebe", "diger"],
    "ebe":     ["yıl", "nufus", "hastane", "yatak", "doktor", "hemsire", "diger"],
    "diger":   ["yıl", "nufus", "hastane", "yatak", "doktor", "hemsire", "ebe"],
    "hemsire": ["yıl", "nufus", "hastane", "yatak", "doktor", "ebe", "diger"],
    "yatak":   ["yıl", "nufus", "hastane", "doktor", "hemsire", "ebe", "diger"],
    "hastane": ["yıl", "nufus", "yatak", "doktor", "hemsire", "ebe", "diger"],
}
print("Feature map: hedefler →", list(feature_map.keys()))

# %% [3] K-FOLD ======================================================
folds = [
    (range(2002, 2017), range(2017, 2021)),    # Fold 1
    (range(2002, 2018), range(2018, 2022)),    # Fold 2
    (range(2002, 2019), range(2019, 2023)),    # Fold 3
    (range(2002, 2020), range(2020, 2024)),    # Fold 4
    (range(2002, 2021), range(2021, 2025)),    # Fold 5 — ANA FOLD
]
print(f"Toplam {len(folds)} fold")

# %% [4.0] MODEL REGISTRY KURULUMU ==================================
# Her model 4.x hücresinde aşağıdaki sözlüğe kendini kaydeder.
# Hücreyi her yeniden başlattığında bu sözlüğü sıfırla.

from sklearn.model_selection import GridSearchCV

INNER_CV = 2                          # parametre araması için iç k-fold
SCORING  = "neg_mean_absolute_error"  # iç değerlendirme metriği

models_registry = {}   # {ad: {"factory": callable, "scale": bool}}
print("Model registry sıfırlandı. Aşağıdaki 4.1, 4.2, ... hücrelerini çalıştır.")

# %% [4.1] LINEAR REGRESSION =========================================
from sklearn.linear_model import LinearRegression
def make_linear():
    grid = {"fit_intercept": [True, False]}
    return GridSearchCV(LinearRegression(), grid,
                        cv=INNER_CV, scoring=SCORING, n_jobs=-1)
models_registry["LinearRegression"] = {"factory": make_linear, "scale": False}
print(f"[4.1] LinearRegression kaydedildi. (Toplam: {len(models_registry)})")

# %% [4.2] RANDOM FOREST =============================================
from sklearn.ensemble import RandomForestRegressor
def make_rf():
    grid = {"n_estimators": [100, 200, 300],
            "max_depth":    [None, 10]}
    return GridSearchCV(RandomForestRegressor(random_state=42, n_jobs=-1), grid,
                        cv=INNER_CV, scoring=SCORING, n_jobs=-1)
models_registry["RandomForest"] = {"factory": make_rf, "scale": False}
print(f"[4.2] RandomForest kaydedildi. (Toplam: {len(models_registry)})")

# %% [4.3] EXTRA TREES ===============================================
from sklearn.ensemble import ExtraTreesRegressor
def make_et():
    grid = {"n_estimators": [100, 200, 300],
            "max_depth":    [None, 10]}
    return GridSearchCV(ExtraTreesRegressor(random_state=42, n_jobs=-1), grid,
                        cv=INNER_CV, scoring=SCORING, n_jobs=-1)
models_registry["ExtraTrees"] = {"factory": make_et, "scale": False}
print(f"[4.3] ExtraTrees kaydedildi. (Toplam: {len(models_registry)})")

# %% [4.4] GRADIENT BOOSTING =========================================
from sklearn.ensemble import GradientBoostingRegressor
def make_gb():
    grid = {"n_estimators":  [100, 200],
            "learning_rate": [0.05, 0.1],
            "max_depth":     [3, 5]}
    return GridSearchCV(GradientBoostingRegressor(random_state=42), grid,
                        cv=INNER_CV, scoring=SCORING, n_jobs=-1)
models_registry["GradientBoosting"] = {"factory": make_gb, "scale": False}
print(f"[4.4] GradientBoosting kaydedildi. (Toplam: {len(models_registry)})")

# %% [4.5] XGBOOST ===================================================
from xgboost import XGBRegressor
def make_xgb():
    grid = {"n_estimators":  [200, 300],
            "learning_rate": [0.05, 0.1],
            "max_depth":     [3, 6]}
    base = XGBRegressor(random_state=42, verbosity=0, objective="reg:squarederror")
    return GridSearchCV(base, grid, cv=INNER_CV, scoring=SCORING, n_jobs=-1)
models_registry["XGBoost"] = {"factory": make_xgb, "scale": False}
print(f"[4.5] XGBoost kaydedildi. (Toplam: {len(models_registry)})")

# %% [4.6] CATBOOST ==================================================
try:
    from catboost import CatBoostRegressor
    HAVE_CAT = True
except ImportError:
    HAVE_CAT = False
    print("[UYARI] catboost yüklü değil → pip install catboost")

if HAVE_CAT:
    def make_cat():
        grid = {"iterations": [200, 300],
                "depth":      [4, 6, 8]}
        return GridSearchCV(CatBoostRegressor(verbose=0, random_seed=42), grid,
                            cv=INNER_CV, scoring=SCORING, n_jobs=-1)
    models_registry["CatBoost"] = {"factory": make_cat, "scale": False}
    print(f"[4.6] CatBoost kaydedildi. (Toplam: {len(models_registry)})")

# %% [4.7] SVR (ölçekleme gerekli) ===================================
from sklearn.svm import SVR
def make_svr():
    grid = {"C":       [1, 10, 50],
            "epsilon": [0.05, 0.1]}
    return GridSearchCV(SVR(), grid, cv=INNER_CV, scoring=SCORING, n_jobs=-1)
models_registry["SVR"] = {"factory": make_svr, "scale": True}
print(f"[4.7] SVR kaydedildi. (Toplam: {len(models_registry)})")

# %% [4.8] KNN (ölçekleme gerekli) ===================================
from sklearn.neighbors import KNeighborsRegressor
def make_knn():
    grid = {"n_neighbors": [3, 5, 7, 10],
            "weights":     ["uniform", "distance"]}
    return GridSearchCV(KNeighborsRegressor(), grid,
                        cv=INNER_CV, scoring=SCORING, n_jobs=-1)
models_registry["KNN"] = {"factory": make_knn, "scale": True}
print(f"[4.8] KNN kaydedildi. (Toplam: {len(models_registry)})")

# %% [4.9] BAGGING ===================================================
from sklearn.ensemble import BaggingRegressor
def make_bag():
    grid = {"n_estimators": [50, 100, 200],
            "max_samples":  [0.7, 1.0]}
    return GridSearchCV(BaggingRegressor(random_state=42, n_jobs=-1), grid,
                        cv=INNER_CV, scoring=SCORING, n_jobs=-1)
models_registry["Bagging"] = {"factory": make_bag, "scale": False}
print(f"[4.9] Bagging kaydedildi. (Toplam: {len(models_registry)})")

# %% [4.10] VOTING (RF + XGB + GB) ===================================
from sklearn.ensemble import VotingRegressor
def make_voting():
    # Voting'i ayrı tunlamak yerine sabit kombinasyon (yüzeysel hızlı baseline)
    return VotingRegressor([
        ("rf",  RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)),
        ("xgb", XGBRegressor(n_estimators=200, random_state=42, verbosity=0,
                             objective="reg:squarederror")),
        ("gb",  GradientBoostingRegressor(n_estimators=200, random_state=42)),
    ])
models_registry["Voting"] = {"factory": make_voting, "scale": False}
print(f"[4.10] Voting kaydedildi. (Toplam: {len(models_registry)})")

# %% [4.11] STACKING (RF + XGB → Linear meta) ========================
from sklearn.ensemble import StackingRegressor
def make_stacking():
    return StackingRegressor(
        estimators=[
            ("rf",  RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)),
            ("xgb", XGBRegressor(n_estimators=200, random_state=42, verbosity=0,
                                 objective="reg:squarederror")),
        ],
        final_estimator=LinearRegression(),
        n_jobs=-1)
models_registry["Stacking"] = {"factory": make_stacking, "scale": False}
print(f"[4.11] Stacking kaydedildi. (Toplam: {len(models_registry)})")

# %% [4.X] REGISTRY KONTROL =========================================
print(f"\nRegistry'de {len(models_registry)} model var:")
for ad, info in models_registry.items():
    print(f"  - {ad:<20}  scale={info['scale']}")

# %% [5] METRİK FONKSİYONU ==========================================
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                             r2_score, precision_score, recall_score, f1_score)

def metrikler(y_true, y_pred):
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

print("Metrik fonksiyonu hazır.")

# %% [6] TEK MODEL EĞİTİM FONKSİYONU ================================
butun_sonuclar  = []   # tüm metrik kayıtları
butun_tahminler = []   # Fold-5 detay tahminler
en_iyi_params   = []   # her GridSearchCV'nin best_params_

def egit_model(model_ad):
    """Bir modeli TÜM hedefler ve TÜM foldlar için eğitir."""
    global butun_sonuclar, butun_tahminler, en_iyi_params
    # Aynı modeli yeniden çalıştırırsan eski kayıtları temizle
    butun_sonuclar  = [r for r in butun_sonuclar  if r.get("Model") != model_ad]
    butun_tahminler = [r for r in butun_tahminler if r.get("model") != model_ad]
    en_iyi_params   = [r for r in en_iyi_params   if r.get("Model") != model_ad]

    info = models_registry[model_ad]
    print(f"\n========== {model_ad} ==========")
    t0 = time.time()
    for hedef, feats in feature_map.items():
        cols = list(dict.fromkeys(["il_adı"] + feats + [hedef]))
        veri = df[cols].dropna().copy()
        if veri.empty:
            continue
        for f_idx, (tr_y, te_y) in enumerate(folds, 1):
            train = veri[veri["yıl"].isin(tr_y)]
            test  = veri[veri["yıl"].isin(te_y)]
            if train.empty or test.empty:
                continue
            X_tr, y_tr = train[feats].values, train[hedef].values
            X_te, y_te = test[feats].values,  test[hedef].values
            if info["scale"]:
                sc = StandardScaler().fit(X_tr)
                X_tr, X_te = sc.transform(X_tr), sc.transform(X_te)
            try:
                mdl = info["factory"]()
                mdl.fit(X_tr, y_tr)
                pred = np.clip(mdl.predict(X_te), 0, None)
                bp = getattr(mdl, "best_params_", None)
            except Exception as e:
                print(f"  [HATA] {hedef}/fold{f_idx}: {e}")
                continue
            m = metrikler(y_te, pred)
            m.update(dict(Model=model_ad, Hedef=hedef, Fold=f_idx,
                          TrainSon=max(tr_y), TestSon=max(te_y)))
            butun_sonuclar.append(m)
            if bp:
                en_iyi_params.append(dict(Model=model_ad, Hedef=hedef,
                                          Fold=f_idx, **bp))
            if f_idx == 5:
                for i, (yr, il) in enumerate(zip(test["yıl"].values, test["il_adı"].values)):
                    butun_tahminler.append(dict(il_adı=il, yıl=int(yr), hedef=hedef,
                                                model=model_ad,
                                                gercek=float(y_te[i]),
                                                tahmin=float(pred[i])))
        print(f"  {hedef:<10}  5 fold bitti.")
    print(f"[{model_ad}] toplam süre: {(time.time()-t0)/60:.2f} dk")

print("egit_model() fonksiyonu hazır. Tek tek veya hepsini çalıştırabilirsin.")
print("Örnek: egit_model('RandomForest')")

# %% [7] TÜM MODELLERİ ÇALIŞTIR ====================================
# Burayı çalıştırırsan registry'deki tüm modeller eğitilir.
# Tek tek çalıştırmak istersen: egit_model("LinearRegression") şeklinde.
t_genel = time.time()
for ad in models_registry.keys():
    egit_model(ad)
print(f"\n[BİTTİ] Tüm modeller. Süre: {(time.time()-t_genel)/60:.1f} dk")

# %% [8] ARIMA (univariate, küçük order arama) =====================
try:
    from statsmodels.tsa.arima.model import ARIMA
    HAVE_ARIMA = True
except ImportError:
    HAVE_ARIMA = False
    print("[UYARI] statsmodels yok → pip install statsmodels")

ARIMA_ORDERS = [(1,1,1), (2,1,1), (0,1,1), (1,1,2)]

def arima_best_forecast(seri, h):
    """En düşük AIC'li order ile h adım ileri tahmin."""
    best = None
    best_aic = float("inf")
    for o in ARIMA_ORDERS:
        try:
            m = ARIMA(seri, order=o).fit()
            if m.aic < best_aic:
                best_aic = m.aic
                best = m
        except Exception:
            continue
    if best is None:
        return None
    return np.clip(np.asarray(best.forecast(steps=h)), 0, None)

if HAVE_ARIMA:
    print("\n========== ARIMA (univariate) ==========")
    # önce kayıtları temizle
    butun_sonuclar  = [r for r in butun_sonuclar  if r.get("Model") != "ARIMA"]
    butun_tahminler = [r for r in butun_tahminler if r.get("model") != "ARIMA"]

    t0 = time.time()
    for hedef in feature_map.keys():
        veri = df[["il_adı", "yıl", hedef]].dropna()
        if veri.empty: continue
        for f_idx, (tr_y, te_y) in enumerate(folds, 1):
            gercekler, tahminler = [], []
            for il in veri["il_adı"].unique():
                seri = veri[veri["il_adı"] == il].sort_values("yıl")
                tr = seri[seri["yıl"].isin(tr_y)][hedef].values
                te = seri[seri["yıl"].isin(te_y)][hedef].values
                if len(tr) < 8 or len(te) == 0:
                    continue
                pred = arima_best_forecast(tr, len(te))
                if pred is None: continue
                gercekler.extend(te.tolist())
                tahminler.extend(pred.tolist())
                if f_idx == 5:
                    yrs = seri[seri["yıl"].isin(te_y)]["yıl"].values
                    for i, yr in enumerate(yrs):
                        butun_tahminler.append(dict(il_adı=il, yıl=int(yr), hedef=hedef,
                                                    model="ARIMA",
                                                    gercek=float(te[i]),
                                                    tahmin=float(pred[i])))
            if gercekler:
                m = metrikler(gercekler, tahminler)
                m.update(dict(Model="ARIMA", Hedef=hedef, Fold=f_idx,
                              TrainSon=max(tr_y), TestSon=max(te_y)))
                butun_sonuclar.append(m)
        print(f"  {hedef:<10}  bitti.")
    print(f"ARIMA süre: {(time.time()-t0)/60:.2f} dk")

# %% [9] PROPHET (opsiyonel, yavaş) ================================
PROPHET_ACTIVE = False    # True yaparsan çalışır

if PROPHET_ACTIVE:
    try:
        from prophet import Prophet
        HAVE_PROPHET = True
    except ImportError:
        HAVE_PROPHET = False
        print("[UYARI] prophet yok → pip install prophet")

    if HAVE_PROPHET:
        print("\n========== Prophet ==========")
        butun_sonuclar  = [r for r in butun_sonuclar  if r.get("Model") != "Prophet"]
        butun_tahminler = [r for r in butun_tahminler if r.get("model") != "Prophet"]
        t0 = time.time()
        for hedef in feature_map.keys():
            veri = df[["il_adı", "yıl", hedef]].dropna()
            for f_idx, (tr_y, te_y) in enumerate(folds, 1):
                gercekler, tahminler = [], []
                for il in veri["il_adı"].unique():
                    seri = veri[veri["il_adı"] == il].sort_values("yıl")
                    tr = seri[seri["yıl"].isin(tr_y)]
                    te = seri[seri["yıl"].isin(te_y)]
                    if len(tr) < 8 or te.empty: continue
                    try:
                        dp = pd.DataFrame({
                            "ds": pd.to_datetime(tr["yıl"].astype(str)+"-01-01"),
                            "y":  tr[hedef].values})
                        m = Prophet(yearly_seasonality=False, daily_seasonality=False,
                                    weekly_seasonality=False)
                        m.fit(dp)
                        future = pd.DataFrame({
                            "ds": pd.to_datetime(te["yıl"].astype(str)+"-01-01")})
                        pred = np.clip(m.predict(future)["yhat"].values, 0, None)
                    except Exception:
                        continue
                    gercekler.extend(te[hedef].values.tolist())
                    tahminler.extend(pred.tolist())
                if gercekler:
                    met = metrikler(gercekler, tahminler)
                    met.update(dict(Model="Prophet", Hedef=hedef, Fold=f_idx,
                                    TrainSon=max(tr_y), TestSon=max(te_y)))
                    butun_sonuclar.append(met)
            print(f"  {hedef:<10}  bitti.")
        print(f"Prophet süre: {(time.time()-t0)/60:.2f} dk")

# %% [10] ÖZET TABLOLAR ============================================
sonuc_df = pd.DataFrame(butun_sonuclar)
sonuc_df.to_csv("model_metrikleri_detay.csv", index=False, encoding="utf-8-sig")

fold_ort = (sonuc_df.groupby(["Model", "Hedef", "Fold"])
            [["Accuracy", "MAE", "RMSE", "R2", "ErrorRate", "Precision", "Recall", "F1"]]
            .mean().reset_index())
fold_ort.to_csv("model_metrikleri_fold.csv", index=False, encoding="utf-8-sig")

genel = (sonuc_df.groupby("Model")
         [["Accuracy", "MAE", "RMSE", "R2", "ErrorRate", "Precision", "Recall", "F1"]]
         .mean().round(3).reset_index()
         .sort_values("Accuracy", ascending=False))
genel.to_csv("model_metrikleri_genel.csv", index=False, encoding="utf-8-sig")
print("\n========== GENEL MODEL KARŞILAŞTIRMASI ==========")
print(genel.to_string(index=False))

ana = sonuc_df[sonuc_df["Fold"] == 5]
en_iyiler = (ana.sort_values("Accuracy", ascending=False)
             .groupby("Hedef").head(1)
             [["Hedef", "Model", "Accuracy", "MAE", "RMSE", "R2"]])
en_iyiler.to_csv("en_iyi_modeller.csv", index=False, encoding="utf-8-sig")
print("\n========== HER HEDEF İÇİN EN İYİ MODEL (Fold 5) ==========")
print(en_iyiler.to_string(index=False))

pd.DataFrame(butun_tahminler).to_csv("tahminler_fold5_2021_2024.csv",
                                     index=False, encoding="utf-8-sig")

if en_iyi_params:
    pd.DataFrame(en_iyi_params).to_csv("en_iyi_hiperparametreler.csv",
                                       index=False, encoding="utf-8-sig")
    print("\n[OK] en_iyi_hiperparametreler.csv kaydedildi.")

print("\n[OK] Çıktılar:")
print("  - model_metrikleri_detay.csv")
print("  - model_metrikleri_fold.csv")
print("  - model_metrikleri_genel.csv")
print("  - en_iyi_modeller.csv")
print("  - en_iyi_hiperparametreler.csv")
print("  - tahminler_fold5_2021_2024.csv")