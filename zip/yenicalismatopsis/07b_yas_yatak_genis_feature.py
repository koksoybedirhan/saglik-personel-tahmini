# =====================================================================
# 07b - YAS_0_14 ve YATAK_ORAN için ek feature'larla yeniden test
#
# 07'deki tüm özelliklere + all_data24.csv'deki sağlık altyapı verilerinin
# lag özelliklerini ekler. Sadece 2 hedef üzerinde çalışır.
#
# Çıktı:
#   - topsis_metrikleri_yas_yatak_genis.csv
#   - topsis_en_iyi_yas_yatak_genis.csv
# =====================================================================

# %% [1] KÜTÜPHANELER ===============================================
import warnings, time, os
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
from sklearn.metrics          import (mean_absolute_error, mean_squared_error,
                                      r2_score, precision_score, recall_score, f1_score)
from sklearn.model_selection  import GridSearchCV

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

# %% [2] VERİ + GENİŞ FEATURE ENGINEERING ===========================
df = pd.read_csv("topsis_veri.csv").sort_values(["il_adı", "yıl"]).reset_index(drop=True)
hedefler_meta = pd.read_csv("topsis_hedef_yillar.csv")

# Sadece bu iki hedef üzerinde çalış
HEDEFLER = ["yas_0_14", "yatak_oran"]

# (a) Nüfus + sağlık altyapı (all_data24/model_giris)
ek = None
for yol in ["../yenicalisma/model_giris.csv", "model_giris.csv", "../model_giris.csv"]:
    if os.path.exists(yol):
        ek = pd.read_csv(yol)
        break
if ek is None:
    raise FileNotFoundError("model_giris.csv bulunamadı")

ek["doktor"] = ek["uzman"] + ek["pratisyen"] + ek["asistan"]
ek = ek[["il_adı", "yıl", "nufus", "hastane", "yatak", "doktor",
         "hemsire", "ebe", "diger"]]
df = df.merge(ek, on=["il_adı", "yıl"], how="left")
print("Veri (sağlık altyapı eklendi):", df.shape)

# (b) il_id + bolge_id
il_id_map = {il: i for i, il in enumerate(sorted(df["il_adı"].unique()))}
df["il_id"] = df["il_adı"].map(il_id_map)
bolge_map = {
    'İstanbul':'Marmara','Edirne':'Marmara','Kırklareli':'Marmara','Tekirdağ':'Marmara',
    'Kocaeli':'Marmara','Sakarya':'Marmara','Yalova':'Marmara','Bilecik':'Marmara',
    'Balıkesir':'Marmara','Çanakkale':'Marmara','Bursa':'Marmara',
    'İzmir':'Ege','Aydın':'Ege','Muğla':'Ege','Manisa':'Ege','Afyonkarahisar':'Ege',
    'Uşak':'Ege','Kütahya':'Ege','Denizli':'Ege',
    'Antalya':'Akdeniz','Mersin':'Akdeniz','Adana':'Akdeniz','Hatay':'Akdeniz',
    'Kahramanmaraş':'Akdeniz','Osmaniye':'Akdeniz','Isparta':'Akdeniz','Burdur':'Akdeniz',
    'Ankara':'İç Anadolu','Konya':'İç Anadolu','Kayseri':'İç Anadolu','Eskişehir':'İç Anadolu',
    'Nevşehir':'İç Anadolu','Aksaray':'İç Anadolu','Kırıkkale':'İç Anadolu','Niğde':'İç Anadolu',
    'Karaman':'İç Anadolu','Sivas':'İç Anadolu','Yozgat':'İç Anadolu','Kırşehir':'İç Anadolu',
    'Çankırı':'İç Anadolu',
    'Samsun':'Karadeniz','Trabzon':'Karadeniz','Ordu':'Karadeniz','Giresun':'Karadeniz',
    'Rize':'Karadeniz','Artvin':'Karadeniz','Tokat':'Karadeniz','Amasya':'Karadeniz',
    'Bartın':'Karadeniz','Zonguldak':'Karadeniz','Karabük':'Karadeniz','Sinop':'Karadeniz',
    'Çorum':'Karadeniz','Bayburt':'Karadeniz','Gümüşhane':'Karadeniz','Kastamonu':'Karadeniz',
    'Bolu':'Karadeniz','Düzce':'Karadeniz',
    'Erzurum':'Doğu Anadolu','Van':'Doğu Anadolu','Malatya':'Doğu Anadolu','Elazığ':'Doğu Anadolu',
    'Kars':'Doğu Anadolu','Ağrı':'Doğu Anadolu','Bitlis':'Doğu Anadolu','Iğdır':'Doğu Anadolu',
    'Tunceli':'Doğu Anadolu','Ardahan':'Doğu Anadolu','Erzincan':'Doğu Anadolu','Bingöl':'Doğu Anadolu',
    'Muş':'Doğu Anadolu','Hakkari':'Doğu Anadolu',
    'Diyarbakır':'Güneydoğu Anadolu','Gaziantep':'Güneydoğu Anadolu','Şanlıurfa':'Güneydoğu Anadolu',
    'Mardin':'Güneydoğu Anadolu','Batman':'Güneydoğu Anadolu','Siirt':'Güneydoğu Anadolu',
    'Şırnak':'Güneydoğu Anadolu','Adıyaman':'Güneydoğu Anadolu','Kilis':'Güneydoğu Anadolu'
}
df["bolge_id"] = df["il_adı"].map(bolge_map).astype("category").cat.codes

df = df.sort_values(["il_adı", "yıl"]).reset_index(drop=True)

# (c) Nüfus dinamiği
df["nufus_growth"] = df.groupby("il_adı")["nufus"].pct_change()

# (d) Hedef bazlı türev özellikler (lag, rolling, yoy, delta, il_mean)
TUM_HEDEFLER = ["yas_0_14", "yas_65_plus", "hekim_oran", "yatak_oran"]
for h in TUM_HEDEFLER:
    g = df.groupby("il_adı")[h]
    df[f"{h}_lag1"]    = g.shift(1)
    df[f"{h}_lag2"]    = g.shift(2)
    df[f"{h}_rmean3"]  = g.transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    df[f"{h}_rmean5"]  = g.transform(lambda x: x.shift(1).rolling(5, min_periods=2).mean())
    df[f"{h}_yoy"]     = g.pct_change().shift(1)
    df[f"{h}_delta"]   = g.diff().shift(1)
    df[f"{h}_il_mean"] = g.transform(lambda x: x.shift(1).expanding().mean())

# (e) SAĞLIK ALTYAPI LAG ÖZELLİKLERİ — yeni eklenen, leak'siz
#     (Aynı yıl değerini KULLANMA — sadece lag1, lag2)
ALTYAPI = ["hastane", "yatak", "doktor", "hemsire", "ebe", "diger"]
for a in ALTYAPI:
    g = df.groupby("il_adı")[a]
    df[f"{a}_lag1"] = g.shift(1)
    df[f"{a}_lag2"] = g.shift(2)
    df[f"{a}_yoy"]  = g.pct_change().shift(1)
print(f"Veri (altyapı lag'leri eklendi): {df.shape}, sütun sayısı: {df.shape[1]}")

# %% [3] FEATURE MAP — her hedefe özel ===============================
# yas_0_14 için: ebe sayısı + altyapı = doğumla ilgili
# yatak_oran için: hastane + yatak + doktor = yatak kapasitesiyle ilgili
FEATURE_MAP = {
    "yas_0_14": [
        # ortak
        "yıl", "il_id", "bolge_id", "nufus", "nufus_growth",
        # yaş kendi türevleri
        "yas_0_14_lag1", "yas_0_14_lag2", "yas_0_14_rmean3", "yas_0_14_rmean5",
        "yas_0_14_yoy", "yas_0_14_delta", "yas_0_14_il_mean",
        # cross-hedef
        "yas_65_plus_lag1", "hekim_oran_lag1", "yatak_oran_lag1",
        # sağlık altyapı (ebe = doğum bağı, hemsire = sağlık genel)
        "ebe_lag1", "ebe_lag2", "ebe_yoy",
        "hemsire_lag1", "hastane_lag1", "doktor_lag1",
    ],
    "yatak_oran": [
        # ortak
        "yıl", "il_id", "bolge_id", "nufus", "nufus_growth",
        # yatak kendi türevleri
        "yatak_oran_lag1", "yatak_oran_lag2", "yatak_oran_rmean3", "yatak_oran_rmean5",
        "yatak_oran_yoy", "yatak_oran_delta", "yatak_oran_il_mean",
        # cross-hedef
        "yas_0_14_lag1", "yas_65_plus_lag1", "hekim_oran_lag1",
        # sağlık altyapı (yatak ve hastane direkt ilgili — sadece lag!)
        "yatak_lag1", "yatak_lag2", "yatak_yoy",
        "hastane_lag1", "hastane_lag2", "hastane_yoy",
        "doktor_lag1",
    ],
}
for h, f in FEATURE_MAP.items():
    print(f"  {h:<15} → {len(f)} özellik")

# %% [4] K-FOLD ======================================================
def make_folds(yil_baslangic, yil_son, test_size=3, n_folds=5):
    folds = []
    for i in range(n_folds, 0, -1):
        test_son = yil_son - (i - 1) * 1
        test_basla = test_son - test_size + 1
        train_son = test_basla - 1
        if train_son - yil_baslangic + 1 < 8:
            continue
        folds.append((range(yil_baslangic, train_son + 1),
                      range(test_basla, test_son + 1)))
    return folds[-n_folds:]

# %% [5] MODEL REGISTRY ==============================================
INNER_CV = 2
SCORING = "neg_mean_absolute_error"
models_registry = {
    "LinearRegression":      {"factory": lambda: GridSearchCV(LinearRegression(),
                                {"fit_intercept": [True, False]},
                                cv=INNER_CV, scoring=SCORING, n_jobs=-1), "scale": False, "type": "ml"},
    "Ridge":                 {"factory": lambda: GridSearchCV(Ridge(random_state=42),
                                {"alpha": [0.1, 1.0, 10.0]},
                                cv=INNER_CV, scoring=SCORING, n_jobs=-1), "scale": False, "type": "ml"},
    "ElasticNet":            {"factory": lambda: GridSearchCV(ElasticNet(random_state=42, max_iter=5000),
                                {"alpha": [0.1, 1.0], "l1_ratio": [0.3, 0.5, 0.7]},
                                cv=INNER_CV, scoring=SCORING, n_jobs=-1), "scale": False, "type": "ml"},
    "RandomForest":          {"factory": lambda: GridSearchCV(RandomForestRegressor(random_state=42, n_jobs=-1),
                                {"n_estimators": [100, 200], "max_depth": [None, 10]},
                                cv=INNER_CV, scoring=SCORING, n_jobs=-1), "scale": False, "type": "ml"},
    "ExtraTrees":            {"factory": lambda: GridSearchCV(ExtraTreesRegressor(random_state=42, n_jobs=-1),
                                {"n_estimators": [100, 200], "max_depth": [None, 10]},
                                cv=INNER_CV, scoring=SCORING, n_jobs=-1), "scale": False, "type": "ml"},
    "GradientBoosting":      {"factory": lambda: GridSearchCV(GradientBoostingRegressor(random_state=42),
                                {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1]},
                                cv=INNER_CV, scoring=SCORING, n_jobs=-1), "scale": False, "type": "ml"},
    "HistGradientBoosting":  {"factory": lambda: GridSearchCV(HistGradientBoostingRegressor(random_state=42),
                                {"max_iter": [100, 200], "learning_rate": [0.05, 0.1]},
                                cv=INNER_CV, scoring=SCORING, n_jobs=-1), "scale": False, "type": "ml"},
    "XGBoost":               {"factory": lambda: GridSearchCV(
                                XGBRegressor(random_state=42, verbosity=0, objective="reg:squarederror"),
                                {"n_estimators": [200, 300], "learning_rate": [0.05, 0.1]},
                                cv=INNER_CV, scoring=SCORING, n_jobs=-1), "scale": False, "type": "ml"},
    "SVR":                   {"factory": lambda: GridSearchCV(SVR(),
                                {"C": [1, 10], "epsilon": [0.05, 0.1]},
                                cv=INNER_CV, scoring=SCORING, n_jobs=-1), "scale": True, "type": "ml"},
    "KNN":                   {"factory": lambda: GridSearchCV(KNeighborsRegressor(),
                                {"n_neighbors": [3, 5, 7], "weights": ["uniform", "distance"]},
                                cv=INNER_CV, scoring=SCORING, n_jobs=-1), "scale": True, "type": "ml"},
    "Bagging":               {"factory": lambda: GridSearchCV(BaggingRegressor(random_state=42, n_jobs=-1),
                                {"n_estimators": [50, 100], "max_samples": [0.7, 1.0]},
                                cv=INNER_CV, scoring=SCORING, n_jobs=-1), "scale": False, "type": "ml"},
    "Voting":                {"factory": lambda: VotingRegressor([
                                ("rf", RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)),
                                ("xgb", XGBRegressor(n_estimators=200, random_state=42, verbosity=0,
                                                     objective="reg:squarederror")),
                                ("gb", GradientBoostingRegressor(n_estimators=200, random_state=42))]),
                              "scale": False, "type": "ml"},
    "Stacking":              {"factory": lambda: StackingRegressor(
                                estimators=[
                                    ("rf", RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)),
                                    ("xgb", XGBRegressor(n_estimators=200, random_state=42, verbosity=0,
                                                         objective="reg:squarederror"))],
                                final_estimator=LinearRegression(), n_jobs=-1),
                              "scale": False, "type": "ml"},
    "HoltWinters":           {"factory": None, "scale": False, "type": "uni"},
    "ARIMA":                 {"factory": None, "scale": False, "type": "uni"},
}
if HAVE_CAT:
    models_registry["CatBoost"] = {"factory": lambda: GridSearchCV(
        CatBoostRegressor(verbose=0, random_seed=42),
        {"iterations": [200, 300], "depth": [4, 6]},
        cv=INNER_CV, scoring=SCORING, n_jobs=-1), "scale": False, "type": "ml"}
if HAVE_PROPHET:
    models_registry["Prophet"] = {"factory": None, "scale": False, "type": "uni"}

print(f"\nToplam {len(models_registry)} model kayıtlı.")

# %% [6] METRİK ======================================================
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

# %% [7] UNIVARIATE FORECAST =========================================
def uni_holt(seri, h):
    s = pd.Series(seri).astype(float)
    if len(s) < 4 or s.std() == 0: return np.repeat(s.iloc[-1], h)
    try:
        m = ExponentialSmoothing(s, trend="add", seasonal=None).fit(optimized=True)
        return np.clip(m.forecast(steps=h).values, 0, None)
    except: return np.repeat(s.iloc[-1], h)

def uni_arima(seri, h):
    s = np.asarray(seri, dtype=float)
    if len(s) < 6: return np.repeat(s[-1], h)
    best, ba = None, np.inf
    for o in [(1,1,1),(2,1,1),(0,1,1),(1,1,2)]:
        try:
            m = ARIMA(s, order=o).fit()
            if m.aic < ba: best, ba = m, m.aic
        except: continue
    if best is None: return np.repeat(s[-1], h)
    return np.clip(np.asarray(best.forecast(steps=h)), 0, None)

def uni_prophet(yrs, vals, h):
    if not HAVE_PROPHET or len(vals) < 4: return np.repeat(vals[-1], h)
    try:
        dp = pd.DataFrame({"ds": pd.to_datetime([f"{y}-01-01" for y in yrs]), "y": vals})
        m = Prophet(yearly_seasonality=False, daily_seasonality=False, weekly_seasonality=False)
        m.fit(dp)
        son = max(yrs)
        future = pd.DataFrame({"ds": pd.to_datetime([f"{son+i}-01-01" for i in range(1, h+1)])})
        return np.clip(m.predict(future)["yhat"].values, 0, None)
    except: return np.repeat(vals[-1], h)

def uni_forecast(ad, yrs, vals, h):
    if ad == "HoltWinters": return uni_holt(vals, h)
    if ad == "ARIMA":       return uni_arima(vals, h)
    if ad == "Prophet":     return uni_prophet(list(yrs), list(vals), h)
    raise ValueError(ad)

# %% [8] EĞİTİM DÖNGÜSÜ ==============================================
butun_sonuclar = []

def egit_hedef(hedef):
    yb = int(hedefler_meta[hedefler_meta["hedef"] == hedef]["min_yil"].iloc[0])
    ys = int(hedefler_meta[hedefler_meta["hedef"] == hedef]["max_yil"].iloc[0])
    folds = make_folds(yb, ys, test_size=3, n_folds=5)
    veri = df[(df["yıl"] >= yb) & (df["yıl"] <= ys)].dropna(subset=[hedef]).copy()
    feats = FEATURE_MAP[hedef]

    print(f"\n========== {hedef} ({yb}-{ys}) ==========")
    t0 = time.time()
    for ad, info in models_registry.items():
        for f_idx, (tr_y, te_y) in enumerate(folds, 1):
            tr = veri[veri["yıl"].isin(tr_y)]
            te = veri[veri["yıl"].isin(te_y)]
            if tr.empty or te.empty: continue
            try:
                if info["type"] == "ml":
                    feats_kullan = [f for f in feats if f in tr.columns and tr[f].notna().any()]
                    train = tr[feats_kullan + [hedef]].dropna()
                    test_ok = te[feats_kullan + [hedef]].dropna()
                    if train.empty or test_ok.empty: continue
                    X_tr, y_tr = train[feats_kullan].values, train[hedef].values
                    X_te, y_te = test_ok[feats_kullan].values, test_ok[hedef].values
                    if info["scale"]:
                        sc = StandardScaler().fit(X_tr)
                        X_tr, X_te = sc.transform(X_tr), sc.transform(X_te)
                    mdl = info["factory"]()
                    mdl.fit(X_tr, y_tr)
                    pred = np.clip(mdl.predict(X_te), 0, None)
                else:
                    gercekler, tahminler = [], []
                    for il in veri["il_adı"].unique():
                        s_il = veri[veri["il_adı"] == il].sort_values("yıl")
                        s_tr = s_il[s_il["yıl"].isin(tr_y)][hedef].values
                        s_te = s_il[s_il["yıl"].isin(te_y)][hedef].values
                        yrs_tr = s_il[s_il["yıl"].isin(tr_y)]["yıl"].values
                        if len(s_tr) < 6 or len(s_te) == 0: continue
                        pr = uni_forecast(ad, yrs_tr, s_tr, len(s_te))
                        gercekler.extend(s_te.tolist()); tahminler.extend(pr.tolist())
                    if not gercekler: continue
                    y_te = np.array(gercekler); pred = np.array(tahminler)

                m = metrikler(y_te, pred)
                m.update(dict(Model=ad, Hedef=hedef, Fold=f_idx,
                              TrainSon=max(tr_y), TestSon=max(te_y)))
                butun_sonuclar.append(m)
            except Exception as e:
                print(f"  [HATA] {ad}/fold{f_idx}: {e}")
                continue
        print(f"  {ad:<22} bitti")
    print(f"{hedef} süre: {(time.time()-t0)/60:.1f} dk")

# %% [9] ÇALIŞTIR ====================================================
for h in HEDEFLER:
    egit_hedef(h)

# %% [10] ÖZET =======================================================
sdf = pd.DataFrame(butun_sonuclar)
sdf.to_csv("topsis_metrikleri_yas_yatak_genis.csv", index=False, encoding="utf-8-sig")

genel = (sdf.groupby(["Hedef", "Model"])
         [["Accuracy", "MAE", "RMSE", "R2", "ErrorRate", "Precision", "Recall", "F1"]]
         .mean().round(3).reset_index()
         .sort_values(["Hedef", "Accuracy"], ascending=[True, False]))
print("\n========== HEDEF × MODEL (tüm fold ortalamaları) ==========")
print(genel.to_string(index=False))

ana = sdf[sdf["Fold"] == 5]
en_iyi = (ana.sort_values("Accuracy", ascending=False).groupby("Hedef").head(1)
          [["Hedef", "Model", "Accuracy", "MAE", "RMSE", "R2"]])
en_iyi.to_csv("topsis_en_iyi_yas_yatak_genis.csv", index=False, encoding="utf-8-sig")
print("\n========== HER HEDEF İÇİN EN İYİ (Fold 5) ==========")
print(en_iyi.to_string(index=False))
