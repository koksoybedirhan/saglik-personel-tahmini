# =====================================================================
# 07 - TOPSIS HEDEFLERİ İÇİN MODEL EĞİTİMİ
#
# 4 hedef: yas_0_14, yas_65_plus, hekim_oran, yatak_oran
# Her hedef kendi yıl aralığında (csv'den otomatik) çalışır.
#
# K-fold: 5 rolling expanding fold, her test 3 yıl
# Modeller (17): Linear, Ridge, ElasticNet, RandomForest, ExtraTrees,
#                GradientBoosting, HistGradientBoosting, XGBoost,
#                CatBoost, SVR, KNN, Bagging, Voting, Stacking,
#                ARIMA, HoltWinters, Prophet
#
# Çıktılar:
#   - topsis_metrikleri_detay.csv   (model × hedef × fold)
#   - topsis_metrikleri_genel.csv   (model bazında genel)
#   - topsis_en_iyi_modeller.csv    (her hedef için Fold-5 en iyi)
#   - topsis_tahminler_fold5.csv    (Fold-5 test tahminleri)
# =====================================================================

# %% [1] KÜTÜPHANELER ===============================================
import warnings, time
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

INNER_CV = 2
SCORING  = "neg_mean_absolute_error"

# %% [2] VERİYİ YÜKLE + GENİŞ FEATURE ENGINEERING ====================
import os
df = pd.read_csv("topsis_veri.csv").sort_values(["il_adı", "yıl"]).reset_index(drop=True)
hedefler_meta = pd.read_csv("topsis_hedef_yillar.csv")
HEDEFLER = ["yas_0_14", "yas_65_plus", "hekim_oran", "yatak_oran"]

# (a) Toplam il nüfusu
for yol in ["../yenicalisma/model_giris.csv", "model_giris.csv", "../model_giris.csv"]:
    if os.path.exists(yol):
        nf = pd.read_csv(yol)[["il_adı", "yıl", "nufus"]]
        df = df.merge(nf, on=["il_adı", "yıl"], how="left")
        print(f"Nüfus eklendi ({yol})")
        break
else:
    df["nufus"] = np.nan
    print("[UYARI] model_giris.csv bulunamadı")

# (b) İl kimliği (sayısal — tree modeller için)
il_id_map = {il: i for i, il in enumerate(sorted(df["il_adı"].unique()))}
df["il_id"] = df["il_adı"].map(il_id_map)

# (c) Bölge bilgisi — coğrafi gruplama
bolge_map = {
    'İstanbul':'Marmara','Edirne':'Marmara','Kırklareli':'Marmara','Tekirdağ':'Marmara',
    'Kocaeli':'Marmara','Sakarya':'Marmara','Yalova':'Marmara','Bilecik':'Marmara',
    'Balıkesir':'Marmara','Çanakkale':'Marmara','Bursa':'Marmara',
    'İzmir':'Ege','Aydın':'Ege','Muğla':'Ege','Manisa':'Ege',
    'Afyonkarahisar':'Ege','Uşak':'Ege','Kütahya':'Ege','Denizli':'Ege',
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
df["bolge"] = df["il_adı"].map(bolge_map)
df["bolge_id"] = df["bolge"].astype("category").cat.codes
print(f"Bölge ataması bilinmeyen: {df['bolge'].isna().sum()} satır")

# (d) Nüfus büyüme oranı
df = df.sort_values(["il_adı", "yıl"]).reset_index(drop=True)
df["nufus_growth"] = df.groupby("il_adı")["nufus"].pct_change()
df["nufus_growth_5y"] = df.groupby("il_adı")["nufus"].transform(lambda x: x.pct_change(periods=5))

# (e) Her hedef için lag, rolling, yoy, delta, il_mean
for h in HEDEFLER:
    g = df.groupby("il_adı")[h]
    df[f"{h}_lag1"]   = g.shift(1)
    df[f"{h}_lag2"]   = g.shift(2)
    df[f"{h}_rmean3"] = g.transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    df[f"{h}_rmean5"] = g.transform(lambda x: x.shift(1).rolling(5, min_periods=2).mean())
    df[f"{h}_yoy"]    = g.pct_change().shift(1)         # önceki yılın değişim oranı
    df[f"{h}_delta"]  = g.diff().shift(1)               # önceki yılın mutlak değişimi
    # İl-ortalama (target encoding) — leak'i azaltmak için lag-tabanlı genişleyen ortalama:
    df[f"{h}_il_mean"] = g.transform(lambda x: x.shift(1).expanding().mean())

print(f"\nVeri (genişletilmiş): {df.shape}")
print(f"Toplam {df.shape[1]} sütun var. İlk 8 örnek:")
print(df.head(8).to_string(index=False))
print("\nHedef yıl aralıkları:")
print(hedefler_meta.to_string(index=False))

# %% [3] FEATURE MAP — genişletilmiş set =============================
# Her hedef için:
#   - bolge_id, il_id, nufus, nufus_growth          → şehir bilgisi
#   - yıl                                           → zaman
#   - kendi: lag1, lag2, rmean3, rmean5, yoy, delta, il_mean (7 özellik)
#   - diğer 3 hedefin lag1'i                       → cross-target sinyali
def feats_for(hedef):
    digerleri = [h for h in HEDEFLER if h != hedef]
    base = ["yıl", "il_id", "bolge_id", "nufus", "nufus_growth"]
    kendi = [f"{hedef}_lag1", f"{hedef}_lag2", f"{hedef}_rmean3", f"{hedef}_rmean5",
             f"{hedef}_yoy", f"{hedef}_delta", f"{hedef}_il_mean"]
    cross = [f"{h}_lag1" for h in digerleri]
    return base + kendi + cross

FEATURE_MAP = {h: feats_for(h) for h in HEDEFLER}
print(f"\nGenişletilmiş feature map (her hedef için {len(FEATURE_MAP['yas_0_14'])} özellik):")
for h, f in FEATURE_MAP.items():
    print(f"  {h:<15} → {len(f)} özellik")

# %% [4] K-FOLD: hedef bazında dinamik =============================
def make_folds(yil_baslangic, yil_son, test_size=3, n_folds=5):
    """Rolling expanding: 5 fold, her test 3 yıl. En sondan başlar."""
    folds = []
    for i in range(n_folds, 0, -1):
        test_son = yil_son - (i - 1) * 1
        test_basla = test_son - test_size + 1
        train_son = test_basla - 1
        if train_son - yil_baslangic + 1 < 8:  # min 8 yıl train
            continue
        folds.append((range(yil_baslangic, train_son + 1),
                      range(test_basla, test_son + 1)))
    # Son fold ana fold olsun
    return folds[-n_folds:]

# Test
for h in HEDEFLER:
    yb = int(hedefler_meta[hedefler_meta["hedef"] == h]["min_yil"].iloc[0])
    ys = int(hedefler_meta[hedefler_meta["hedef"] == h]["max_yil"].iloc[0])
    fs = make_folds(yb, ys)
    print(f"{h:<15} ({yb}-{ys}) {len(fs)} fold:")
    for i, (tr, te) in enumerate(fs, 1):
        print(f"   Fold {i}: train {min(tr)}-{max(tr)} | test {min(te)}-{max(te)}")

# %% [5.0] MODEL REGISTRY ==========================================
models_registry = {}

# %% [5.1] LinearRegression
def f_linear():
    return GridSearchCV(LinearRegression(), {"fit_intercept": [True, False]},
                        cv=INNER_CV, scoring=SCORING, n_jobs=-1)
models_registry["LinearRegression"] = {"factory": f_linear, "scale": False, "type": "ml"}

# %% [5.2] Ridge
def f_ridge():
    return GridSearchCV(Ridge(random_state=42), {"alpha": [0.1, 1.0, 10.0]},
                        cv=INNER_CV, scoring=SCORING, n_jobs=-1)
models_registry["Ridge"] = {"factory": f_ridge, "scale": False, "type": "ml"}

# %% [5.3] ElasticNet
def f_enet():
    return GridSearchCV(ElasticNet(random_state=42, max_iter=5000),
                        {"alpha": [0.1, 1.0], "l1_ratio": [0.3, 0.5, 0.7]},
                        cv=INNER_CV, scoring=SCORING, n_jobs=-1)
models_registry["ElasticNet"] = {"factory": f_enet, "scale": False, "type": "ml"}

# %% [5.4] RandomForest
def f_rf():
    return GridSearchCV(RandomForestRegressor(random_state=42, n_jobs=-1),
                        {"n_estimators": [100, 200], "max_depth": [None, 10]},
                        cv=INNER_CV, scoring=SCORING, n_jobs=-1)
models_registry["RandomForest"] = {"factory": f_rf, "scale": False, "type": "ml"}

# %% [5.5] ExtraTrees
def f_et():
    return GridSearchCV(ExtraTreesRegressor(random_state=42, n_jobs=-1),
                        {"n_estimators": [100, 200], "max_depth": [None, 10]},
                        cv=INNER_CV, scoring=SCORING, n_jobs=-1)
models_registry["ExtraTrees"] = {"factory": f_et, "scale": False, "type": "ml"}

# %% [5.6] GradientBoosting
def f_gb():
    return GridSearchCV(GradientBoostingRegressor(random_state=42),
                        {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1]},
                        cv=INNER_CV, scoring=SCORING, n_jobs=-1)
models_registry["GradientBoosting"] = {"factory": f_gb, "scale": False, "type": "ml"}

# %% [5.7] HistGradientBoosting
def f_hgb():
    return GridSearchCV(HistGradientBoostingRegressor(random_state=42),
                        {"max_iter": [100, 200], "learning_rate": [0.05, 0.1]},
                        cv=INNER_CV, scoring=SCORING, n_jobs=-1)
models_registry["HistGradientBoosting"] = {"factory": f_hgb, "scale": False, "type": "ml"}

# %% [5.8] XGBoost
def f_xgb():
    base = XGBRegressor(random_state=42, verbosity=0, objective="reg:squarederror")
    return GridSearchCV(base, {"n_estimators": [200, 300], "learning_rate": [0.05, 0.1]},
                        cv=INNER_CV, scoring=SCORING, n_jobs=-1)
models_registry["XGBoost"] = {"factory": f_xgb, "scale": False, "type": "ml"}

# %% [5.9] CatBoost
if HAVE_CAT:
    def f_cat():
        return GridSearchCV(CatBoostRegressor(verbose=0, random_seed=42),
                            {"iterations": [200, 300], "depth": [4, 6]},
                            cv=INNER_CV, scoring=SCORING, n_jobs=-1)
    models_registry["CatBoost"] = {"factory": f_cat, "scale": False, "type": "ml"}

# %% [5.10] SVR
def f_svr():
    return GridSearchCV(SVR(), {"C": [1, 10], "epsilon": [0.05, 0.1]},
                        cv=INNER_CV, scoring=SCORING, n_jobs=-1)
models_registry["SVR"] = {"factory": f_svr, "scale": True, "type": "ml"}

# %% [5.11] KNN
def f_knn():
    return GridSearchCV(KNeighborsRegressor(),
                        {"n_neighbors": [3, 5, 7], "weights": ["uniform", "distance"]},
                        cv=INNER_CV, scoring=SCORING, n_jobs=-1)
models_registry["KNN"] = {"factory": f_knn, "scale": True, "type": "ml"}

# %% [5.12] Bagging
def f_bag():
    return GridSearchCV(BaggingRegressor(random_state=42, n_jobs=-1),
                        {"n_estimators": [50, 100], "max_samples": [0.7, 1.0]},
                        cv=INNER_CV, scoring=SCORING, n_jobs=-1)
models_registry["Bagging"] = {"factory": f_bag, "scale": False, "type": "ml"}

# %% [5.13] Voting
def f_vote():
    return VotingRegressor([
        ("rf",  RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)),
        ("xgb", XGBRegressor(n_estimators=200, random_state=42, verbosity=0,
                             objective="reg:squarederror")),
        ("gb",  GradientBoostingRegressor(n_estimators=200, random_state=42))])
models_registry["Voting"] = {"factory": f_vote, "scale": False, "type": "ml"}

# %% [5.14] Stacking
def f_stack():
    return StackingRegressor(
        estimators=[
            ("rf",  RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)),
            ("xgb", XGBRegressor(n_estimators=200, random_state=42, verbosity=0,
                                 objective="reg:squarederror"))],
        final_estimator=LinearRegression(), n_jobs=-1)
models_registry["Stacking"] = {"factory": f_stack, "scale": False, "type": "ml"}

# %% [5.15] HoltWinters (univariate)
models_registry["HoltWinters"] = {"factory": None, "scale": False, "type": "uni"}

# %% [5.16] ARIMA (univariate)
models_registry["ARIMA"] = {"factory": None, "scale": False, "type": "uni"}

# %% [5.17] Prophet (univariate, opsiyonel)
if HAVE_PROPHET:
    models_registry["Prophet"] = {"factory": None, "scale": False, "type": "uni"}

print(f"\nToplam {len(models_registry)} model kayıtlı.")

# %% [6] METRİK FONKSİYONU =========================================
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

# %% [7] UNIVARIATE FORECAST FONKSİYONLARI =========================
def uni_holt(seri, h):
    s = pd.Series(seri).astype(float)
    if len(s) < 4 or s.std() == 0: return np.repeat(s.iloc[-1], h)
    try:
        m = ExponentialSmoothing(s, trend="add", seasonal=None).fit(optimized=True)
        return np.clip(m.forecast(steps=h).values, 0, None)
    except Exception: return np.repeat(s.iloc[-1], h)

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

# %% [8] EĞİTİM DÖNGÜSÜ FONKSİYONU =================================
butun_sonuclar  = []
butun_tahminler = []

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
                    iller_t = test_ok["il_adı"].values if "il_adı" in test_ok.columns else te["il_adı"].values[:len(y_te)]
                    yillar_t = te["yıl"].values[:len(y_te)]
                else:  # univariate
                    gercekler, tahminler_il, iller_t, yillar_t = [], [], [], []
                    for il in veri["il_adı"].unique():
                        s_il = veri[veri["il_adı"] == il].sort_values("yıl")
                        s_tr = s_il[s_il["yıl"].isin(tr_y)][hedef].values
                        s_te = s_il[s_il["yıl"].isin(te_y)][hedef].values
                        yrs_tr = s_il[s_il["yıl"].isin(tr_y)]["yıl"].values
                        if len(s_tr) < 6 or len(s_te) == 0: continue
                        pr = uni_forecast(ad, yrs_tr, s_tr, len(s_te))
                        gercekler.extend(s_te.tolist())
                        tahminler_il.extend(pr.tolist())
                        iller_t.extend([il]*len(s_te))
                        yillar_t.extend(s_il[s_il["yıl"].isin(te_y)]["yıl"].tolist())
                    if not gercekler: continue
                    y_te = np.array(gercekler); pred = np.array(tahminler_il)

                m = metrikler(y_te, pred)
                m.update(dict(Model=ad, Hedef=hedef, Fold=f_idx,
                              TrainSon=max(tr_y), TestSon=max(te_y)))
                butun_sonuclar.append(m)
                if f_idx == 5:
                    for i in range(len(y_te)):
                        butun_tahminler.append(dict(
                            il_adı=str(iller_t[i]), yıl=int(yillar_t[i]), hedef=hedef,
                            model=ad, gercek=float(y_te[i]), tahmin=float(pred[i])))
            except Exception as e:
                print(f"  [HATA] {ad}/fold{f_idx}: {e}")
                continue
        print(f"  {ad:<22} bitti")
    print(f"{hedef} süre: {(time.time()-t0)/60:.1f} dk")

# %% [9] TÜM HEDEFLERİ EĞİT ========================================
t_all = time.time()
for h in HEDEFLER:
    egit_hedef(h)
print(f"\nTüm süre: {(time.time()-t_all)/60:.1f} dk")

# %% [10] ÖZET TABLOLAR ============================================
sdf = pd.DataFrame(butun_sonuclar)
sdf.to_csv("topsis_metrikleri_detay.csv", index=False, encoding="utf-8-sig")

genel = (sdf.groupby("Model")[["Accuracy","MAE","RMSE","R2","ErrorRate","Precision","Recall","F1"]]
         .mean().round(3).reset_index().sort_values("Accuracy", ascending=False))
genel.to_csv("topsis_metrikleri_genel.csv", index=False, encoding="utf-8-sig")
print("\n========== GENEL ==========")
print(genel.to_string(index=False))

ana = sdf[sdf["Fold"] == 5]
en_iyi = (ana.sort_values("Accuracy", ascending=False).groupby("Hedef").head(1)
          [["Hedef", "Model", "Accuracy", "MAE", "RMSE", "R2"]])
en_iyi.to_csv("topsis_en_iyi_modeller.csv", index=False, encoding="utf-8-sig")
print("\n========== HER HEDEF İÇİN EN İYİ (Fold 5) ==========")
print(en_iyi.to_string(index=False))

pd.DataFrame(butun_tahminler).to_csv("topsis_tahminler_fold5.csv",
                                     index=False, encoding="utf-8-sig")
print("\n[OK] Çıktılar yazıldı.")