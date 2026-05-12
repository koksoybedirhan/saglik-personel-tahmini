# =====================================================================
# 03 - GELECEK TAHMİNİ İÇİN VERİ HAZIRLIĞI
#
# Bu script 04 ve 05 (tekil + ensemble) için ortak baseline kurar:
#   - model_giris.csv'yi yükler
#   - doktor = uzman + pratisyen + asistan birleşik kolonunu ekler
#   - 2025-2030 için il bazlı nüfus projeksiyonu üretir
#     (TÜİK projeksiyonu varsa onu kullanır, yoksa lineer ekstrapolasyon)
#
# Çıktılar:
#   - gelecek_baz.csv          (2002-2024 tam + 2025-2030 nüfus + diğer hedefler NaN)
#   - nufus_projeksiyonu.csv   (sadece 2025-2030 nüfus, kontrol amaçlı)
# =====================================================================

# %% [1] KÜTÜPHANELER & VERİ ========================================
import os, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

df = pd.read_csv("model_giris.csv").sort_values(["il_adı", "yıl"]).reset_index(drop=True)
df["doktor"] = df["uzman"] + df["pratisyen"] + df["asistan"]
print("Veri:", df.shape)
print("İller:", df["il_adı"].nunique())
print("Yıl aralığı:", df["yıl"].min(), "→", df["yıl"].max())

HEDEFLER = ["doktor", "ebe", "diger", "hemsire", "yatak", "hastane"]

# %% [2] TÜİK PROJEKSİYONUNU OKU (varsa) ============================
TUIK_YOL = "hamveriler/tuik26-30hesaplanmis/Turkiye_Nufus_Projeksiyon_Guncellenmis.xlsx"

nufus_2025_2030 = {}    # {(il, yıl): nufus}

if os.path.exists(TUIK_YOL):
    try:
        tp = pd.read_excel(TUIK_YOL)
        print("\nTÜİK projeksiyonu okundu. Sütunlar:", list(tp.columns))
        # Sütun adları "il_adı" ve yıllar (2025, 2026, ...) varsayımı
        il_kolonu = None
        for c in tp.columns:
            if str(c).lower() in ("il_adı", "il", "il_ad", "il_adi"):
                il_kolonu = c; break
        if il_kolonu is None:
            il_kolonu = tp.columns[0]
        for _, r in tp.iterrows():
            il = str(r[il_kolonu]).strip()
            for y in range(2025, 2031):
                if y in tp.columns:
                    nufus_2025_2030[(il, y)] = float(r[y])
        print(f"TÜİK projeksiyonundan {len(nufus_2025_2030)} (il, yıl) çifti alındı.")
    except Exception as e:
        print(f"[UYARI] TÜİK dosyası okunamadı: {e}")

# %% [3] EKSİK YILLAR İÇİN LİNEER EKSTRAPOLASYON ====================
# TÜİK'te bulamadıklarımız için son 10 yılın trendinden tahmin
eksik = []
for il in df["il_adı"].unique():
    seri = df[df["il_adı"] == il].dropna(subset=["nufus"]).tail(10)
    if len(seri) < 3:
        continue
    coef = np.polyfit(seri["yıl"], seri["nufus"], 1)
    for y in range(2025, 2031):
        if (il, y) not in nufus_2025_2030:
            nufus_2025_2030[(il, y)] = float(np.polyval(coef, y))
            eksik.append((il, y))

if eksik:
    print(f"\nLineer ekstrapolasyonla doldurulan: {len(eksik)} (il, yıl) çifti")

# %% [4] BASELINE DATAFRAME =========================================
# 2002-2024 mevcut + 2025-2030 boş satırlar (sadece nüfus dolu)
yeni_satirlar = []
for il in df["il_adı"].unique():
    for y in range(2025, 2031):
        yeni_satirlar.append({
            "il_adı": il, "yıl": y,
            "nufus": nufus_2025_2030.get((il, y), np.nan),
            **{h: np.nan for h in HEDEFLER}
        })

gelecek_df = pd.DataFrame(yeni_satirlar)
baz = pd.concat([df, gelecek_df], ignore_index=True).sort_values(["il_adı", "yıl"]).reset_index(drop=True)
print("\nBaseline boyut:", baz.shape)
print(baz[baz["yıl"].isin([2024, 2025, 2026])].head(8).to_string(index=False))

baz.to_csv("gelecek_baz.csv", index=False, encoding="utf-8-sig")
print("\n[OK] gelecek_baz.csv kaydedildi.")

# %% [5] KONTROL DOSYASI ============================================
nf_df = pd.DataFrame([
    {"il_adı": il, **{y: nufus_2025_2030.get((il, y), np.nan) for y in range(2025, 2031)}}
    for il in sorted(df["il_adı"].unique())
])
nf_df.to_csv("nufus_projeksiyonu.csv", index=False, encoding="utf-8-sig")
print("[OK] nufus_projeksiyonu.csv kaydedildi (kontrol amaçlı).")
print(nf_df.head().to_string(index=False))
