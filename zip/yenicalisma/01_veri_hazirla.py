# =====================================================================
# 01 - VERİ HAZIRLAMA
# all_data24.csv (uzun format) → geniş format
# Çıktı: model_giris.csv  (il, yıl × uzman, pratisyen, asistan, hemsire,
#                          ebe, diger, hastane, yatak, nufus)
# =====================================================================

import pandas as pd
import numpy as np

GIRIS = "all_data24.csv"          # gerek varsa dosya yolunu düzenle
CIKTI = "model_giris.csv"

# 1) Veriyi oku ------------------------------------------------------
df = pd.read_csv(GIRIS)
print("Ham veri boyutu:", df.shape)
print("Sütunlar:", df.columns.tolist())

# 2) Eczacı ve dişçiyi at -------------------------------------------
df = df[~df["kategori"].isin(["eczaci", "dis"])].copy()

# 3) Hastane: 3 kurum (devlet/özel/üniversite) toplamı --------------
hastane = (df[df["veri_tipi"] == "hastane_sayisi"]
           .groupby(["il_adı", "yıl"])["değer"].sum().reset_index()
           .rename(columns={"değer": "hastane"}))

# 4) Yatak: 3 kurum toplamı -----------------------------------------
yatak = (df[df["veri_tipi"] == "yatak_sayisi"]
         .groupby(["il_adı", "yıl"])["değer"].sum().reset_index()
         .rename(columns={"değer": "yatak"}))

# 5) Nüfus ----------------------------------------------------------
nufus = (df[df["veri_tipi"] == "nufus"]
         .groupby(["il_adı", "yıl"])["değer"].sum().reset_index()
         .rename(columns={"değer": "nufus"}))

# 6) Personel: her kategori (uzman/pratisyen/asistan/hemsire/ebe/diger)
personel_kats = ["uzman", "pratisyen", "asistan", "hemsire", "ebe", "diger"]
personel_pivot = (df[df["kategori"].isin(personel_kats)]
                  .groupby(["il_adı", "yıl", "kategori"])["değer"].sum()
                  .reset_index()
                  .pivot(index=["il_adı", "yıl"], columns="kategori", values="değer")
                  .reset_index())
personel_pivot.columns.name = None

# 7) Hepsini birleştir ----------------------------------------------
veri = (hastane
        .merge(yatak,         on=["il_adı", "yıl"], how="outer")
        .merge(nufus,         on=["il_adı", "yıl"], how="outer")
        .merge(personel_pivot, on=["il_adı", "yıl"], how="outer"))

# Eksik (NaN) hücreleri il bazında zaman yönünde lineer interpolasyon
veri = veri.sort_values(["il_adı", "yıl"]).reset_index(drop=True)
say_kolon = ["hastane", "yatak", "nufus"] + personel_kats
veri[say_kolon] = (veri.groupby("il_adı")[say_kolon]
                       .apply(lambda x: x.interpolate(limit_direction="both"))
                       .reset_index(drop=True))

veri = veri.dropna(subset=["nufus"]).copy()

print("\nGeniş format boyutu:", veri.shape)
print("Yıl aralığı:", veri["yıl"].min(), "-", veri["yıl"].max())
print("İl sayısı:", veri["il_adı"].nunique())
print(veri.head())

veri.to_csv(CIKTI, index=False, encoding="utf-8-sig")
print(f"\n[OK] {CIKTI} kaydedildi.")
