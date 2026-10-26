# =====================================================================
# 06 - TOPSIS VERİ HAZIRLIĞI
#
# 4 hedef değişken:
#   1) yas_0_14       — 0-14 yaş nüfus ORANI (yüzde)         [2007-2025]
#   2) yas_65_plus    — 65+ yaş nüfus ORANI (yüzde)          [2007-2025]
#   3) hekim_oran     — 1000 kişi başına hekim sayısı        [2002-2024]
#   4) yatak_oran     — 100 bin kişi başına yatak sayısı     [2002-2024]
#
# Girdiler (uploads klasöründen veya yenicalisma altından):
#   - 2025nufus-yas.csv               (yaş grubu mutlak rakamlar)
#   - 1000kisibasinahekim.csv         (hekim oranı, pivot)
#   - 100binkisibasinahastaneyatak.csv (yatak oranı, pivot)
#   - model_giris.csv                 (toplam il nüfusu — yaş yüzdesi için)
#
# Çıktı:
#   - topsis_veri.csv  →  uzun format: il_adı, yıl, hedef, deger
# =====================================================================

# %% [1] KÜTÜPHANELER ===============================================
import warnings
warnings.filterwarnings("ignore")
import re
import numpy as np
import pandas as pd

# %% [2] YARDIMCI - PİVOT TÜİK CSV PARSER ===========================
def parse_pivot_tuik(yol, isim):
    """
    "100binkisibasinahastaneyatak.csv" gibi pivot TÜİK CSV'lerini okur.
    Format:
       Satır 0-1: başlık satırları (il isimleri 2. satırda)
       Satır 2-3: boş
       Satır 4+:  '...| Ölçüm bazında | YIL | val_il1 | val_il2 | ...'
    Çıktı: long format DataFrame [il_adı, yıl, deger]
    """
    with open(yol, encoding="utf-8") as f:
        lines = f.readlines()

    # 2. satırdaki il isimlerini al
    il_satir = lines[1].split("|")
    iller = []
    for c in il_satir[3:]:
        c = c.strip()
        if not c: continue
        # "Adana-1", "Adıyaman-2" → "Adana", "Adıyaman"
        ad = re.sub(r"-\d+$", "", c)
        iller.append(ad)

    veri = []
    for ln in lines[4:]:
        parts = ln.rstrip("\n").split("|")
        if len(parts) < 4: continue
        # Yıl 2. veya 3. kolonda olabilir
        yil = None
        for p in parts[:4]:
            ps = p.strip()
            if ps.isdigit() and 1990 < int(ps) < 2100:
                yil = int(ps); break
        if yil is None: continue
        # Yıl pozisyonundan sonraki ilk 81 değer iller
        yil_idx = parts.index(str(yil))
        degerler = parts[yil_idx + 1:]
        for i, il in enumerate(iller):
            if i >= len(degerler): break
            v = degerler[i].strip().replace(",", ".")
            try:
                veri.append({"il_adı": il, "yıl": yil, isim: float(v)})
            except ValueError:
                continue
    return pd.DataFrame(veri)

# %% [3] HEKİM ORANI (1000 kişi başına) =============================
hekim = parse_pivot_tuik("1000kisibasinahekim.csv", "hekim_oran")
print(f"hekim_oran: {hekim.shape}, yıl: {hekim['yıl'].min()}-{hekim['yıl'].max()}, il: {hekim['il_adı'].nunique()}")
print(hekim.head())

# %% [4] YATAK ORANI (100 bin kişi başına) ==========================
yatak = parse_pivot_tuik("100binkisibasinahastaneyatak.csv", "yatak_oran")
print(f"yatak_oran: {yatak.shape}, yıl: {yatak['yıl'].min()}-{yatak['yıl'].max()}")
print(yatak.head())

# %% [5] YAŞ VERİSİ - 2025nufus-yas.csv parser =====================
# Format: yıllar satır blokları halinde, her ilin 9 yaş grubu kolon
# Yaş kolonları: 0-4 | 10-14 | 5-9 | 65-69 | 70-74 | 75-79 | 80-84 | 85-89 | 90+

def parse_yas_csv(yol):
    with open(yol, encoding="utf-8") as f:
        lines = f.readlines()
    # Yaş grupları satır 2 (index 2)
    yas_satir = lines[2].split("|")
    yas_kolon = [c.strip() for c in yas_satir if c.strip()]
    print("Yaş grupları:", yas_kolon)

    veri = []
    aktif_yil = None
    for ln in lines[5:]:
        parts = ln.rstrip("\n").split("|")
        if len(parts) < 3: continue
        # Yıl mı il mi tespit et
        p0 = parts[0].strip()
        if p0.isdigit() and 2000 < int(p0) < 2100:
            aktif_yil = int(p0)
            p1 = parts[1].strip() if len(parts) > 1 else ""
        else:
            p1 = p0
        if aktif_yil is None: continue
        # il_adi parts[1] veya parts[0]
        il_ham = parts[1].strip() if (p0.isdigit() and 2000 < int(p0) < 2100) else parts[0].strip()
        if not il_ham: continue
        il = re.sub(r"-\d+$", "", il_ham)
        # değerler — il'den sonraki kolonlarda
        baslangic = 2 if (p0.isdigit() and 2000 < int(p0) < 2100) else 1
        degerler_str = [p.strip() for p in parts[baslangic:baslangic + len(yas_kolon)]]
        degerler = []
        for v in degerler_str:
            try: degerler.append(float(v))
            except: degerler.append(np.nan)
        if len(degerler) < len(yas_kolon): continue
        kayit = {"il_adı": il, "yıl": aktif_yil}
        for k, v in zip(yas_kolon, degerler):
            kayit[k] = v
        veri.append(kayit)
    return pd.DataFrame(veri)

yas_ham = parse_yas_csv("2025nufus-yas.csv")
print(f"\nYaş ham: {yas_ham.shape}, yıl: {yas_ham['yıl'].min()}-{yas_ham['yıl'].max()}")
print(yas_ham.head())

# %% [6] YAŞ GRUBU TOPLAMLARI =======================================
# 0-14 = "0-4" + "5-9" + "10-14"
# 65+  = "65-69" + "70-74" + "75-79" + "80-84" + "85-89" + "90+"
yas_ham["yas_0_14_mutlak"] = yas_ham[["0-4", "5-9", "10-14"]].sum(axis=1)
yas_ham["yas_65_plus_mutlak"] = yas_ham[["65-69", "70-74", "75-79", "80-84", "85-89", "90+"]].sum(axis=1)
yas_ham["nufus_yas_toplam"] = yas_ham[["0-4", "5-9", "10-14",
                                       "65-69", "70-74", "75-79", "80-84", "85-89", "90+"]].sum(axis=1)

# Yüzde için toplam il nüfusu lazım
toplam_nufus = pd.read_csv("model_giris.csv")[["il_adı", "yıl", "nufus"]]
yas_ham = yas_ham.merge(toplam_nufus, on=["il_adı", "yıl"], how="left")

# Eksik nüfus için yaş toplam üzerinden tahmin
yas_ham["nufus"] = yas_ham["nufus"].fillna(yas_ham["nufus_yas_toplam"])

yas_ham["yas_0_14"]    = 100 * yas_ham["yas_0_14_mutlak"] / yas_ham["nufus"]
yas_ham["yas_65_plus"] = 100 * yas_ham["yas_65_plus_mutlak"] / yas_ham["nufus"]

print("\nYaş yüzdesi örnek:")
print(yas_ham[["il_adı", "yıl", "yas_0_14", "yas_65_plus"]].head(10).to_string(index=False))

# %% [7] BİRLEŞTİR ==================================================
yas_long = yas_ham[["il_adı", "yıl", "yas_0_14", "yas_65_plus"]]
hekim_long = hekim[["il_adı", "yıl", "hekim_oran"]]
yatak_long = yatak[["il_adı", "yıl", "yatak_oran"]]

# Tüm hedefleri (il, yıl) bazında birleştir
topsis_wide = (yas_long
               .merge(hekim_long, on=["il_adı", "yıl"], how="outer")
               .merge(yatak_long, on=["il_adı", "yıl"], how="outer")
               .sort_values(["il_adı", "yıl"]).reset_index(drop=True))

print(f"\ntopsis_wide: {topsis_wide.shape}")
print(topsis_wide.head(15).to_string(index=False))

# Eksik hücre sayısı
print("\nEksik hücre sayısı:")
print(topsis_wide.isna().sum())

topsis_wide.to_csv("topsis_veri.csv", index=False, encoding="utf-8-sig")
print("\n[OK] topsis_veri.csv kaydedildi.")

# %% [8] HEDEF YIL ARALIKLARI =======================================
hedef_yillar = {}
for h in ["yas_0_14", "yas_65_plus", "hekim_oran", "yatak_oran"]:
    s = topsis_wide.dropna(subset=[h])
    hedef_yillar[h] = (int(s["yıl"].min()), int(s["yıl"].max()))
    print(f"  {h:<15}  {hedef_yillar[h]}")

pd.DataFrame([{"hedef": k, "min_yil": v[0], "max_yil": v[1]}
              for k, v in hedef_yillar.items()]).to_csv(
    "topsis_hedef_yillar.csv", index=False, encoding="utf-8-sig")
print("[OK] topsis_hedef_yillar.csv kaydedildi.")
