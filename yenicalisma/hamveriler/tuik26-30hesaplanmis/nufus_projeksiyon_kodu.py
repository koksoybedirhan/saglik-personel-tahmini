"""
=============================================================================
TÜİK Nüfus Projeksiyonu Güncelleme Scripti
=============================================================================
Amaç   : TÜİK'in 2023 ADNKS baz yılıyla ürettiği il bazlı projeksiyon
         tablosunu, 2023-2024-2025 gerçek ADNKS verileri ile güncellemek.

Yöntem : Bias-Correction (Hata Düzeltme) Yöntemi
---------------------------------------------------------------------------
TÜİK'in orijinal projeksiyonu kohort-bileşen yöntemiyle üretilmiştir.
2023 tahmin baz yılı olduğu için 2024 ve 2025 için tahmin farkları
(sapma) ölçülebilir hale gelmiştir.

Adımlar:
  1. Her il için TÜİK'in öngördüğü 2024 ve 2025 değerleri ile
     gerçekleşen ADNKS değerleri arasındaki sapma hesaplanır.
  2. Bu sapmalardan il bazında ortalama yıllık mutlak hata (bias) elde edilir.
  3. Bias, 2026-2030 yılları için TÜİK'in orijinal tahmin değerlerine
     doğrusal olarak uygulanır (bias*yıl_uzaklığı formülüyle).
  4. 2023-2025 yılları gerçek ADNKS verileriyle sabit tutulur.

Kaynak: TÜİK, Nüfus Projeksiyonları 2023-2100, Yayın No: 53699
        TÜİK, Adrese Dayalı Nüfus Kayıt Sistemi, 2023-2025
=============================================================================
"""

import pandas as pd
import numpy as np
import unicodedata
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

# ─── YARDIMCI FONKSİYON ───────────────────────────────────────────────────
def ascii_norm(s):
    """Türkçe karakterleri normalize ederek eşleşme anahtarı üretir."""
    if not isinstance(s, str): return ''
    nfkd = unicodedata.normalize('NFKD', s.strip().upper())
    return ''.join(c for c in nfkd if not unicodedata.combining(c))

# ─── 1. VERİ OKUMA ────────────────────────────────────────────────────────
print("Veriler okunuyor...")

# TÜİK Orijinal Projeksiyon (2023-2030)
xls = pd.read_excel('/mnt/user-data/uploads/Yıllara_Göre_İl_Nüfusları.xls')
xls.columns = ['Il','y2023','y2024','y2025','y2026','y2027','y2028','y2029','y2030']
df_tuik = xls.iloc[2:84].copy().reset_index(drop=True)
df_tuik = df_tuik[df_tuik['y2023'].apply(
    lambda x: str(x).replace('.','').replace(',','').isdigit() if pd.notna(x) else False
)]
for c in ['y2023','y2024','y2025','y2026','y2027','y2028','y2029','y2030']:
    df_tuik[c] = pd.to_numeric(df_tuik[c], errors='coerce')
df_tuik['key'] = df_tuik['Il'].apply(ascii_norm)
df_tuik = df_tuik[df_tuik['key'] != 'TOPLAM-TOTAL'].reset_index(drop=True)

# Gerçek ADNKS Verisi (2023-2025)
df_adnks = pd.read_excel('/mnt/user-data/uploads/Cinsiyete_Göre_Nüfus.xlsx', header=1)
df_adnks = df_adnks[pd.to_numeric(df_adnks['Yıl'], errors='coerce').notna()].copy()
df_adnks['Yıl'] = pd.to_numeric(df_adnks['Yıl']).astype(int)
df_adnks['Toplam'] = pd.to_numeric(df_adnks['Toplam'], errors='coerce')
eski = ['BAYAZIT','CEBELİBEREKET','ELAZIĞ  (ELAZİZ)',
        'İÇEL (İÇEL ve MERSİN)','ŞANLIURFA  (URFA)','ŞEBİNKARAHİSAR']
df_adnks = df_adnks[~df_adnks['İl'].isin(eski)]
df_adnks['key'] = df_adnks['İl'].apply(ascii_norm)

# ADNKS pivot (yıl x il)
adnks_pivot = {}
for yr in [2023, 2024, 2025]:
    tmp = df_adnks[df_adnks['Yıl']==yr][['key','Toplam']].set_index('key')['Toplam']
    adnks_pivot[yr] = tmp

print(f"TÜİK projeksiyon: {len(df_tuik)} il")
print(f"ADNKS 2023: {len(adnks_pivot[2023])} il")

# ─── 2. BIAS-CORRECTION HESAPLAMASI ──────────────────────────────────────
print("\nBias-correction hesaplanıyor...")

results = []
for _, row in df_tuik.iterrows():
    k = row['key']
    il_adi = row['Il']

    # Gerçek değerler
    g2023 = adnks_pivot[2023].get(k, np.nan)
    g2024 = adnks_pivot[2024].get(k, np.nan)
    g2025 = adnks_pivot[2025].get(k, np.nan)

    # TÜİK tahminleri
    t2023 = row['y2023']
    t2024 = row['y2024']
    t2025 = row['y2025']

    # Sapma (gerçek - tahmin): pozitif = TÜİK düşük tahmin etmiş
    e2023 = g2023 - t2023  # 2023 baz yılı → sapma teorik 0, ama ADNKS revizyonu olabilir
    e2024 = g2024 - t2024
    e2025 = g2025 - t2025

    # Yıllık ortalama bias (2024 ve 2025 sapmasından)
    # 2024: 1 yıl ileri, 2025: 2 yıl ileri
    # bias_per_year = ortalama(e2024/1, e2025/2)
    b1 = e2024        # 1 yıl uzaklıkta sapma
    b2 = e2025 / 2    # 2 yıl uzaklıkta, yıllık bileşen
    bias_annual = (b1 + b2) / 2

    # Güncellenmiş projeksiyon: TÜİK_tahmini + bias * yıl_uzaklığı
    # Yıl uzaklığı: 2026→3, 2027→4, 2028→5, 2029→6, 2030→7 (2023 baz yılından)
    y2026_upd = int(round(row['y2026'] + bias_annual * 3))
    y2027_upd = int(round(row['y2027'] + bias_annual * 4))
    y2028_upd = int(round(row['y2028'] + bias_annual * 5))
    y2029_upd = int(round(row['y2029'] + bias_annual * 6))
    y2030_upd = int(round(row['y2030'] + bias_annual * 7))

    results.append({
        'key': k,
        'İl': il_adi,
        # Gerçek ADNKS
        'gercek_2023': int(g2023) if pd.notna(g2023) else None,
        'gercek_2024': int(g2024) if pd.notna(g2024) else None,
        'gercek_2025': int(g2025) if pd.notna(g2025) else None,
        # TÜİK orijinal tahminler
        'tuik_2024': int(t2024),
        'tuik_2025': int(t2025),
        'tuik_2026': int(row['y2026']),
        'tuik_2027': int(row['y2027']),
        'tuik_2028': int(row['y2028']),
        'tuik_2029': int(row['y2029']),
        'tuik_2030': int(row['y2030']),
        # Sapma ve bias
        'sapma_2024': int(round(e2024)),
        'sapma_2025': int(round(e2025)),
        'bias_yillik': round(bias_annual, 1),
        # Güncellenmiş projeksiyon
        'upd_2026': y2026_upd,
        'upd_2027': y2027_upd,
        'upd_2028': y2028_upd,
        'upd_2029': y2029_upd,
        'upd_2030': y2030_upd,
    })

df_res = pd.DataFrame(results)
print(f"Hesaplama tamamlandı: {len(df_res)} il")
print("\nÖrnek sonuçlar (İstanbul, Ankara, Adana):")
ornekler = df_res[df_res['İl'].isin(['İstanbul','Ankara','Adana'])]
print(ornekler[['İl','gercek_2025','tuik_2025','sapma_2025','bias_yillik','upd_2030','tuik_2030']].to_string(index=False))

# ─── 3. EXCEL ÇIKTI ──────────────────────────────────────────────────────
print("\nExcel dosyası oluşturuluyor...")

wb = Workbook()

# ── RENK PALETİ ──
R_BASLIK   = "1F4E79"   # Koyu mavi
R_ALTBAS   = "2E75B6"   # Orta mavi
R_GERCEK   = "D6E4F0"   # Açık mavi (gerçek veri)
R_TUIK     = "FFF2CC"   # Sarı (TÜİK orijinal)
R_UPD      = "E2EFDA"   # Açık yeşil (güncellenmiş)
R_SAPMA_P  = "E2EFDA"   # Yeşil (pozitif sapma)
R_SAPMA_N  = "FCE4D6"   # Turuncu (negatif sapma)
R_TEK      = "FFFFFF"
R_CIFT     = "F5F9FC"

def thin_border():
    t = Side(style='thin', color='BFBFBF')
    return Border(left=t, right=t, top=t, bottom=t)

def header_cell(ws, row, col, value, bg=R_ALTBAS, color='FFFFFF', bold=True, wrap=False, size=10):
    c = ws.cell(row=row, column=col, value=value)
    c.font = Font(bold=bold, size=size, color=color, name='Arial')
    c.fill = PatternFill('solid', start_color=bg)
    c.alignment = Alignment(horizontal='center', vertical='center', wrap_text=wrap)
    c.border = thin_border()
    return c

def data_cell(ws, row, col, value, bg=R_TEK, bold=False, fmt='#,##0', align='right', color='000000'):
    c = ws.cell(row=row, column=col, value=value)
    c.font = Font(bold=bold, size=9, name='Arial', color=color)
    c.fill = PatternFill('solid', start_color=bg)
    c.number_format = fmt
    c.alignment = Alignment(horizontal=align, vertical='center')
    c.border = thin_border()
    return c

# ════════════════════════════════════════════════════════════
# SAYFA 1: ANA ÖZET TABLO
# ════════════════════════════════════════════════════════════
ws1 = wb.active
ws1.title = "Güncellenmiş Projeksiyon"

# Başlık
ws1.merge_cells('A1:N1')
ws1['A1'] = 'TÜRKİYE İL BAZLI NÜFUS PROJEKSİYONU — TÜİK ANA SENARYO (GERÇEK VERİYLE GÜNCELLENMİŞ)'
ws1['A1'].font = Font(bold=True, size=13, color='FFFFFF', name='Arial')
ws1['A1'].fill = PatternFill('solid', start_color=R_BASLIK)
ws1['A1'].alignment = Alignment(horizontal='center', vertical='center')
ws1.row_dimensions[1].height = 28

ws1.merge_cells('A2:N2')
ws1['A2'] = ('Kaynak: TÜİK ADNKS 2023-2025 (gerçek) | TÜİK Nüfus Projeksiyonları 2023-2100, Yayın No:53699, Ana Senaryo | '
             'Yöntem: Bias-Correction (2024-2025 sapması üzerinden yıllık hata düzeltme)')
ws1['A2'].font = Font(italic=True, size=8, color='404040', name='Arial')
ws1['A2'].fill = PatternFill('solid', start_color='DEEAF1')
ws1['A2'].alignment = Alignment(horizontal='center')
ws1.row_dimensions[2].height = 14

# Sütun grup başlıkları (satır 3)
grups = [
    (1, 2, '', R_BASLIK),
    (3, 5, 'GERÇEK ADNKS VERİSİ', R_ALTBAS),
    (6, 7, 'TÜİK ORİJİNAL TAHMİN', "C09000"),
    (8, 8, 'SAPMA ANALİZİ', "7030A0"),
    (9, 13, 'GÜNCELLENMİŞ TAHMİN (Bias-Corrected)', "375623"),
    (14, 14, 'FARK\n2025→2030', R_BASLIK),
]
for start, end, label, bg in grups:
    if start == end:
        header_cell(ws1, 3, start, label, bg=bg, size=9, wrap=True)
    else:
        ws1.merge_cells(start_row=3, start_column=start, end_row=3, end_column=end)
        header_cell(ws1, 3, start, label, bg=bg, size=9, wrap=True)
ws1.row_dimensions[3].height = 26

# Kolon başlıkları (satır 4)
cols = [
    'Sıra', 'İl',
    '2023\n(Gerçek)', '2024\n(Gerçek)', '2025\n(Gerçek)',
    '2024\n(TÜİK)', '2025\n(TÜİK)',
    'Yıllık\nBias',
    '2026', '2027', '2028', '2029', '2030',
    'Büyüme\n%'
]
col_bgs = [R_BASLIK,R_BASLIK,
           R_ALTBAS,R_ALTBAS,R_ALTBAS,
           "C09000","C09000",
           "7030A0",
           "375623","375623","375623","375623","375623",
           R_BASLIK]
for ci, (h, bg) in enumerate(zip(cols, col_bgs), 1):
    header_cell(ws1, 4, ci, h, bg=bg, size=9, wrap=True)
ws1.row_dimensions[4].height = 30

# Veri satırları
for i, (_, row) in enumerate(df_res.iterrows(), 1):
    r = i + 4
    bg = R_TEK if i % 2 == 1 else R_CIFT

    data_cell(ws1, r, 1, i, bg=bg, fmt='0', align='center')
    c = ws1.cell(row=r, column=2, value=row['İl'])
    c.font = Font(bold=True, size=9, name='Arial')
    c.fill = PatternFill('solid', start_color=bg)
    c.alignment = Alignment(horizontal='left', vertical='center', indent=1)
    c.border = thin_border()

    # Gerçek ADNKS
    for ci, col in enumerate([('gercek_2023',3),('gercek_2024',4),('gercek_2025',5)]):
        data_cell(ws1, r, col[1], row[col[0]], bg=R_GERCEK, bold=(col[1]==5))

    # TÜİK orijinal
    data_cell(ws1, r, 6, row['tuik_2024'], bg=R_TUIK)
    data_cell(ws1, r, 7, row['tuik_2025'], bg=R_TUIK)

    # Bias
    bias = row['bias_yillik']
    bias_bg = R_SAPMA_P if bias >= 0 else R_SAPMA_N
    bias_color = "375623" if bias >= 0 else "C00000"
    data_cell(ws1, r, 8, bias, bg=bias_bg, fmt='+#,##0;-#,##0;0', color=bias_color)

    # Güncellenmiş tahminler
    for ci, col in enumerate(['upd_2026','upd_2027','upd_2028','upd_2029','upd_2030'], 9):
        data_cell(ws1, r, ci, row[col], bg=R_UPD, bold=(ci==13))

    # Büyüme formülü: (M - E) / E
    g_col_upd = get_column_letter(13)
    g_col_bas = get_column_letter(5)
    formula = f'=({g_col_upd}{r}-{g_col_bas}{r})/{g_col_bas}{r}'
    c = ws1.cell(row=r, column=14, value=formula)
    c.font = Font(size=9, name='Arial')
    c.fill = PatternFill('solid', start_color=bg)
    c.number_format = '+0.0%;-0.0%;0.0%'
    c.alignment = Alignment(horizontal='center', vertical='center')
    c.border = thin_border()
    ws1.row_dimensions[r].height = 15

# Toplam satırı
total_r = len(df_res) + 5
ws1.cell(total_r, 1, value='').fill = PatternFill('solid', start_color=R_BASLIK)
ws1.cell(total_r, 1).border = thin_border()
c = ws1.cell(total_r, 2, value='TÜRKİYE TOPLAMI')
c.font = Font(bold=True, color='FFFFFF', name='Arial', size=10)
c.fill = PatternFill('solid', start_color=R_BASLIK)
c.alignment = Alignment(horizontal='left', indent=1)
c.border = thin_border()

sums = {
    3: 'gercek_2023', 4: 'gercek_2024', 5: 'gercek_2025',
    6: 'tuik_2024', 7: 'tuik_2025',
    9: 'upd_2026', 10: 'upd_2027', 11: 'upd_2028', 12: 'upd_2029', 13: 'upd_2030'
}
for ci, col in sums.items():
    c = ws1.cell(total_r, ci, value=int(df_res[col].sum()))
    c.font = Font(bold=True, color='FFFFFF', name='Arial', size=9)
    c.fill = PatternFill('solid', start_color=R_BASLIK)
    c.number_format = '#,##0'
    c.alignment = Alignment(horizontal='right')
    c.border = thin_border()

# Bias toplam hücresini boş bırak
for ci in [8]:
    c = ws1.cell(total_r, ci)
    c.fill = PatternFill('solid', start_color=R_BASLIK)
    c.border = thin_border()

# Büyüme formülü toplam
gl = get_column_letter(13); gl5 = get_column_letter(5)
c = ws1.cell(total_r, 14, value=f'=({gl}{total_r}-{gl5}{total_r})/{gl5}{total_r}')
c.font = Font(bold=True, color='FFFFFF', name='Arial', size=9)
c.fill = PatternFill('solid', start_color=R_BASLIK)
c.number_format = '+0.0%;-0.0%;0.0%'
c.alignment = Alignment(horizontal='center')
c.border = thin_border()
ws1.row_dimensions[total_r].height = 18

# Sütun genişlikleri
ws1.column_dimensions['A'].width = 5
ws1.column_dimensions['B'].width = 20
for col in ['C','D','E','F','G']:
    ws1.column_dimensions[col].width = 13
ws1.column_dimensions['H'].width = 10
for col in ['I','J','K','L','M']:
    ws1.column_dimensions[col].width = 13
ws1.column_dimensions['N'].width = 10

# ════════════════════════════════════════════════════════════
# SAYFA 2: SAPMA ANALİZİ
# ════════════════════════════════════════════════════════════
ws2 = wb.create_sheet("Sapma Analizi")

ws2.merge_cells('A1:H1')
ws2['A1'] = 'TÜİK TAHMİN SAPMA ANALİZİ — 2024 ve 2025 Yılları (Gerçek ADNKS - TÜİK Tahmini)'
ws2['A1'].font = Font(bold=True, size=12, color='FFFFFF', name='Arial')
ws2['A1'].fill = PatternFill('solid', start_color="7030A0")
ws2['A1'].alignment = Alignment(horizontal='center')
ws2.row_dimensions[1].height = 22

hdrs2 = ['İl', 'TÜİK 2024', 'Gerçek 2024', 'Sapma 2024',
         'TÜİK 2025', 'Gerçek 2025', 'Sapma 2025', 'Yıllık Bias']
for ci, h in enumerate(hdrs2, 1):
    header_cell(ws2, 2, ci, h, bg="7030A0", size=9, wrap=True)
ws2.row_dimensions[2].height = 20

for i, (_, row) in enumerate(df_res.iterrows(), 1):
    r = i + 2
    bg = R_TEK if i % 2 == 1 else R_CIFT
    c = ws2.cell(r, 1, value=row['İl'])
    c.font = Font(bold=True, size=9, name='Arial')
    c.fill = PatternFill('solid', start_color=bg)
    c.border = thin_border()

    data_cell(ws2, r, 2, row['tuik_2024'], bg=R_TUIK)
    data_cell(ws2, r, 3, row['gercek_2024'], bg=R_GERCEK)

    s24 = row['sapma_2024']
    data_cell(ws2, r, 4, s24, bg=R_SAPMA_P if s24>=0 else R_SAPMA_N,
              fmt='+#,##0;-#,##0;0', color="375623" if s24>=0 else "C00000")

    data_cell(ws2, r, 5, row['tuik_2025'], bg=R_TUIK)
    data_cell(ws2, r, 6, row['gercek_2025'], bg=R_GERCEK)

    s25 = row['sapma_2025']
    data_cell(ws2, r, 7, s25, bg=R_SAPMA_P if s25>=0 else R_SAPMA_N,
              fmt='+#,##0;-#,##0;0', color="375623" if s25>=0 else "C00000")

    bias = row['bias_yillik']
    data_cell(ws2, r, 8, bias, bg=R_SAPMA_P if bias>=0 else R_SAPMA_N,
              fmt='+#,##0;-#,##0;0', color="375623" if bias>=0 else "C00000")
    ws2.row_dimensions[r].height = 15

for col, w in zip(['A','B','C','D','E','F','G','H'],[20,13,13,13,13,13,13,12]):
    ws2.column_dimensions[col].width = w

# ════════════════════════════════════════════════════════════
# SAYFA 3: ML FORMATI (UZUN FORMAT)
# ════════════════════════════════════════════════════════════
ws3 = wb.create_sheet("ML Formatı")

ws3.merge_cells('A1:E1')
ws3['A1'] = 'İL BAZLI NÜFUS — UZUN FORMAT (Makine Öğrenmesi İçin)'
ws3['A1'].font = Font(bold=True, size=11, color='FFFFFF', name='Arial')
ws3['A1'].fill = PatternFill('solid', start_color="375623")
ws3['A1'].alignment = Alignment(horizontal='center')
ws3.row_dimensions[1].height = 20

for ci, h in enumerate(['Yıl','İl','Nüfus','Veri Türü','Kaynak'], 1):
    header_cell(ws3, 2, ci, h, bg="375623", size=9)
ws3.row_dimensions[2].height = 18

ml_rows = []
for _, row in df_res.iterrows():
    ml_rows += [
        (2023, row['İl'], row['gercek_2023'], 'Gerçek', 'TÜİK ADNKS 2023'),
        (2024, row['İl'], row['gercek_2024'], 'Gerçek', 'TÜİK ADNKS 2024'),
        (2025, row['İl'], row['gercek_2025'], 'Gerçek', 'TÜİK ADNKS 2025'),
        (2026, row['İl'], row['upd_2026'], 'Güncel Tahmin', 'TÜİK Ana Senaryo + Bias-Correction'),
        (2027, row['İl'], row['upd_2027'], 'Güncel Tahmin', 'TÜİK Ana Senaryo + Bias-Correction'),
        (2028, row['İl'], row['upd_2028'], 'Güncel Tahmin', 'TÜİK Ana Senaryo + Bias-Correction'),
        (2029, row['İl'], row['upd_2029'], 'Güncel Tahmin', 'TÜİK Ana Senaryo + Bias-Correction'),
        (2030, row['İl'], row['upd_2030'], 'Güncel Tahmin', 'TÜİK Ana Senaryo + Bias-Correction'),
    ]

renk_ml = {'Gerçek': R_GERCEK, 'Güncel Tahmin': R_UPD}
for ri, (yil, il, nufus, tur, kaynak) in enumerate(ml_rows, 3):
    bg = renk_ml.get(tur, R_TEK)
    data_cell(ws3, ri, 1, yil, bg=bg, fmt='0', align='center')
    c = ws3.cell(ri, 2, value=il)
    c.font = Font(size=9, name='Arial')
    c.fill = PatternFill('solid', start_color=bg)
    c.border = thin_border()
    data_cell(ws3, ri, 3, int(nufus), bg=bg)
    c = ws3.cell(ri, 4, value=tur)
    c.font = Font(size=9, name='Arial', italic=True)
    c.fill = PatternFill('solid', start_color=bg)
    c.border = thin_border()
    c = ws3.cell(ri, 5, value=kaynak)
    c.font = Font(size=8, name='Arial', italic=True, color='595959')
    c.fill = PatternFill('solid', start_color=bg)
    c.border = thin_border()
    ws3.row_dimensions[ri].height = 14

for col, w in zip(['A','B','C','D','E'],[7,20,15,18,38]):
    ws3.column_dimensions[col].width = w

# ─── KAYDET ───────────────────────────────────────────────────────────────
output = '/mnt/user-data/outputs/Turkiye_Nufus_Projeksiyon_Guncellenmis.xlsx'
wb.save(output)
print(f"\n✓ Excel kaydedildi: {output}")
print(f"  Sayfa 1: {len(df_res)} il, özet tablo (2023-2030)")
print(f"  Sayfa 2: Sapma analizi")
print(f"  Sayfa 3: {len(ml_rows)} satır ML formatı")
