import pandas as pd
import re

# ============================================================
# PARSER: TÜİK pipe-delimited dosyaları (hastane, yatak, personel)
# ============================================================
def parse_tuik_pipe(filepath):
    with open(filepath, 'rb') as f:
        raw = f.read()
    text = raw.decode('utf-8-sig')
    lines = text.split('\n')

    # 2. satır: il isimleri (index 3'ten itibaren, "İlAdı-PlateNo" formatında)
    il_satiri = lines[1].split('|')
    iller = [re.sub(r'-\d+$', '', x.strip()).strip() for x in il_satiri[3:] if x.strip()]

    records = []
    current_kat = ''
    current_kur = ''

    for line in lines[4:]:
        parts = line.split('|')
        if len(parts) < 4:
            continue
        yil_str = parts[2].strip()
        if not yil_str.isdigit():
            continue
        yil = int(yil_str)

        # Kategori ve kurum yalnızca ilk satırda dolu gelir; sonraki yıllar boş gelir
        if parts[0].strip():
            current_kat = parts[0].strip()
        if parts[1].strip():
            current_kur = parts[1].strip()

        for il, deger in zip(iller, parts[3:3 + len(iller)]):
            d = deger.strip()
            try:
                val = float(d) if d else None
            except ValueError:
                val = None
            records.append({
                'il_adi': il,
                'yil': yil,
                'ham_kat': current_kat,
                'ham_kur': current_kur,
                'deger': val
            })

    return pd.DataFrame(records)


# ============================================================
# PARSER: Nüfus (nufüs.csv + nufus2324.csv)
# ============================================================
def parse_nufus():
    # --- 2002-2022: nufüs.csv ---
    with open('nufüs.csv', 'rb') as f:
        raw = f.read()
    text = raw.decode('utf-8-sig')
    lines = text.split('\r\n')

    # 4. satır yılları içerir
    yillar = [
        int(x.strip().replace(' ', ''))
        for x in lines[3].split(';')
        if x.strip().replace(' ', '').isdigit()
    ]

    records = []
    for line in lines[4:]:
        parts = line.split(';')
        il = parts[0].strip()
        if not il:
            continue
        if 'Toplam' in il or 'Kaynak' in il or 'Source' in il or il.startswith('('):
            continue
        # Değerler 3. indeksten başlar (il, boş, boş, değer1, değer2, ...)
        for yil, deger in zip(yillar, parts[3:3 + len(yillar)]):
            d = deger.strip().replace(' ', '').replace('\xa0', '')
            try:
                val = float(d) if d else None
            except ValueError:
                val = None
            records.append({'il_adi': il, 'yil': yil, 'deger': val})

    df1 = pd.DataFrame(records)

    # --- 2023-2024: nufus2324.csv ---
    with open('nufus2324.csv', 'rb') as f:
        raw = f.read()
    text = raw.decode('utf-8-sig')
    lines2 = text.split('\r\n')

    records2 = []
    for line in lines2[1:]:
        parts = line.split(';')
        if len(parts) < 3 or not parts[0].strip():
            continue
        il = parts[0].strip()
        for yil_str, deger in zip(['2023', '2024'], parts[1:3]):
            d = deger.strip().replace(' ', '').replace('\xa0', '')
            try:
                val = float(d) if d else None
            except ValueError:
                val = None
            records2.append({'il_adi': il, 'yil': int(yil_str), 'deger': val})

    df2 = pd.DataFrame(records2)
    return pd.concat([df1, df2], ignore_index=True)


# ============================================================
# EŞLEME TABLOLARI
# ============================================================

# Ham kurum adı -> standart kurum etiketi
kurum_map = {
    '1. (Hastane (Sağlık Bakanlığı))':                       'devlet',
    '2. (Hastane (Özel))':                                   'ozel',
    '3. (Hastane (Üniversite))':                             'universite',
    '1. (Uzman Hekim) ve 1. (Sağlık Bakanlığı)':            'devlet',
    '1. (Uzman Hekim) ve 2. (Üniversite)':                  'universite',
    '1. (Uzman Hekim) ve 3. (Özel)':                        'ozel',
    '2. (Pratisyen Hekim) ve 1. (Sağlık Bakanlığı)':        'devlet',
    '2. (Pratisyen Hekim) ve 2. (Üniversite)':              'universite',
    '2. (Pratisyen Hekim) ve 3. (Özel)':                    'ozel',
    '3. (Asistan Hekim) ve 1. (Sağlık Bakanlığı)':          'devlet',
    '3. (Asistan Hekim) ve 2. (Üniversite)':                'universite',
    '3. (Asistan Hekim) ve 3. (Özel)':                      'ozel',
    '4. (Diş Hekimi) ve 1. (Sağlık Bakanlığı)':             'devlet',
    '4. (Diş Hekimi) ve 2. (Üniversite)':                   'universite',
    '4. (Diş Hekimi) ve 3. (Özel)':                         'ozel',
    '5. (Hemşire) ve 1. (Sağlık Bakanlığı)':                'devlet',
    '5. (Hemşire) ve 2. (Üniversite)':                      'universite',
    '5. (Hemşire) ve 3. (Özel)':                            'ozel',
    '6. (Ebe) ve 1. (Sağlık Bakanlığı)':                    'devlet',
    '6. (Ebe) ve 2. (Üniversite)':                          'universite',
    '6. (Ebe) ve 3. (Özel)':                                'ozel',
    '7. (Eczacı) ve 1. (Sağlık Bakanlığı)':                 'devlet',
    '7. (Eczacı) ve 2. (Üniversite)':                       'universite',
    '7. (Eczacı) ve 3. (Özel)':                             'ozel',
    '98. (Diğer Sağlık Personeli) ve 1. (Sağlık Bakanlığı)':'devlet',
    '98. (Diğer Sağlık Personeli) ve 2. (Üniversite)':      'universite',
    '98. (Diğer Sağlık Personeli) ve 3. (Özel)':            'ozel',
}


def get_veritipi_kategori(ham_kat, ham_kur):
    """Ham TÜİK kategori/kurum adlarını standart veri_tipi ve kategori etiketlerine dönüştürür."""
    if 'Uzman' in ham_kur:         return 'personel', 'uzman'
    if 'Pratisyen' in ham_kur:     return 'personel', 'pratisyen'
    if 'Asistan' in ham_kur:       return 'personel', 'asistan'
    if 'Diş Hekimi' in ham_kur:    return 'personel', 'dis'
    if 'Hemşire' in ham_kur:       return 'personel', 'hemsire'
    if 'Ebe' in ham_kur:           return 'personel', 'ebe'
    if 'Eczacı' in ham_kur:        return 'personel', 'eczaci'
    if 'Diğer Sağlık' in ham_kur:  return 'personel', 'diger'
    if 'Yatak' in ham_kat:         return 'yatak_sayisi', 'yatak'
    if 'Hastane' in ham_kat:       return 'hastane_sayisi', 'hastane'
    return None, None


# ============================================================
# ANA DÖNÜŞÜM
# ============================================================
all_rows = []

# Hastane, yatak ve personel dosyaları
dosyalar = [
    ('hastane.csv', False),
    ('yatak.csv',   True),   # yatak için veri_tipi zorla atanır
    ('personel.csv', False),
]

for dosya, is_yatak in dosyalar:
    df = parse_tuik_pipe(dosya)
    for _, r in df.iterrows():
        if is_yatak:
            vt, kat = 'yatak_sayisi', 'yatak'
        else:
            vt, kat = get_veritipi_kategori(r['ham_kat'], r['ham_kur'])
        kurum = kurum_map.get(r['ham_kur'], 'devlet')
        if vt:
            all_rows.append({
                'il_adı':    r['il_adi'],
                'yıl':       r['yil'],
                'değer':     r['deger'],
                'veri_tipi': vt,
                'kategori':  kat,
                'kurum':     kurum
            })

# Nüfus
df_nuf = parse_nufus()
for _, r in df_nuf.iterrows():
    all_rows.append({
        'il_adı':    r['il_adi'],
        'yıl':       r['yil'],
        'değer':     r['deger'],
        'veri_tipi': 'nufus',
        'kategori':  'nufus',
        'kurum':     'yok'
    })

# DataFrame oluştur ve sırala
df_out = pd.DataFrame(all_rows, columns=['il_adı', 'yıl', 'değer', 'veri_tipi', 'kategori', 'kurum'])
df_out = df_out.sort_values(['veri_tipi', 'kategori', 'kurum', 'yıl', 'il_adı']).reset_index(drop=True)

# Kaydet
df_out.to_csv('all_data2.csv', index=False, encoding='utf-8-sig')

print(f"all_data2.csv oluşturuldu — {len(df_out)} satır")
print(df_out.groupby(['veri_tipi', 'kategori', 'kurum']).size().reset_index(name='count').to_string())
