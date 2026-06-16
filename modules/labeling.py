import pandas as pd

_pos_dict = None
_neg_dict = None

def load_inset():
    global _pos_dict, _neg_dict
    if _pos_dict is None or _neg_dict is None:
        url_positif = 'https://raw.githubusercontent.com/fajri91/InSet/master/positive.tsv'
        url_negatif = 'https://raw.githubusercontent.com/fajri91/InSet/master/negative.tsv'
        lexicon_pos = pd.read_csv(url_positif, sep='\t')
        lexicon_neg = pd.read_csv(url_negatif, sep='\t')
        _pos_dict = dict(zip(lexicon_pos.iloc[:, 0], lexicon_pos.iloc[:, 1]))
        _neg_dict = dict(zip(lexicon_neg.iloc[:, 0], lexicon_neg.iloc[:, 1]))
    return _pos_dict, _neg_dict

def hitung_skor(text: str, pos_dict: dict, neg_dict: dict) -> pd.Series:
    kata_kata = str(text).split()
    pos_score = 0
    neg_score = 0

    for kata in kata_kata:
        if kata in pos_dict:
            pos_score += pos_dict[kata]
        elif kata in neg_dict:
            neg_score += neg_dict[kata]

    total = pos_score + neg_score
    if total > 0:
        label = 'Positif'
    elif total < 0:
        label = 'Negatif'
    else:
        label = 'Netral'

    return pd.Series([pos_score, neg_score, total, label])

def run_labeling(df: pd.DataFrame, progress_callback=None) -> pd.DataFrame:
    if progress_callback:
        progress_callback(0, 2, "Memuat lexicon InSet...")

    pos_dict, neg_dict = load_inset()

    if progress_callback:
        progress_callback(1, 2, "Menghitung skor sentimen setiap teks...")

    df = df.copy()
    df[['skor_positif', 'skor_negatif', 'total_skor', 'label']] = df['clean_text'].apply(
        lambda x: hitung_skor(x, pos_dict, neg_dict)
    )

    if progress_callback:
        progress_callback(2, 2, "Pelabelan selesai!")

    return df