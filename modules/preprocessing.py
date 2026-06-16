import re
import pandas as pd
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException

DetectorFactory.seed = 0

_slang_dict = None

def load_slang_dict():
    global _slang_dict
    if _slang_dict is None:
        url_kamus = 'https://raw.githubusercontent.com/nasalsabila/kamus-alay/master/colloquial-indonesian-lexicon.csv'
        df_kamus = pd.read_csv(url_kamus)
        _slang_dict = dict(zip(df_kamus['slang'], df_kamus['formal']))
    return _slang_dict

def step_case_folding(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['clean_text'] = df['full_text'].astype(str).str.lower()
    return df

def step_remove_url(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['clean_text'] = df['clean_text'].apply(
        lambda x: re.sub(r'http\S+|www\S+|https\S+', '', str(x), flags=re.MULTILINE)
    )
    return df

def step_clean_text(df: pd.DataFrame) -> pd.DataFrame:
    def clean(text):
        text = str(text)
        text = re.sub(r'\@\w+', '', text)
        text = re.sub(r'#', '', text)
        text = re.sub(r'[^a-z\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    df = df.copy()
    df['clean_text'] = df['clean_text'].apply(clean)
    return df

def step_normalisasi(df: pd.DataFrame) -> pd.DataFrame:
    slang_dict = load_slang_dict()
    def normalisasi(text):
        kata_kata = str(text).split()
        return ' '.join([slang_dict.get(k, k) for k in kata_kata])
    df = df.copy()
    df['clean_text'] = df['clean_text'].apply(normalisasi)
    return df

def step_filter(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['clean_text'] = df['clean_text'].replace(r'^\s*$', pd.NA, regex=True)
    df = df.dropna(subset=['clean_text'])
    df = df.drop_duplicates(subset=['clean_text'], keep='first')
    df = df[df['clean_text'].astype(str).str.split().str.len() >= 3]

    def is_indonesian(text):
        try:
            return detect(str(text)) == 'id'
        except LangDetectException:
            return False

    mask = df['clean_text'].apply(is_indonesian)
    df = df[mask].reset_index(drop=True)
    return df

def run_preprocessing(df_raw: pd.DataFrame, progress_callback=None) -> pd.DataFrame:
    steps = [
        (step_case_folding,  "Case folding (lowercase)..."),
        (step_remove_url,    "Menghapus URL..."),
        (step_clean_text,    "Membersihkan mention, emoji, karakter khusus..."),
        (step_normalisasi,   "Normalisasi kata slang..."),
        (step_filter,        "Filtering bahasa & duplikat..."),
    ]
    total = len(steps)
    df = df_raw.copy()

    for i, (fn, pesan) in enumerate(steps):
        if progress_callback:
            progress_callback(i, total, pesan)
        df = fn(df)

    if progress_callback:
        progress_callback(total, total, "Preprocessing selesai!")

    return df