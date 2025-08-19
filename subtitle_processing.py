import os
import re
import pickle
import webvtt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# 1) Video ID 

video_id_map = {
    'machine learning what is machine learning introduction to machine learning 2024 simplilearn': 'ukzFI9rgwfU',
    'introduction to artificial intelligence what is ai artificial intelligence tutorial simplilearn': 'SSE4M0gcmvE',
    'what is deep learning introduction to deep learning deep learning tutorial simplilearn': 'FbxTVRfQFuI'
}

# 2) File list
videos = [
    (
        'Machine Learning What Is Machine Learning Introduction To Machine Learning 2024 Simplilearn',
        'Machine Learning.vtt'
    ),
    (
        'Introduction To Artificial Intelligence What Is AI Artificial Intelligence Tutorial Simplilearn',
        'Artificial Intelligence.vtt'
    ),
    (
        'What is Deep Learning Introduction to Deep Learning Deep Learning Tutorial Simplilearn',
        'Deep Learning.vtt'
    )
]

_space_re = re.compile(r'\s+')

def normalize_minimal(text: str) -> str:
    if not isinstance(text, str):
        return ''
    t = text.lower()
    t = _space_re.sub(' ', t).strip()
    return t

# DF (with YT ID)
all_subtitles = []

for raw_title, file_path in videos:
    cleaned_title = re.sub(r'\s+', ' ', raw_title).strip().lower()
    youtube_id = video_id_map.get(cleaned_title, '')

    if not youtube_id:
        print(f"⚠️ YouTube ID not found for title: {cleaned_title}")
        continue

    for caption in webvtt.read(file_path):
        start = caption.start
        end = caption.end
        text_raw = caption.text or ''
        all_subtitles.append([start, end, text_raw, cleaned_title, youtube_id])

# DF with RAW text
df = pd.DataFrame(
    all_subtitles,
    columns=['Start Time', 'End Time', 'Subtitle Text Raw', 'Video Title', 'YouTube ID']
)

#  CLEAN column for indexing
df['Subtitle Text Clean'] = df['Subtitle Text Raw'].apply(normalize_minimal)

df['Subtitle Text'] = df['Subtitle Text Clean']

before_drop = len(df)
df = df[df['Subtitle Text Clean'].str.len() > 0].copy()
after_drop = len(df)

df.sort_values(by=['YouTube ID', 'Start Time'], inplace=True, ignore_index=True)
df.to_csv('cleaned_subtitles.csv', index=False)

# 5) TF-IDF fit 
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),  
    sublinear_tf=True,
    lowercase=True,       
    min_df=1
)
tfidf_matrix = vectorizer.fit_transform(df['Subtitle Text Clean'])

with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

with open('tfidf_matrix.pkl', 'wb') as f:
    pickle.dump(tfidf_matrix, f)

# 6) Export metadata 
metadata = df[['Subtitle Text Raw', 'Subtitle Text Clean', 'Subtitle Text',
               'Start Time', 'End Time', 'Video Title', 'YouTube ID']]
metadata.to_csv('indexed_metadata.csv', index=False)

print(" Files saved.")
print(f"   Rows read:        {len(all_subtitles)}")
print(f"   Rows after clean: {after_drop} (dropped {before_drop - after_drop} empty rows)")
print(f"   Unique videos:    {df['YouTube ID'].nunique()}")
print(f"   TF-IDF shape:     {tfidf_matrix.shape}")
