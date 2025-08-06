import webvtt
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

#  stopwords 
nltk.download('stopwords')

# Video ID mapping
video_id_map = {
    'machine learning what is machine learning introduction to machine learning 2024 simplilearn': 'ukzFI9rgwfU',
    'introduction to artificial intelligence what is ai artificial intelligence tutorial simplilearn': 'SSE4M0gcmvE',
    'what is deep learning introduction to deep learning deep learning tutorial simplilearn': 'FbxTVRfQFuI'
}

#  File path(remeber this path )
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

all_subtitles = []

# Processing
for raw_title, file_path in videos:
    cleaned_title = re.sub(r'\s+', ' ', raw_title).strip().lower()
    youtube_id = video_id_map.get(cleaned_title, '')

    if not youtube_id:
        print(f"⚠️ YouTube ID not found for title: {cleaned_title}")
        continue

    for caption in webvtt.read(file_path):
        all_subtitles.append([
            caption.start,
            caption.end,
            caption.text,
            cleaned_title,
            youtube_id
        ])

# ✅ DF with YT ID
df = pd.DataFrame(all_subtitles, columns=[
    'Start Time', 'End Time', 'Subtitle Text', 'Video Title', 'YouTube ID'
])

# Cleaning
df['Subtitle Text'] = df['Subtitle Text'].str.lower()
df['Subtitle Text'] = df['Subtitle Text'].str.replace(r'[^\w\s]', '', regex=True)
df['Subtitle Text'] = df['Subtitle Text'].str.replace(r'\s+', ' ', regex=True).str.strip()

stop_words = set(stopwords.words('english'))
df['Subtitle Text'] = df['Subtitle Text'].apply(
    lambda x: ' '.join([w for w in x.split() if w not in stop_words])
)

df.to_csv('cleaned_subtitles.csv', index=False)

#  for TF-IDF
df['Subtitle Text'] = df['Subtitle Text'].fillna('')
texts = df['Subtitle Text'].tolist()
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(texts)

#  TF-IDF model and matrix
with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

with open('tfidf_matrix.pkl', 'wb') as f:
    pickle.dump(tfidf_matrix, f)

#  metadata CSV with YT ID
metadata = df[['Subtitle Text', 'Start Time', 'End Time', 'Video Title', 'YouTube ID']]
metadata.to_csv('indexed_metadata.csv', index=False)

print("✅ Processing complete. Files saved.")
