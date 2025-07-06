import webvtt
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

nltk.download('stopwords')

# Adjust these paths to match your downloaded files
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
for video_title, file_path in videos:
    for caption in webvtt.read(file_path):
        all_subtitles.append([
            caption.start,
            caption.end,
            caption.text,
            video_title
        ])

df = pd.DataFrame(all_subtitles, columns=['Start Time', 'End Time', 'Subtitle Text', 'Video Title'])

# Clean subtitles
df['Video Title'] = df['Video Title'].str.replace(r'\s+', ' ', regex=True).str.strip().str.lower()
df['Subtitle Text'] = df['Subtitle Text'].str.lower()
df['Subtitle Text'] = df['Subtitle Text'].str.replace(r'[^\w\s]', '', regex=True)
df['Subtitle Text'] = df['Subtitle Text'].str.replace(r'\s+', ' ', regex=True).str.strip()

stop_words = set(stopwords.words('english'))
df['Subtitle Text'] = df['Subtitle Text'].apply(lambda x: ' '.join([w for w in x.split() if w not in stop_words]))

# Save cleaned subtitles
df.to_csv('cleaned_subtitles.csv', index=False)

# TF-IDF
df['Subtitle Text'] = df['Subtitle Text'].fillna('')
texts = df['Subtitle Text'].tolist()
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(texts)

with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

with open('tfidf_matrix.pkl', 'wb') as f:
    pickle.dump(tfidf_matrix, f)

metadata = df[['Subtitle Text', 'Start Time', 'End Time', 'Video Title']]
metadata.to_csv('indexed_metadata.csv', index=False)

print("✅ Processing complete. Files saved.")

