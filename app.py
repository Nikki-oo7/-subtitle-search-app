from flask import Flask, render_template, request
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re

app = Flask(__name__)

# Load vectorizer and TF-IDF matrix
with open('tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

with open('tfidf_matrix.pkl', 'rb') as f:
    tfidf_matrix = pickle.load(f)

metadata = pd.read_csv('indexed_metadata.csv')

# Map lowercase video titles to their YouTube IDs
video_id_map = {
    'machine learning what is machine learning introduction to machine learning 2024 simplilearn': 'ukzFI9rgwfU',
    'introduction to artificial intelligence what is ai artificial intelligence tutorial simplilearn': 'FbxTVRfQFuI',
    'what is deep learning introduction to deep learning deep learning tutorial simplilearn': 'SSE4M0gcmvE'
}

def timestamp_to_seconds(timestamp):
    parts = timestamp.split(':')
    return int(float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2]))

def search_subtitles(query, top_n=5):
    query = query.lower()
    query_vector = vectorizer.transform([query])
    similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    top_indices = np.argsort(similarities)[::-1][:top_n]

    if all(similarities[top_indices] == 0):
        return pd.DataFrame()

    results = metadata.iloc[top_indices].copy()
    results['Similarity Score'] = similarities[top_indices]

    context_blocks = []
    jump_links = []
    thumbnails = []
    video_titles = []
    timestamps_readable = []

    for idx in top_indices:
        row = metadata.iloc[idx]
        video_title = row['Video Title'].strip().lower()
        video_id = video_id_map.get(video_title, '')
        seconds = timestamp_to_seconds(row['Start Time'])

        # Build jump link
        link = f"https://www.youtube.com/watch?v={video_id}&t={seconds}s" if video_id else ''
        jump_links.append(link)

        # Readable timestamp (e.g. 0:05:21)
        readable_time = row['Start Time']
        timestamps_readable.append(readable_time)

        # Thumbnail URL
        thumbnail = f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg" if video_id else ''
        thumbnails.append(thumbnail)

        # Capitalized video title
        video_titles.append(video_title.title())

        # Context block
        before = metadata.iloc[idx - 1]['Subtitle Text'] if idx > 0 else ''
        match_line = row['Subtitle Text']
        after = metadata.iloc[idx + 1]['Subtitle Text'] if idx + 1 < len(metadata) else ''
        highlighted = re.sub(f"({re.escape(query)})", r"<mark>\1</mark>", match_line, flags=re.IGNORECASE)
        context = f"{before}\n{highlighted}\n{after}"
        context_blocks.append(context)

    # Add new polished fields
    results['Context Block'] = context_blocks
    results['Jump Link'] = jump_links
    results['Thumbnail'] = thumbnails
    results['Video Title Clean'] = video_titles
    results['Readable Time'] = timestamps_readable

    return results.sort_values(by='Similarity Score', ascending=False)

@app.route("/", methods=["GET", "POST"])
def index():
    results = None
    query = ""
    selected_count = 5  # Default value
    if request.method == "POST":
        query = request.form.get("query")
        selected_count = int(request.form.get("count", 5))  # Use dropdown value
        if query:
            results_df = search_subtitles(query, top_n=selected_count)
            results = results_df.to_dict(orient="records") if not results_df.empty else []
    return render_template("index.html", results=results, query=query, selected_count=selected_count)

if __name__ == "__main__":
    app.run(debug=True)
