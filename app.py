from flask import Flask, render_template, request
import os, html, pandas as pd, pickle, re
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

app = Flask(__name__)

# Health check (for HF Spaces readiness)
@app.get("/health")
def health():
    return {"ok": True}, 200

# Path

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VECT_PATH = os.path.join(BASE_DIR, 'tfidf_vectorizer.pkl')
MATR_PATH = os.path.join(BASE_DIR, 'tfidf_matrix.pkl')
META_PATH = os.path.join(BASE_DIR, 'indexed_metadata.csv')

# Load artifacts

with open(VECT_PATH, 'rb') as f:
    vectorizer = pickle.load(f)
with open(MATR_PATH, 'rb') as f:
    tfidf_matrix = pickle.load(f)
metadata = pd.read_csv(META_PATH)

# columns available (supports transition)
CLEAN_COL = 'Subtitle Text Clean' if 'Subtitle Text Clean' in metadata.columns else 'Subtitle Text'
RAW_COL   = 'Subtitle Text Raw'   if 'Subtitle Text Raw'   in metadata.columns else CLEAN_COL

# Utils
_space_re = re.compile(r'\s+')

def normalize_minimal(text: str) -> str:
    """Lowercase + collapse spaces. No stopword removal. No punctuation strip."""
    if not isinstance(text, str):
        return ''
    t = text.lower()
    t = _space_re.sub(' ', t).strip()
    return t

def highlight_exact_phrase(text: str, phrase: str) -> str:
    """
    Escape HTML then highlight the exact query as a whole word/phrase (case-insensitive).
    Prevents 'ai' from highlighting inside 'domain'.
    """
    safe = html.escape(text or '')
    p = (phrase or '').strip()
    if not p:
        return safe

    if ' ' not in p:
        pattern = re.compile(rf'(?<![A-Za-z0-9_]){re.escape(p)}(?![A-Za-z0-9_])', re.IGNORECASE)
    else:
        parts = [re.escape(w) for w in re.split(r'\s+', p)]
        inner = r'\s+'.join(parts)
        pattern = re.compile(rf'(?<![A-Za-z0-9_]){inner}(?![A-Za-z0-9_])', re.IGNORECASE)

    return pattern.sub(r'<mark>\g<0></mark>', safe)

def timestamp_to_seconds(timestamp):
    try:
        h, m, s = timestamp.split(':')
        return int(h) * 3600 + int(m) * 60 + float(s)
    except Exception:
        return 0

# Core search
def search_subtitles(query):
    """
    Returns (results_df, total_hits)
    - Results are ALL positive-similarity matches, sorted by relevance (desc).
    - Context uses RAW text; highlight is exact-phrase on the match line only.
    """
    raw_query = (query or '').strip()
    q_norm = normalize_minimal(raw_query)
    if not q_norm:
        return pd.DataFrame(), 0

    sims = cosine_similarity(vectorizer.transform([q_norm]), tfidf_matrix).flatten()
    order = np.argsort(sims)[::-1]

    # Keep only positive-similarity hits
    eps = 1e-12
    pos_order = order[sims[order] > eps]
    total_hits = int(pos_order.size)
    if total_hits == 0:
        return pd.DataFrame(), 0

    # Build all results 
    results = metadata.iloc[pos_order].copy()

    links, thumbs, titles, times, contexts, scores = [], [], [], [], [], []
    for row_idx in pos_order:
        row = metadata.iloc[row_idx]
        vid = row.get('YouTube ID', '')
        start_time = row.get('Start Time', '0:00:00')
        secs = max(0, timestamp_to_seconds(start_time) - 2)

        links.append(f"https://www.youtube.com/watch?v={vid}&t={secs}s" if vid else '')
        times.append(start_time)
        thumbs.append(f"https://img.youtube.com/vi/{vid}/hqdefault.jpg" if vid else '')
        titles.append((row.get('Video Title') or '').title())
        scores.append(float(sims[row_idx]))

        # RAW context with same-video guard
        before_raw = after_raw = ''
        if row_idx > 0 and metadata.iloc[row_idx - 1].get('YouTube ID', '') == vid:
            before_raw = metadata.iloc[row_idx - 1].get(RAW_COL, '')
        if row_idx + 1 < len(metadata) and metadata.iloc[row_idx + 1].get('YouTube ID', '') == vid:
            after_raw = metadata.iloc[row_idx + 1].get(RAW_COL, '')

        match_raw = row.get(RAW_COL, '')
        match_hl = highlight_exact_phrase(match_raw, raw_query)

        context = f"{html.escape(before_raw or '')}\n{match_hl}\n{html.escape(after_raw or '')}"
        contexts.append(context)

    results['Context Block']     = contexts
    results['Jump Link']         = links
    results['Thumbnail']         = thumbs
    results['Video Title Clean'] = titles
    results['Readable Time']     = times
    results['Similarity Score']  = scores  # handy for debugging/thresholds

    return results, total_hits

# Routes
@app.route("/", methods=["GET", "POST"])
def index():
    results = None
    query = ""
    total_hits = 0
    if request.method == "POST":
        query = request.form.get("query", "")
        if query.strip():
            df, total_hits = search_subtitles(query)
            results = df.to_dict(orient="records") if not df.empty else []
    return render_template("index.html", results=results, query=query, total_hits=total_hits)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port, debug=False)
