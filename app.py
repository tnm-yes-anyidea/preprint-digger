from flask import Flask, request, jsonify
from flask_cors import CORS
import urllib.request
import xml.etree.ElementTree as ET
from sentence_transformers import SentenceTransformer
import numpy as np

app = Flask(__name__)
CORS(app)

# Load the tiny, CPU-optimized embedding model (~80MB footprint)
print("Loading semantic model into RAM...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model ready!")

# Cache to store today's papers and their mathematical vectors
db_cache = {'category': None, 'papers': [], 'embeddings': None}

def fetch_daily_arxiv(category, max_results=150):
    url = f'http://export.arxiv.org/api/query?search_query=cat:{category}&sortBy=submittedDate&sortOrder=descending&max_results={max_results}'
    with urllib.request.urlopen(url) as response:
        xml = response.read()

    root = ET.fromstring(xml)
    namespace = {'atom': 'http://www.w3.org/2005/Atom'}
    papers = []

    for entry in root.findall('atom:entry', namespace):
        papers.append({
            'id': entry.find('atom:id', namespace).text.split('/')[-1],
            'title': entry.find('atom:title', namespace).text.replace('\n', ' ').strip(),
            'summary': entry.find('atom:summary', namespace).text.replace('\n', ' ').strip(),
            'link': entry.find('atom:id', namespace).text,
            'authors': [a.find('atom:name', namespace).text for a in entry.findall('atom:author', namespace)]
        })
    return papers

@app.route('/api/radar', methods=['POST'])
def semantic_radar():
    data = request.json
    category = data.get('category', 'cs.LG')
    user_query = data.get('query', '')

    if not user_query:
        return jsonify({"error": "Please describe your research area."}), 400

    # 1. Fetch & Embed the daily batch (only runs once per category change)
    if db_cache['category'] != category or not db_cache['papers']:
        print(f"Fetching latest {category} papers from arXiv...")
        papers = fetch_daily_arxiv(category, 150)
        summaries = [p['summary'] for p in papers]

        print("Converting abstracts to semantic vectors...")
        embeddings = model.encode(summaries)

        db_cache['category'] = category
        db_cache['papers'] = papers
        db_cache['embeddings'] = embeddings

    # 2. Convert the user's plain-text research description into a vector
    query_vector = model.encode([user_query])[0]

    # 3. Calculate Cosine Similarity (How close the meanings are mathematically)
    q_norm = query_vector / np.linalg.norm(query_vector)
    db_norm = db_cache['embeddings'] / np.linalg.norm(db_cache['embeddings'], axis=1)[:, np.newaxis]
    similarities = np.dot(db_norm, q_norm)

    # 4. Grab the top 15 highest-scoring papers
    top_indices = np.argsort(similarities)[::-1][:15]

    results = []
    for idx in top_indices:
        paper = db_cache['papers'][idx].copy()
        # Convert similarity float to a percentage score
        paper['match_score'] = round(float(similarities[idx]) * 100, 1)
        results.append(paper)

    return jsonify(results)

if __name__ == '__main__':
    app.run(port=5000)
