# /// script
# dependencies = [
#     "flask",
#     "flask-cors",
#     "fastembed",
#     "sqlite-vec",
# ]
# ///
import sqlite3
import sqlite_vec
import urllib.request
import xml.etree.ElementTree as ET
import json
import threading
from flask import Flask, request, jsonify
from flask_cors import CORS
from fastembed import TextEmbedding

app = Flask(__name__)
CORS(app)

# 1. Load the ultra-light ONNX embedding model (sentence-transformers/all-MiniLM-L6-v2 has 384 dimensions)
print("Loading FastEmbed Model...")
model = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
print("Model Ready!")

# 2. Initialize SQLite with Vector Extension
def get_db():
    db = sqlite3.connect("arxiv_radar.db", check_same_thread=False)
    db.enable_load_extension(True)
    sqlite_vec.load(db)
    db.enable_load_extension(False)
    return db

db = get_db()
db.execute('''
    CREATE TABLE IF NOT EXISTS papers (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        arxiv_id TEXT UNIQUE,
        category TEXT,
        title TEXT,
        summary TEXT,
        authors TEXT,
        link TEXT,
        published TEXT
    )
''')
# sqlite-vec uses a virtual table to index the vectors
db.execute('''
    CREATE VIRTUAL TABLE IF NOT EXISTS vec_papers USING vec0(
        embedding float[384]
    )
''')
db.commit()

def fetch_and_store_arxiv(category, max_results=25):
    url = f'http://export.arxiv.org/api/query?search_query=cat:{category}&sortBy=submittedDate&sortOrder=descending&max_results={max_results}'
    with urllib.request.urlopen(url) as response:
        xml = response.read()

    root = ET.fromstring(xml)
    namespace = {'atom': 'http://www.w3.org/2005/Atom'}

    new_papers = []
    for entry in root.findall('atom:entry', namespace):
        arxiv_id = entry.find('atom:id', namespace).text.split('/')[-1]

        # Check if we already have this paper in our database
        if db.execute("SELECT 1 FROM papers WHERE arxiv_id = ?", (arxiv_id,)).fetchone():
            continue

        new_papers.append({
            'arxiv_id': arxiv_id,
            'category': category,
            'title': entry.find('atom:title', namespace).text.replace('\n', ' ').strip(),
            'summary': entry.find('atom:summary', namespace).text.replace('\n', ' ').strip(),
            'link': entry.find('atom:id', namespace).text,
            'published': entry.find('atom:published', namespace).text[:10],
            'authors': [a.find('atom:name', namespace).text for a in entry.findall('atom:author', namespace)]
        })

    if new_papers:
        print(f"Embedding {len(new_papers)} new papers...")
        summaries = [p['summary'] for p in new_papers]
        # FastEmbed generates NumPy arrays
        embeddings = list(model.embed(summaries))

        cursor = db.cursor()
        for p, emb in zip(new_papers, embeddings):
            # Insert metadata
            cursor.execute('''
                INSERT INTO papers (arxiv_id, category, title, summary, authors, link, published)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (p['arxiv_id'], p['category'], p['title'], p['summary'], json.dumps(p['authors']), p['link'], p['published']))

            row_id = cursor.lastrowid

            # Insert vector into sqlite-vec virtual table (accepts JSON strings)
            cursor.execute('''
                INSERT INTO vec_papers(rowid, embedding) VALUES (?, ?)
            ''', (row_id, json.dumps(emb.tolist())))
        db.commit()

def background_sync(category):
    print(f"Background syncing {category}...")
    fetch_and_store_arxiv(category, 25) 
    print("Sync complete.")

@app.route('/api/sync', methods=['POST'])
def trigger_sync():
    data = request.json
    category = data.get('category', 'cs.LG')
    
    # Run ingestion in a separate thread so the UI doesn't block
    threading.Thread(target=background_sync, args=(category,)).start()
    return jsonify({"status": "Sync started in background"})

@app.route('/api/radar', methods=['POST'])
def semantic_radar():
    data = request.json
    category = data.get('category', 'cs.LG')
    user_query = data.get('query', '')

    if not user_query:
        return jsonify({"error": "Query required"}), 400

    # Embed the user's search query
    query_vector = list(model.embed([user_query]))[0].tolist()

    # Perform Native SQL Vector Search using sqlite-vec's MATCH operator
    cursor = db.cursor()
    cursor.execute('''
        SELECT p.title, p.summary, p.authors, p.link, p.published, v.distance
        FROM vec_papers v
        JOIN papers p ON p.id = v.rowid
        WHERE v.embedding MATCH ? AND p.category = ?
        ORDER BY v.distance
        LIMIT 15
    ''', (json.dumps(query_vector), category))

    results = []
    for row in cursor.fetchall():
        # sqlite-vec returns L2 distance (lower is better). We invert it to a 0-100 score for the UI.
        raw_distance = float(row[5])
        score = max(0, min(100, round((1.0 - (raw_distance / 2.0)) * 100)))

        results.append({
            'title': row[0],
            'summary': row[1],
            'authors': json.loads(row[2]),
            'link': row[3],
            'published': row[4],
            'match_score': score
        })

    return jsonify(results)

if __name__ == '__main__':
    app.run(port=5000)
