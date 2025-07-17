from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

app = Flask(__name__)
CORS(app)

@app.route('/search', methods=['POST'])
def search():
    data = request.get_json()
    emails = data.get('emails', [])
    query = data.get('query', '')

    if not emails or not query:
        return jsonify({'best_match': 'No results found'}), 400

    vectorizer = TfidfVectorizer().fit(emails + [query])
    vectors = vectorizer.transform(emails + [query])
    similarities = cosine_similarity(vectors[-1], vectors[:-1]).flatten()

    best_index = similarities.argmax()
    best_email = emails[best_index] if similarities[best_index] > 0 else 'No relevant match found'

    return jsonify({'best_match': best_email})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
