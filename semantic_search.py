from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer, util

app = Flask(__name__)
model = SentenceTransformer('all-MiniLM-L6-v2')

@app.route('/search', methods=['POST'])
def search():
    data = request.json
    emails = data['emails']
    query = data['query']

    email_embeddings = model.encode(emails, convert_to_tensor=True)
    query_embedding = model.encode(query, convert_to_tensor=True)

    scores = util.cos_sim(query_embedding, email_embeddings)[0]
    best_index = scores.argmax().item()

    return jsonify({
        'best_match': emails[best_index],
        'score': float(scores[best_index])
    })

if __name__ == '__main__':
    app.run(port=5001)
