from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import torch
import os

app = Flask(__name__)

tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

def get_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].numpy()

@app.route('/search', methods=['POST'])
def search():
    data = request.get_json()
    emails = data.get('emails', [])
    query = data.get('query', '')

    if not emails or not query:
        return jsonify({'best_match': ''})

    query_embedding = get_embedding(query)
    email_embeddings = [get_embedding(email) for email in emails]

    sims = cosine_similarity(query_embedding, email_embeddings).flatten()
    best_index = sims.argmax()
    return jsonify({'best_match': emails[best_index]})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port)
