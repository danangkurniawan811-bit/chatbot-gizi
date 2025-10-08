import openai
import MySQLdb
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, request, jsonify
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from openai import OpenAI
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
nltk.download('stopwords')
from nltk.corpus import stopwords
import os
from dotenv import load_dotenv
load_dotenv()  # Memuat variabel dari file .env


# Konfigurasi API
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY


# Inisialisasi model GPT dari Langchain
chat_model = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY)

# Mengonfigurasi Memory untuk Menjaga Konteks Percakapan
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Mendapatkan stop words Bahasa Indonesia dari NLTK
stop_words_id = stopwords.words('indonesian')
# Fungsi untuk mencari dengan cosine similarity menggunakan TF-IDF
def search_with_cosine_similarity(query, documents, top_k=1):
    vectorizer = TfidfVectorizer(stop_words=stop_words_id)
    all_documents = [query] + documents
    tfidf_matrix = vectorizer.fit_transform(all_documents)
    cosine_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    # Debugging: tampilkan skor cosine similarity per dokumen
    print(f"Cosine Similarities: {cosine_similarities}")
    ranked_results = sorted(zip(documents, cosine_similarities), key=lambda x: x[1], reverse=True)
    return ranked_results[:top_k]

# Fungsi untuk mencari jawaban berbasis kandungan gizi dan rasa
def search_database_with_similarity(query, top_k=1):
    db = MySQLdb.connect(
    host=os.getenv("MYSQL_HOST"),
    user=os.getenv("MYSQL_USER"),
    password=os.getenv("MYSQL_PASSWORD"),
    database=os.getenv("MYSQL_DB"),
    port=int(os.getenv("MYSQL_PORT"))
)

    cursor = db.cursor()

    cursor.execute("SELECT id, name, proteins, calories, carbohydrate, fat, manfaat FROM nutrition_data")
    rows = cursor.fetchall()

    documents = []
    for id, name, proteins, calories, carbohydrate, fat, manfaat in rows:
        doc = f"Nama: {name}, Protein: {proteins}g, Kalori: {calories}kcal, Karbohidrat: {carbohydrate}g, Lemak: {fat}g, Manfaat: {manfaat}"
        documents.append(doc)

    top_results = search_with_cosine_similarity(query, documents, top_k)

    results = []
    for result in top_results:
        doc, score = result
        parts = doc.split(", ")

        def extract_part(parts, index, label):
            try:
                if len(parts) > index and ":" in parts[index]:
                    return parts[index].split(":", 1)[1].strip()
            except Exception:
                pass
            return "N/A"

        name = extract_part(parts, 0, "Nama")
        proteins = extract_part(parts, 1, "Protein")
        calories = extract_part(parts, 2, "Kalori")
        carbohydrate = extract_part(parts, 3, "Karbohidrat")
        fat = extract_part(parts, 4, "Lemak")
        manfaat = extract_part(parts, 7, "Manfaat")

        results.append({
            'name': name,
            'score': score,
            'proteins': proteins,
            'calories': calories,
            'carbohydrate': carbohydrate,
            'fat': fat,
            'manfaat': manfaat
        })

    db.close()
    return results

# Fungsi untuk mendapatkan jawaban dinamis dari GPT berdasarkan pencarian
from openai import OpenAI

client = OpenAI(api_key=OPENAI_API_KEY)

def get_dynamic_gpt_answer(query, top_results=[]):
    if not top_results:
        context = "Tidak ada data relevan yang ditemukan."
    else:
        context = ""
        for food in top_results:
            context += (
                f"Nama: {food['name']}, "
                f"Protein: {food['proteins']}g, "
                f"Kalori: {food['calories']}kcal, "
                f"Karbohidrat: {food['carbohydrate']}g, "
                f"Lemak: {food['fat']}g, "
                f"Manfaat: {food['manfaat']}.\n"
            )

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "Kamu adalah asisten gizi pintar. Jawablah dengan bahasa alami, ramah, dan menyertakan kandungan nutrisi jika tersedia."
                },
                {
                    "role": "user",
                    "content": f"Pertanyaan: {query}\nBerikut data makanan:\n{context}\nJawab berdasarkan data tersebut."
                }
            ],
            temperature=0.7,
            max_tokens=300
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error: {str(e)}")
        return "Maaf, terjadi kesalahan saat memproses jawaban."



# Membuat dan menjalankan aplikasi Flask
app = Flask(__name__)

@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    query = data.get('query', '').strip().lower()

    if query:
        top_results = search_database_with_similarity(query, top_k=1)
        gpt_answer = get_dynamic_gpt_answer(query, top_results)

        response_text = f"🤖 Penjelasan:\n{gpt_answer}\n\n📊 Rank Jawaban:\n"
        for result in top_results:
            response_text += f"✨ {result['name']} (Skor: {result['score']:.4f})\n"
            response_text += f"- Protein: {result['proteins']}g\n"
            response_text += f"- Karbohidrat: {result['carbohydrate']}\n"
            response_text += f"- Lemak: {result['fat']}\n"
            response_text += f"- Kalori: {result['calories']}\n"

        response_text += "\n\nSemoga membantu! Ada pertanyaan lain yang bisa saya jawab? 😊"
        return jsonify({"answer": response_text})
    else:
        return jsonify({"answer": "Maaf, saya tidak menerima pertanyaan yang valid. Bisa dicoba lagi? 😊"})

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
