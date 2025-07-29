import mysql.connector
import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS
from joblib import load

app = Flask(__name__)
CORS(app)

# === MODEL & PREPROCESSING ===
model = load("model/svm_polynomial_best_model.joblib")
scaler = load("model/scaler.joblib")
pca = load("model/pca.joblib")  # Hapus jika tidak pakai PCA

# === DATABASE CONNECTION ===
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    database="screening_tbc"
)

# === LOGIN ===
@app.route('/api/login', methods=['POST'])
def login():
    data = request.json
    username = data.get('username')
    password = data.get('password')

    cursor = db.cursor(dictionary=True)
    cursor.execute("SELECT * FROM users WHERE username = %s", (username,))
    user = cursor.fetchone()

    if user and user['password'] == password:
        return jsonify({
            'user_id': user['id'],
            'username': user['username'],
            'role': user['role']
        }), 200
    else:
        return jsonify({'message': 'Username atau password salah'}), 401

# === REGISTER ===
@app.route('/api/register', methods=['POST'])
def register():
    data = request.json
    username = data.get('username')
    password = data.get('password')
    role = data.get('role', 'user')
    current_user_role = data.get('current_user_role', 'user')

    if role == 'admin' and current_user_role != 'admin':
        return jsonify({'message': 'Hanya admin yang bisa membuat akun admin'}), 403

    try:
        cursor = db.cursor()
        cursor.execute("SELECT * FROM users WHERE username = %s", (username,))
        if cursor.fetchone():
            return jsonify({'message': 'Username sudah digunakan'}), 400

        cursor.execute(
            "INSERT INTO users (username, password, role) VALUES (%s, %s, %s)",
            (username, password, role)
        )
        db.commit()
        return jsonify({'message': 'Registrasi berhasil'}), 201

    except mysql.connector.Error as err:
        print("🔥 REGISTER ERROR:", err)
        return jsonify({'message': 'Terjadi kesalahan pada server'}), 500

# === PREDIKSI ===
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        features = data.get('features')

        if not features or len(features) != 22:
            return jsonify({'error': 'Input fitur tidak valid'}), 400

        # Preprocessing
        X = np.array(features).reshape(1, -1)
        X_scaled = scaler.transform(X)
        X_final = pca.transform(X_scaled)  # Gunakan X_scaled jika tidak pakai PCA

        # Prediksi probabilitas kelas POSITIF (index ke-1)
        probability = model.predict_proba(X_final)[0][1]  # Probabilitas kelas 1
        prediction = 1 if probability >= 0.5 else 0       # Tentukan hasil manual

        return jsonify({
            'prediction': int(prediction),
            'probability': float(probability)
        })

    except Exception as e:
        print("🔥 PREDICT ERROR:", str(e))
        return jsonify({'error': str(e)}), 500

# === RUN APP ===
if __name__ == '__main__':
    app.run(debug=True, port=5000)
