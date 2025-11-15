import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import ast
from tqdm import tqdm

# -----------------------------
# Configuración CESGA
# -----------------------------
BASE_PATH = "C:\\Users\\Usuario\\OneDrive - Universidade de Santiago de Compostela\\GRIA\\4º CURSO. 1º CUADRIMESTRE\\Tecnoloxías da Linguaxe\\práctica 4\\material"
POSTS_CSV = os.path.join(BASE_PATH, "post_by_author.csv")
EMBEDDING_FILE = os.path.join(BASE_PATH, "all_traits_embedding.npy")
OUTPUT_CSV = os.path.join(BASE_PATH, "top10_posts_per_user.csv")
TOP_K = 10
BATCH_SIZE = 512  # Ajustable según VRAM

# -----------------------------
# 1️⃣ Cargar embedding del Big-5
# -----------------------------
mean_embedding = np.load(EMBEDDING_FILE).reshape(1, -1)

# -----------------------------
# 2️⃣ Cargar posts de usuarios
# -----------------------------
df = pd.read_csv(POSTS_CSV)
df['body'] = df['body'].apply(ast.literal_eval)

# -----------------------------
# 3️⃣ Cargar modelo en GPU
# -----------------------------
print("🔹 Loading model on GPU...")
model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

# -----------------------------
# 4️⃣ Función para embeddings por batches
# -----------------------------
def encode_batches(sentences, batch_size=512):
    embeddings = []
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i+batch_size]
        emb_batch = model.encode(batch, normalize_embeddings=True)
        embeddings.append(emb_batch)
    return np.vstack(embeddings)

# -----------------------------
# 5️⃣ Función para top-K posts por usuario
# -----------------------------
def get_top_k_posts(posts, embedding_ref, top_k=10, batch_size=512):
    if len(posts) == 0:
        return []

    embeddings_posts = encode_batches(posts, batch_size)
    sims = cosine_similarity(embeddings_posts, embedding_ref).flatten()
    top_indices = np.argsort(sims)[::-1][:min(top_k, len(posts))]
    return [(posts[i], sims[i]) for i in top_indices]

# -----------------------------
# 6️⃣ Aplicar por usuario
# -----------------------------
results = []
for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing users"):
    username = row['username']
    posts = row['body']
    top_posts = get_top_k_posts(posts, mean_embedding, TOP_K, BATCH_SIZE)
    for post, score in top_posts:
        results.append({
            "username": username,
            "post": post,
            "similarity": score
        })

# -----------------------------
# 7️⃣ Guardar resultados
# -----------------------------
df_results = pd.DataFrame(results)
df_results.to_csv(OUTPUT_CSV, index=False)
print(f"✅ Saved top posts per user in {OUTPUT_CSV}")
