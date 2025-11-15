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
EMBEDDING_FILE = os.path.join(BASE_PATH, "trait_embeddings.csv")
OUTPUT_CSV = os.path.join(BASE_PATH, "top10_posts_per_user.csv")
TOP_K = 10
BATCH_SIZE = 512

TRAIT_ORDER = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]

# -----------------------------
# 1️⃣ Cargar embeddings Big Five desde CSV
# -----------------------------
print("🔹 Loading trait embeddings CSV...")

df_traits = pd.read_csv(EMBEDDING_FILE)

# Ordenamos para garantizar orden Big-5
df_traits = df_traits.set_index("trait").loc[TRAIT_ORDER].reset_index()

# Extraemos solo columnas dim_*
dim_cols = [c for c in df_traits.columns if c.startswith("dim_")]
all_trait_embeddings = df_traits[dim_cols].to_numpy()  # shape = (5, D)

print(f"✔ Loaded trait embeddings with shape {all_trait_embeddings.shape}")

# -----------------------------
# 2️⃣ Cargar posts de usuarios
# -----------------------------
df = pd.read_csv(POSTS_CSV)
df['body'] = df['body'].apply(ast.literal_eval)

# -----------------------------
# 3️⃣ Cargar modelo
# -----------------------------
print("🔹 Loading model...")
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
# 5️⃣ Top-K posts por usuario (con 5 rasgos)
# -----------------------------
def get_top_k_posts(posts, trait_embeddings, top_k=10, batch_size=512):
    if len(posts) == 0:
        return []

    # Embeddings posts → (N, D)
    embeddings_posts = encode_batches(posts, batch_size)

    # Similaridad → (N, 5)
    sims = cosine_similarity(embeddings_posts, trait_embeddings)

    # Media para ranking → (N,)
    mean_scores = sims.mean(axis=1)

    # Top K
    top_indices = np.argsort(mean_scores)[::-1][:min(top_k, len(posts))]

    # Construimos resultados
    top_results = []
    for i in top_indices:
        entry = {
            "post": posts[i],
            "sim_openness": sims[i, 0],
            "sim_conscientiousness": sims[i, 1],
            "sim_extraversion": sims[i, 2],
            "sim_agreeableness": sims[i, 3],
            "sim_neuroticism": sims[i, 4],
            "mean_similarity": mean_scores[i]
        }
        top_results.append(entry)

    return top_results

# -----------------------------
# 6️⃣ Aplicar por usuario
# -----------------------------
results = []
for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing users"):
    username = row["username"]
    posts = row["body"]

    top_posts = get_top_k_posts(posts, all_trait_embeddings, TOP_K, BATCH_SIZE)

    for entry in top_posts:
        entry["username"] = username
        results.append(entry)

# -----------------------------
# 7️⃣ Guardar resultados
# -----------------------------
df_results = pd.DataFrame(results)
df_results.to_csv(OUTPUT_CSV, index=False)

print(f"✅ Saved top posts per user in {OUTPUT_CSV}")
