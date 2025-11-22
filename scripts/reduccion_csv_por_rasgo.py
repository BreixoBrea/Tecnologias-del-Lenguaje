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
POSTS_CSV = os.path.join(BASE_PATH, "posts_by_author.csv")
EMBEDDING_FILE = os.path.join(BASE_PATH, "trait_embeddings.csv")
OUTPUT_CSV = os.path.join(BASE_PATH, "top50_posts_per_user.csv")

TOP_K = 50
BATCH_SIZE = 2048

TRAIT_ORDER = ["Agreeableness", "Openness", "Conscientiousness", "Extraversion", "Neuroticism"]

# -----------------------------
# 1️⃣ Cargar embeddings Big Five desde CSV
# -----------------------------
print("🔹 Loading trait embeddings CSV...")

df_traits = pd.read_csv(EMBEDDING_FILE)
df_traits = df_traits.set_index("trait").loc[TRAIT_ORDER].reset_index()

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
# 5️⃣ Top-K posts por usuario Y por rasgo
# -----------------------------
def get_top_k_posts_per_trait(posts, trait_embeddings, top_k=10, batch_size=512):
    if len(posts) == 0:
        return {}

    # Embeddings posts → (N, D)
    embeddings_posts = encode_batches(posts, batch_size)

    results = {}

    # Para cada rasgo individualmente
    for t_idx, trait_vec in enumerate(trait_embeddings):

        # similitud (1,N)
        sims = cosine_similarity([trait_vec], embeddings_posts)[0]

        # TOP-K similitud
        top_idx = np.argsort(sims)[::-1][:min(top_k, len(posts))]

        # guardamos
        results[t_idx] = [
            {
                "post": posts[i],
                "similarity": sims[i]
            }
            for i in top_idx
        ]

    return results

# -----------------------------
# 6️⃣ Aplicar por usuario
# -----------------------------
results = []

for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing users"):
    username = row["username"]
    posts = row["body"]

    # Diccionario: trait_index -> lista de posts top
    top_by_trait = get_top_k_posts_per_trait(posts, all_trait_embeddings, TOP_K, BATCH_SIZE)

    # Guardar cada resultado en estructura plana
    for t_idx, entries in top_by_trait.items():
        trait_name = TRAIT_ORDER[t_idx]

        for entry in entries:
            results.append({
                "username": username,
                "trait": trait_name,
                "post": entry["post"],
                "similarity": entry["similarity"]
            })

# -----------------------------
# 7️⃣ Guardar resultados
# -----------------------------
df_results = pd.DataFrame(results)

# Asegurar username al principio
cols = ["username", "trait", "post", "similarity"]
df_results = df_results[cols]

df_results.to_csv(OUTPUT_CSV, index=False)

print(f"✅ Saved top posts per user in {OUTPUT_CSV}")
