import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import ast
from tqdm import tqdm
import csv

# -----------------------------
# Configuración CESGA
# -----------------------------
BASE_PATH = "/mnt/netapp2/Store_uni/home/usc/cursos/curso1278/teclin"
POSTS_CSV = os.path.join(BASE_PATH, "posts_by_author.csv")
EMBEDDING_FILE = os.path.join(BASE_PATH, "trait_embeddings_pos_neg.csv")
OUTPUT_CSV = os.path.join(BASE_PATH, "top50_posts_per_user.csv")
TOP_K = 50
BATCH_SIZE = 2048

TRAIT_ORDER = ["Agreeableness", "Openness", "Conscientiousness", "Extraversion", "Neuroticism"]

# -----------------------------
# 1️⃣ Cargar embeddings Big Five (pos/neg) desde CSV
# -----------------------------
print("🔹 Loading trait embeddings CSV...")

df_traits = pd.read_csv(EMBEDDING_FILE)
# df_traits columns: trait, polarity, dim_0...dim_383
dim_cols = [c for c in df_traits.columns if c.startswith("dim_")]

# Estructura:
# embeddings_dict[trait]["positive"] = vector
# embeddings_dict[trait]["negative"] = vector
embeddings_dict = {}

for trait in TRAIT_ORDER:
    sub = df_traits[df_traits["trait"] == trait]
    embeddings_dict[trait] = {
        "positive": sub[sub["polarity"]=="positive"][dim_cols].to_numpy()[0],
        "negative": sub[sub["polarity"]=="negative"][dim_cols].to_numpy()[0]
    }

print("✔ Loaded trait embeddings (pos/neg)")

# -----------------------------
# 2️⃣ Cargar posts de usuarios
# -----------------------------
df = pd.read_csv(POSTS_CSV)
df['body'] = df['body'].apply(ast.literal_eval)

# -----------------------------
# 3️⃣ Cargar modelo
# -----------------------------
print("🔹 Loading model...")
model = SentenceTransformer("all-MiniLM-L6-v2", device="cuda")

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
def get_top_k_posts_per_trait(posts, embeddings_dict, trait_name, top_k=10, batch_size=512):
    if len(posts) == 0:
        return {}

    # Embeddings de posts → (N, D)
    embeddings_posts = encode_batches(posts, batch_size)

    # Embeddings positivos y negativos del rasgo
    emb_pos = embeddings_dict[trait_name]["positive"]
    emb_neg = embeddings_dict[trait_name]["negative"]

    # Similaridades
    sims_pos = cosine_similarity([emb_pos], embeddings_posts)[0]
    sims_neg = cosine_similarity([emb_neg], embeddings_posts)[0]

    # Top-K posts (positivo)
    top_pos_idx = np.argsort(sims_pos)[::-1][:min(top_k, len(posts))]
    # Top-K posts (negativo)
    top_neg_idx = np.argsort(sims_neg)[::-1][:min(top_k, len(posts))]

    return {
        "positive": [
            {"post": posts[i], "similarity": sims_pos[i]}
            for i in top_pos_idx
        ],
        "negative": [
            {"post": posts[i], "similarity": sims_neg[i]}
            for i in top_neg_idx
        ]
    }

# -----------------------------
# 6️⃣ Aplicar por usuario
# -----------------------------
results = []

for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing users"):
    username = row["username"]
    posts = row["body"]

    for trait_name in TRAIT_ORDER:
        entries = get_top_k_posts_per_trait(
            posts, embeddings_dict, trait_name, TOP_K, BATCH_SIZE
        )

        # Guardar positivo y negativo
        for polarity in ["positive", "negative"]:
            for entry in entries[polarity]:
                results.append({
                    "username": username,
                    "trait": trait_name,
                    "polarity": polarity,
                    "post": entry["post"],
                    "similarity": entry["similarity"]
                })

# -----------------------------
# 7️⃣ Guardar resultados
# -----------------------------
df_results = pd.DataFrame(results)

cols = ["username", "trait", "polarity", "post", "similarity"]
df_results = df_results[cols]

df_results.to_csv(OUTPUT_CSV, index=False, quoting=csv.QUOTE_ALL)

print(f"✅ Saved top posts per user (pos/neg) in {OUTPUT_CSV}")
