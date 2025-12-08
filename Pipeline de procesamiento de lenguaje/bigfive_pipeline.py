import os
import pickle
import numpy as np
import pandas as pd
import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr


# 0. Configuración

STORE = "/mnt/netapp2/Store_uni/home/usc/cursos/curso1070"
POSTS_FILE = os.path.join(STORE, "posts_corregido.csv")

# Redirigir cache Hugging Face dentro de Store_uni
os.environ["HF_HOME"] = os.path.join(STORE, "huggingface_cache")
os.makedirs(os.environ["HF_HOME"], exist_ok=True)

keywords = ["agreeableness","conscientiousness","extraversion","neuroticism","openness",
            "high","medium","low","%"]
traits = ["agreeableness","conscientiousness","extraversion","neuroticism","openness"]


# 1. Cargar posts en chunks

chunksize = 500_000
posts_list = []

for chunk in pd.read_csv(POSTS_FILE, chunksize=chunksize, engine="python",
                         sep=",", quoting=csv.QUOTE_MINIMAL, on_bad_lines="skip"):
    if ("username" in chunk.columns) and ("body" in chunk.columns):
        chunk = chunk[["username", "body"]].dropna()
        posts_list.append(chunk)

posts = pd.concat(posts_list, ignore_index=True)
print(f"Posts cargados: {len(posts)}")

# 2. Crear features por usuario

# Conteo de keywords
def count_keywords(text):
    text = str(text).lower()
    return sum(k in text for k in keywords)

posts['keyword_count'] = posts['body'].apply(count_keywords)

# Agregar TF-IDF simplificado por usuario
vectorizer = TfidfVectorizer(max_features=500)
tfidf_matrix = vectorizer.fit_transform(posts['body'])
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())
tfidf_df['username'] = posts['username'].values

# Agregar por usuario (media de TF-IDF + suma de keywords)
user_features = tfidf_df.groupby('username').mean()
user_features['keyword_count'] = posts.groupby('username')['keyword_count'].sum()


# 3. Crear DataFrame train

train = user_features.copy()

# Si no tienes valores reales de traits, simular algunos
for t in traits:
    if t not in train.columns:
        train[t] = np.random.rand(len(train))  # reemplazar con datos reales si los tienes


# 4. Entrenamiento Ridge

models_mt = {}
alpha_grid = [0.1, 1.0, 10.0, 50.0, 100.0]

# Crear carpeta para modelos dentro de Store_uni
model_dir = os.path.join(STORE, "models/ridge")
os.makedirs(model_dir, exist_ok=True)

for t in traits:
    print(f"\n=== Selección de mejor Ridge para {t} ===")
    df_t = train.dropna(subset=[t])
    X = df_t[user_features.columns]
    y = df_t[t]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    best_mse = np.inf
    best_model = None
    best_metrics = {}

    for alpha in alpha_grid:
        model = Ridge(alpha=alpha, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        mse = mean_squared_error(y_test, preds)
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        pearson_corr, _ = pearsonr(y_test, preds)

        if mse < best_mse:
            best_mse = mse
            best_model = model
            best_metrics = {"MSE": mse, "MAE": mae, "R2": r2, "Pearson": pearson_corr}

    print(f"MSE: {best_metrics['MSE']:.2f} | MAE: {best_metrics['MAE']:.2f} | "
          f"R²: {best_metrics['R2']:.4f} | Pearson: {best_metrics['Pearson']:.4f}")
    models_mt[t] = best_model

    # Guardar modelo dentro de Store_uni
    ruta_archivo = os.path.join(model_dir, f"ridge_mt_{t}.pkl")
    with open(ruta_archivo, "wb") as f:
        pickle.dump(best_model, f)

print("\nMejores modelos Ridge guardados como ridge_mt_[trait].pkl")

