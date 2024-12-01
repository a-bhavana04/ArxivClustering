import streamlit as st
import joblib
import tensorflow_hub as hub
import tensorflow as tf
import pandas as pd
import numpy as np
import random

# Load the trained KNeighborsClassifier model
knn_model = joblib.load('/content/drive/MyDrive/knn_model_4.joblib')

# Load the Universal Sentence Encoder model
MODEL_URL = "https://tfhub.dev/google/universal-sentence-encoder/4"
sentence_encoder_layer = hub.KerasLayer(MODEL_URL,
                                        input_shape=[],
                                        dtype=tf.string,
                                        trainable=False,
                                        name="use")

def preprocess_data(data):
    abstracts = data["abstract"].to_list()

    if not abstracts:
        st.error("Error: Abstracts list is empty.")
        return None
    else:
        embeddings = []
        batch_size = 3000
        num_batches = (len(abstracts) + batch_size - 1) // batch_size
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(abstracts))
            batch_abstracts = abstracts[start_idx:end_idx]

            if batch_abstracts:  # Check if batch is not empty
                batch_embeddings = sentence_encoder_layer(batch_abstracts)
                embeddings.extend(batch_embeddings.numpy())
            else:
                st.warning(f"Warning: Batch {i} is empty.")

        embeddings = np.array(embeddings)

        y = data.index
        return embeddings

def recommend_similar_titles(sample_idx, embeddings, df1, knn_model):
    dist, index = knn_model.kneighbors(X=embeddings[sample_idx].reshape(1,-1))
    st.write("Sample:", df1.iloc[sample_idx]["title"])
    for j in range(1,6):
        st.write(f"Recommendation {j}:", df1.iloc[index[0][j]]['title'])
    st.write("===============\n")

# Sample DataFrame (replace with your actual DataFrame)
data = pd.read_csv("/content/drive/MyDrive/df4.csv")

# Preprocess the data
input_embeddings = preprocess_data(data)

# Streamlit App
st.title("Title Recommendation System")

# Example: Recommend similar titles for 5 random samples
for _ in range(5):
    sample_idx = random.randint(0, data.shape[0]-1)  # Ensure sample index is within bounds
    recommend_similar_titles(sample_idx, input_embeddings, data, knn_model)