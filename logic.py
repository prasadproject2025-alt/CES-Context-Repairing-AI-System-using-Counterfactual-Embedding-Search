import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sentence_transformers import SentenceTransformer, util

# This is the class your app is looking for!
class CESSystem:
    def __init__(self):
        print("--- initializing CES System ---")
        self.vectorizer = TfidfVectorizer()
        self.classifier = LogisticRegression()
        self.embedder = None 
        
        # Load and Train immediately
        self._train_classifier()
        self._load_embedder()
        print("--- System Ready ---")

    def _train_classifier(self):
        try:
            df = pd.read_csv("dataset.csv")
            X = df['text']
            y = df['label']
            X_vec = self.vectorizer.fit_transform(X)
            self.classifier.fit(X_vec, y)
            print(">> Classifier trained successfully")
        except Exception as e:
            print(f"Error training classifier: {e}")

    def _load_embedder(self):
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        print(">> Embedding model loaded")

    def detect_missing_context(self, query):
        vec = self.vectorizer.transform([query])
        prediction = self.classifier.predict(vec)[0]
        return bool(prediction == 1)

    def generate_counterfactuals(self, query):
        possible_intents = [
            "weather report", "stock price analysis", "traffic update", 
            "sales report", "exam result", "cricket match score", 
            "blood test report", "machine learning model accuracy"
        ]
        cfs = []
        for intent in possible_intents:
            cfs.append(f"{query} for {intent}")
            cfs.append(f"Show {intent}")
        return cfs

    def rank_interpretations(self, query, cfs, context_history):
        query_emb = self.embedder.encode(query, convert_to_tensor=True)
        context_emb = self.embedder.encode(context_history, convert_to_tensor=True)
        cf_embs = self.embedder.encode(cfs, convert_to_tensor=True)
        
        scores = []
        for i, cf_emb in enumerate(cf_embs):
            sim_query = util.cos_sim(query_emb, cf_emb)
            sim_context = util.cos_sim(context_emb, cf_emb)
            final_score = (0.4 * sim_query) + (0.6 * sim_context)
            scores.append((cfs[i], final_score.item()))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores