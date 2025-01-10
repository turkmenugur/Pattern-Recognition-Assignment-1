from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd

class MusicRecommender:
    def __init__(self, user_artist_matrix):
        self.user_artist_matrix = user_artist_matrix
        self.user_similarity = None
        self.artist_similarity = None
        
    def fit(self):
        print("Benzerlik matrisleri hesaplanıyor...")
        self.user_similarity = cosine_similarity(self.user_artist_matrix)
        self.artist_similarity = cosine_similarity(self.user_artist_matrix.T)
        
    def recommend_for_user(self, user_id, n_recommendations=5):
        if user_id not in self.user_artist_matrix.index:
            raise ValueError("Kullanıcı bulunamadı!")
            
        user_idx = self.user_artist_matrix.index.get_loc(user_id)
        user_artists = self.user_artist_matrix.iloc[user_idx]
        
        # Benzer kullanıcıların dinlediği sanatçıları öner
        similar_scores = self.user_similarity[user_idx]
        weighted_scores = np.zeros(len(self.user_artist_matrix.columns))
        
        for other_idx, similarity in enumerate(similar_scores):
            if other_idx != user_idx:
                weighted_scores += similarity * self.user_artist_matrix.iloc[other_idx]
        
        # Kullanıcının zaten dinlediği sanatçıları filtrele
        weighted_scores[user_artists > 0] = 0
        
        # En yüksek puanlı sanatçıları döndür
        recommendations = pd.Series(
            weighted_scores, 
            index=self.user_artist_matrix.columns
        ).nlargest(n_recommendations)
        
        return recommendations

