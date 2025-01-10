import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from data_preparation import load_lastfm_data, create_user_artist_matrix
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

class MusicRecommender:
    def __init__(self):
        self.user_artists = load_lastfm_data()
        self.user_artist_matrix = create_user_artist_matrix(self.user_artists)
        print("Benzerlik matrisi hesaplanıyor...")
        self.similarity_matrix = cosine_similarity(self.user_artist_matrix)
        
    def get_similar_users(self, user_id, n=5):
        if user_id not in self.user_artist_matrix.index:
            return None
        
        user_idx = self.user_artist_matrix.index.get_loc(user_id)
        similar_users = self.similarity_matrix[user_idx]
        similar_user_indices = np.argsort(similar_users)[-n-1:-1][::-1]  
        
        return list(zip(
            self.user_artist_matrix.index[similar_user_indices],
            similar_users[similar_user_indices]
        ))
    
    def get_user_artists(self, user_id):
        if user_id not in self.user_artist_matrix.index:
            return None
        
        user_artists = self.user_artist_matrix.loc[user_id]
        listened_artists = user_artists[user_artists > 0].sort_values(ascending=False)
        return listened_artists
    
    def recommend_artists(self, user_id, n=5):
        if user_id not in self.user_artist_matrix.index:
            return None
            
        user_artists = self.user_artist_matrix.loc[user_id]
        unlistened_artists = user_artists[user_artists == 0].index
        
        similar_users = self.get_similar_users(user_id, n=10)
        recommendations = []
        
        for artist in unlistened_artists:
            score = 0
            total_similarity = 0
            for similar_user, similarity in similar_users:
                if similarity > 0.1:  
                    score += similarity * self.user_artist_matrix.loc[similar_user, artist]
                    total_similarity += similarity
            if total_similarity > 0:
                score = score / total_similarity  
            recommendations.append((artist, score))
        
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations[:n]

    def calculate_mae(self, user_id, recommendations):
        actual_scores = self.user_artist_matrix.loc[user_id, [artist for artist, score in recommendations]]
        predicted_scores = [score for artist, score in recommendations]
        mae = mean_absolute_error(actual_scores, predicted_scores)
        return mae

    def calculate_r2(self, user_id, recommendations):
        actual_scores = self.user_artist_matrix.loc[user_id, [artist for artist, score in recommendations]]
        predicted_scores = [score for artist, score in recommendations]
        r2 = r2_score(actual_scores, predicted_scores)
        return r2

if __name__ == "__main__":
    recommender = MusicRecommender()
   
    print("\nMevcut kullanıcı ID'leri:")
    print(list(recommender.user_artist_matrix.index)[:5])  
    
    example_user = recommender.user_artist_matrix.index[0]
    print(f"\nSeçilen kullanıcı ID: {example_user}")
        
    print("\nKullanıcının En Çok Dinlediği 5 Sanatçı:")
    top_artists = recommender.get_user_artists(example_user)
    for artist, score in top_artists[:5].items():
        print(f"Sanatçı: {artist}, Skor: {score:.3f}")
    
    print("\nBenzer Kullanıcılar:")
    similar_users = recommender.get_similar_users(example_user)
    for user, similarity in similar_users:
        print(f"Kullanıcı: {user}, Benzerlik: {similarity:.3f}")
    
    print("\nÖnerilen Sanatçılar:")
    recommendations = recommender.recommend_artists(example_user)
    for artist, score in recommendations:
        print(f"Sanatçı: {artist}, Öneri Skoru: {score:.3f}")

    mae = recommender.calculate_mae(example_user, recommendations)
    r2 = recommender.calculate_r2(example_user, recommendations)
    print(f"\nMAE: {mae:.3f}")

    actual_scores = recommender.user_artist_matrix.loc[example_user, [artist for artist, score in recommendations]]
    predicted_scores = [score for artist, score in recommendations]
    plt.scatter(actual_scores, predicted_scores)
    plt.xlabel("Gerçek Skor")
    plt.ylabel("Öneri Skoru")
    plt.title("Öneri Skoru Grafiği")
    plt.show()
