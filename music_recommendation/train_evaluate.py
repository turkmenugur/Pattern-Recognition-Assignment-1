from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
from data_preparation import load_lastfm_data, create_user_artist_matrix
from collaborative_filtering import MusicRecommender

def train_and_evaluate():
    print("Veri yükleniyor...")
    user_artists = load_lastfm_data()
    user_artist_matrix = create_user_artist_matrix(user_artists)
    
    print("Veri bölünüyor...")
    mask = np.random.rand(*user_artist_matrix.shape) < 0.8
    train = user_artist_matrix.where(mask, 0)
    test = user_artist_matrix.where(~mask, 0)

    print("Model eğitiliyor...")
    recommender = MusicRecommender(train)
    recommender.fit()
    
    print("Model değerlendiriliyor...")
    mse = 0
    count = 0
    
    for user_id in user_artist_matrix.index[:100]:  
        try:
            recommendations = recommender.recommend_for_user(user_id)
            actual = test.loc[user_id]
            mse += mean_squared_error(actual[recommendations.index], recommendations)
            count += 1
        except Exception as e:
            print(f"Hata: {e}")
            continue
    
    mse = mse / count if count > 0 else float('inf')
    print(f"\nModel MSE: {mse:.4f}")
    
    return recommender

if __name__ == "__main__":
    recommender = train_and_evaluate()
