import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os

DATA_PATH = 'data/usersha1-artmbid-artname-plays.tsv'
PROCESSED_DATA_PATH = 'data/user_artist_matrix.pkl'

print(f"Veri dosyası mevcut mu: {os.path.exists(DATA_PATH)}")

def load_lastfm_data():    
    if os.path.exists(PROCESSED_DATA_PATH):
        print("İşlenmiş veri yükleniyor...")
        return pd.read_pickle(PROCESSED_DATA_PATH)
    
    print("İşlenmemiş veri yükleniyor...")
    user_artists = pd.read_csv(DATA_PATH, 
                              sep='\t', 
                              names=['user_id', 'artist_id', 'artist_name', 'plays'])
    
    min_plays = 1000
    user_artists = user_artists[user_artists['plays'] >= min_plays]
    # En aktif kullanıcıları seç
    user_listen_counts = user_artists.groupby('user_id')['plays'].sum()
    active_users = user_listen_counts.nlargest(1000).index 
    user_artists = user_artists[user_artists['user_id'].isin(active_users)]
    # En popüler sanatçıları seç
    artist_counts = user_artists['artist_name'].value_counts()
    popular_artists = artist_counts.nlargest(500).index  
    user_artists = user_artists[user_artists['artist_name'].isin(popular_artists)]

    user_artists['plays_log'] = np.log1p(user_artists['plays'])
    scaler = MinMaxScaler()
    user_artists['plays_normalized'] = scaler.fit_transform(user_artists[['plays_log']])
    
    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
    user_artists.to_pickle(PROCESSED_DATA_PATH)
    return user_artists

def create_user_artist_matrix(user_artists):
    user_artist_grouped = user_artists.groupby(['user_id', 'artist_name'])['plays_normalized'].mean().reset_index()
    matrix = user_artist_grouped.pivot(index='user_id', 
                                     columns='artist_name', 
                                     values='plays_normalized').fillna(0)
    return matrix

def show_dataset_stats(user_artists):
    print("\nVeri Seti İstatistikleri:")
    print(f"Toplam kullanıcı sayısı: {user_artists['user_id'].nunique()}")
    print(f"Toplam sanatçı sayısı: {user_artists['artist_name'].nunique()}")
    print(f"Toplam dinleme sayısı: {len(user_artists)}")
    print("\nEn çok dinlenen 5 sanatçı:")
    print(user_artists.groupby('artist_name')['plays'].sum().sort_values(ascending=False).head())

if __name__ == "__main__":
    user_artists = load_lastfm_data()
    show_dataset_stats(user_artists)
    user_artist_matrix = create_user_artist_matrix(user_artists)
    print("\nKullanıcı-Sanatçı matrisi boyutu:", user_artist_matrix.shape)
