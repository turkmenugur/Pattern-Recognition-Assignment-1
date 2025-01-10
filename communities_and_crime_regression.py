import pandas as pd
import numpy as np
import seaborn as sns
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler, LabelEncoder
from ucimlrepo import fetch_ucirepo

#Veriyi internetten çekme
communities_and_crime = fetch_ucirepo(id=183)

X = communities_and_crime.data.features
y = communities_and_crime.data.targets

print("\nVeri seti inceleniyor...")
print("Başlangıç veri boyutu:", X.shape)

print("\nVeri tipleri:")
print(X.dtypes)

def clean_dataset(df):
    non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns
    print(f"\nSayısal olmayan sütunlar ({len(non_numeric_cols)}):")
    print(non_numeric_cols.tolist())
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df_numeric = df[numeric_cols].copy()
    
    # '?' karakterlerini NaN'a çevir
    df_numeric = df_numeric.replace(['?', 'NA', 'nan', 'NaN'], np.nan)
    
    # NaN değerleri medyan ile doldur
    for col in df_numeric.columns:
        if df_numeric[col].isnull().any():
            median_val = df_numeric[col].median()
            df_numeric[col] = df_numeric[col].fillna(median_val)
    
    return df_numeric

print("\nVeri temizleme başlıyor...")
X_cleaned = clean_dataset(X)
print("Veri temizlendikten sonra:", X_cleaned.shape)

# Uyumsuz değerleri kontrol edip ve temizleme
def remove_outliers(df, threshold=3):
    z_scores = np.abs((df - df.mean()) / df.std())
    mask = (z_scores < threshold).all(axis=1)
    return df[mask]

# Uyumsuz değerleri temizleme
X_clean = remove_outliers(X_cleaned)
y_clean = y.iloc[X_clean.index]

print("Uyumsuz değerler temizlendikten sonra:", X_clean.shape)

# Veriyi eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(
    X_clean, y_clean, test_size=0.2, random_state=42
)
# Validation setine ayırma
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_val_scaled = scaler.transform(X_val)

print("\nVeri seti boyutları:")
print(f"Eğitim seti: {X_train_scaled.shape}")
print(f"Test seti: {X_test_scaled.shape}")
print(f"Validasyon seti: {X_val_scaled.shape}")

print("\nModel eğitiliyor...")
model = LinearRegression()
model.fit(X_train_scaled, y_train)

print("Tahminler yapılıyor...")
y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)
y_val_pred = model.predict(X_val_scaled)

print("\nRegresyon Performans Metrikleri:")
print("\nEğitim Seti:")
print(f"MAE: {mean_absolute_error(y_train, y_train_pred):.4f}")
print(f"R-squared: {r2_score(y_train, y_train_pred):.4f}")

print("\nValidasyon Seti:")
print(f"MAE: {mean_absolute_error(y_val, y_val_pred):.4f}")
print(f"R-squared: {r2_score(y_val, y_val_pred):.4f}")

print("\nTest Seti:")
print(f"MAE: {mean_absolute_error(y_test, y_test_pred):.4f}")
print(f"R-squared: {r2_score(y_test, y_test_pred):.4f}")

plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
plt.scatter(y_test, y_test_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Gerçek Değerler')
plt.ylabel('Tahmin Edilen Değerler')
plt.title('Tahminler vs Gerçek Değerler')

plt.subplot(2, 2, 2)
residuals = y_test - y_test_pred
sns.histplot(residuals, kde=True)
plt.xlabel('Artık Değerler')
plt.ylabel('Frekans')
plt.title('Artık Değerlerin Dağılımı')

plt.subplot(2, 2, 3)
sns.kdeplot(data=pd.DataFrame({'Gerçek': y_test.values.flatten(), 
                              'Tahmin': y_test_pred.flatten()}))
plt.xlabel('Değerler')
plt.ylabel('Yoğunluk')
plt.title('Gerçek ve Tahmin Değerlerinin Dağılımı')

# Özellik Önem Dereceleri
plt.subplot(2, 2, 4)
feature_importance = pd.DataFrame({
    'feature': X_clean.columns,
    'importance': np.abs(model.coef_[0])
})
feature_importance = feature_importance.sort_values('importance', ascending=False)
sns.barplot(data=feature_importance.head(10), x='importance', y='feature')
plt.title('En Önemli 10 Özellik')

plt.tight_layout()
plt.show()

print("\nEn önemli 10 özellik:")
print(feature_importance.head(10))
# Model kaydetme
joblib.dump(model, 'communities_crime_model.joblib')
joblib.dump(scaler, 'communities_crime_scaler.joblib')
print("\nModel ve scaler kaydedildi!")
