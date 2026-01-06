import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

# ====== 0) Dosya Yolu Ayarı ======
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path  = os.path.join(script_dir, 'realfishdataset.csv')

# ====== 1) Veri Yükleme ve Temizleme ======
df = pd.read_csv(data_path)
df['fish'] = df['fish'].str.strip().str.replace(',', '')

# Özellikler (ph, sıcaklık) ve hedef (fish)
X = df[['ph', 'temperature']]
y = df['fish']

# ====== 2) Etiketleme ======
le = LabelEncoder()
y_enc = le.fit_transform(y)
print("Etiketlenen sınıflar:", list(le.classes_))

# ====== 3) Eğitim-Test Ayrımı ======
X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc,
    test_size=0.2,
    random_state=42,
    stratify=y_enc
)

# ====== 4) Modeli Eğitme ======
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# ====== 5) Değerlendirme ======
y_pred = clf.predict(X_test)
print("Doğruluk:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=le.classes_))

# ====== 6) Model ve Encoder Kaydetme ======
model_path = os.path.join(script_dir, 'water_quality_model.pkl')
encoder_path = os.path.join(script_dir, 'label_encoder.pkl')
joblib.dump(clf, model_path)
joblib.dump(le, encoder_path)
print("Model ve etiket kodlayıcı kaydedildi:", model_path, encoder_path)

# ====== 7) Örnek Tahmin Fonksiyonu ======
def predict_species(ph, temperature):
    X_df = pd.DataFrame([[ph, temperature]], columns=['ph', 'temperature'])
    probs      = clf.predict_proba(X_df)[0]
    pred_index = clf.predict(X_df)[0]
    species    = le.inverse_transform([pred_index])[0]

    print(f"\nGirdi -> pH: {ph}, Sıcaklık: {temperature}°C")
    for cls, pr in zip(le.classes_, probs):
        print(f"{cls}: {pr*100:.1f}%")
    print(f"En uygun su canlısı: {species}")

# ====== 8) Demonstrasyon ======
predict_species(7.2, 28.0)
