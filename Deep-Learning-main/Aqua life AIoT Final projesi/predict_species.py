import os
import re
import joblib

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Dosya isimleri 
TEMP_FILE    = 'temp.txt'
PH_FILE      = 'ph.txt'
MODEL_FILE   = 'water_quality_model.pkl'
ENCODER_FILE = 'label_encoder.pkl'

# Son sıcaklık değerini oku
with open(TEMP_FILE, 'r') as f:
    lines = [l.strip() for l in f if l.strip()]
last_temp = lines[-1]                # örn. "2025-05-27…->0,25.2"
temp      = float(last_temp.split(',')[1])

# Son pH değerini oku
with open(PH_FILE, 'r') as f:
    lines = [l.strip() for l in f if l.strip()]
last_ph = lines[-1]                  # örn. "…temperature:25.2°C  pH:7.23"
ph  = float(re.search(r'pH:([\d\.]+)', last_ph).group(1))
clf = joblib.load(MODEL_FILE)
le  = joblib.load(ENCODER_FILE)

import pandas as pd
X = pd.DataFrame([[ph, temp]], columns=['ph', 'temperature'])

# Tahmin yaparken artık X’i kullanın:
probs      = clf.predict_proba(X)[0]
pred_index = clf.predict(X)[0]
species    = le.inverse_transform([pred_index])[0]

# Sonuçları yazdırma kısmı aynı kalacak
print(f"Son ölçümler -> pH: {ph}, Sıcaklık: {temp}°C\n")
for cls, pr in zip(le.classes_, probs):
    print(f"{cls}: %{pr*100:.1f}")
print(f"\nÖnerilen canlı: {species}")
