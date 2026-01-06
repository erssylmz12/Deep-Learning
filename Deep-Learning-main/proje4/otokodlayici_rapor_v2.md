# Teknik Rapor
**YZM 304 – Derin Öğrenme, Ödev 4**  
**Konu:** Otokodlayıcı (Autoencoder) Mimari Karşılaştırması  
**Öğrenci:** Eriş Söylemez – 22290702

## 1. Amaç
MNIST el yazısı rakam veri seti üzerinde üç yaklaşımın sınıflandırma başarılarını ve eğitim dinamiklerini karşılaştırmak:

- **Basit Otokodlayıcı (Simple AE) + Lojistik Regresyon (LR)**
- **Çok Katmanlı/Stacked Otokodlayıcı (Stacked AE) + LR**
- **Ham Piksel Özelliği + LR**

---

## 2. Veri Seti
- **MNIST** – 60 000 eğitim, 10 000 test örneği, 28 × 28 gri‑tonlu resim  
- Pikseller **0‑1** aralığına normalize edildi (`ToTensor()`).

---

## 3. Modeller

| Model | Encoder Boyutu | Decoder Katmanları | Aktivasyon | Kod Boyutu |
|-------|----------------|--------------------|------------|------------|
| Simple AE | 784 → 128 → 64 | 64 → 128 → 784 | ReLU / Sigmoid | 64 |
| Stacked AE | 784 → 256 → 128 → 64 | 64 → 128 → 256 → 784 | ReLU / Sigmoid | 64 |

- **Kayıp işlevi:** MSELoss  
- **Optimizasyon:** Adam (öğrenme oranı = 1e‑2)  
- **Eğitim:** Simple AE → 10 epoch, Stacked AE → 20 epoch  
- Kod (latent) vektörlerine **StandardScaler** uygulanıp **LR** (max_iter = 1000) ile sınıflandırıldı.  
- Karşılaştırma için ham pikseller aynı LR modeline verildi.

---

## 4. Eğitim Kayıpları

### Simple AE – 10 Epoch
| Epoch | Loss |
|------:|------:|
| 1 | 0.0496 |
| 2 | 0.0231 |
| 3 | 0.0163 |
| 4 | 0.0134 |
| 5 | 0.0118 |
| 6 | 0.0108 |
| 7 | 0.0099 |
| 8 | 0.0094 |
| 9 | 0.0089 |
| 10 | **0.0085** |

### Stacked AE – 20 Epoch
| Epoch | Loss |
|------:|------:|
| 1 | 0.0661 |
| 2 | 0.0371 |
| 3 | 0.0290 |
| 4 | 0.0254 |
| 5 | 0.0226 |
| 6 | 0.0208 |
| 7 | 0.0197 |
| 8 | 0.0185 |
| 9 | 0.0175 |
| 10 | 0.0170 |
| 11 | 0.0160 |
| 12 | 0.0158 |
| 13 | 0.0149 |
| 14 | 0.0145 |
| 15 | 0.0141 |
| 16 | 0.0138 |
| 17 | 0.0133 |
| 18 | 0.0130 |
| 19 | 0.0127 |
| 20 | **0.0125** |

---

## 5. Sonuçlar

| Yöntem | Doğruluk (%) |
|--------|--------------|
| **Simple AE + LR** | **88.82** |
| **Stacked AE + LR** | 87.80 |
| **Ham Piksel + LR** | 86.94 |

---

## 6. Tartışma
1. **Temsili Yetkinlik** – 64‑boyutluk kod, 784 piksele göre %92 daha küçük olup daha yüksek performans verdi.  
2. **Model Karmaşıklığı** – Stacked AE’de parametre sayısı ≈ 2.5 × arttı ancak doğrulukta düşüş gözlemlendi.  
3. **Eğitim Kayıpları**  
   - Simple AE, yalnızca 10 epoch’ta **0.0496 → 0.0085** düşerek hızlı ve kararlı bir yakınsama sergiledi.  
   - Stacked AE, daha yüksek başlangıç kaybı (0.0661) ile başladı ve 20 epoch sonunda **0.0125**’e indi; yani mutlak değer olarak Simple AE’den hâlâ yüksekte kaldı.  
   - Bu da, daha derin mimarinin fazla parametre nedeniyle küçük veri setinde *over‑parameterization* riskini taşıdığını gösteriyor.  
4. **Hata Dağılımı** – Tüm modellerde “4 ↔ 9” ve “3 ↔ 5” karışıklıkları öne çıkıyor; AE’ler **0, 1, 7**’de belirgin üstünlük sağladı.

---

## 7. Sonuç ve Öneriler
- **Basit mimari, yüksek isabet.** Ek katmanlar parametre şişirmesine neden olup doğruluğu düşürebilir.  
- **Daha az boyut = Daha hızlı.** Kod vektörleri gömülü sistemler için uygun.  
- **Gelecek Çalışmalar:** Konvolüsyonel AE, kod boyut denemeleri (32/16), Varyasyonel AE.
