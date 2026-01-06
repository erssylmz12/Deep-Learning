# Blog Yazısı
## “Az Parametreyle Daha İyi Sonuç: MNIST’te Otokodlayıcı Maceram”

El yazısı rakamları sınıflandırmak makine öğrenmesinin “Merhaba Dünya!”sı gibidir. Peki **sade bir otokodlayıcı**, ham piksel verisini geride bırakabilir mi?

### Kurulum
- **MNIST** verisini 0‑1 aralığına sıkıştırdım.  
- İki otokodlayıcı kurdum:  
  1. **Simple AE** – Tek ara katman.  
  2. **Stacked AE** – Daha derin, daha karmaşık.  
- Her ikisinin 64‑boyutluk gizli katmanını **lojistik regresyon** ile sınıflandırdım. Kontrol grubum ise ham piksel vektörüydü.

### Eğitim Esnasında Neler Oldu?
| Model | Başlangıç Kayıp | Son Kayıp |
|-------|-----------------|-----------|
| Simple AE (10 ep.) | 0.0496 | **0.0085** |
| Stacked AE (20 ep.) | 0.0661 | **0.0125** |

Simple AE, yarı sürede daha düşük kayba ulaşarak hızla öğrendi!  

### Sonuçlar
| Yöntem | Doğruluk |
|--------|----------|
| **Simple AE + LR** | **%88.82 🎯** |
| Stacked AE + LR | %87.80 |
| Ham Piksel + LR | %86.94 |

> En basit model kazandı!

### Neden Basit Daha İyi?
- **Aşırı Uyum** – Derin ağ, az veri çeşitliliğinde ezber yapabiliyor.  
- **Bilgi Yoğunluğu** – 64 sayı, 784 piksele göre daha temiz bir özet sunuyor.  
- **Hafiflik** – Daha az parametre, daha kısa eğitim süresi.

### Çıkarımlar
- “Basiti dene” kuralı hâlâ altın değerinde.  
- Otokodlayıcılar, klasik ML algoritmalarını bambaşka bir seviyeye çıkarabiliyor.  
- Kod boyutunu kısaltmak model taşınabilirliğini artırıyor.

### Sırada Ne Var?
- **Konvolüsyonel AE** ile uzamsal bilgiyi de içselleştirmek.  
- **Varyasyonel AE** ile daha gürültüye dayanıklı kodlar üretmek.  
- Aynı yaklaşımı renkli **CIFAR‑10** veri setine uygulayıp başarı eğrisini incelemek.

> **Özet:** Daha katmanlı ≠ daha iyi. Bazen “az, çoktur.” 🚀
