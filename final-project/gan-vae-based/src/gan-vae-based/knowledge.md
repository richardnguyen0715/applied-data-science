# 1. Bản chất của bài toán oversampling bằng generative models

## 1.1. Vấn đề của SMOTE và các phương pháp cổ điển

Các phương pháp như:

* SMOTE
* ADASYN

thực chất chỉ:

* Nội suy tuyến tính trong feature space
* Không học phân phối dữ liệu thực

Hệ quả:

* Sinh sample không realistic
* Dễ tạo noise
* Không hoạt động tốt với:

  * dữ liệu ảnh
  * dữ liệu có manifold phức tạp

---

## 1.2. Ý tưởng của GAN/VAE-based oversampling

Thay vì:

> “interpolate data”

Ta làm:

> “learn data distribution p(x | y) và sample từ đó”

Formal:

* Với mỗi class ( y ), học phân phối:
  [
  p(x | y)
  ]

Sau đó:

* Generate thêm dữ liệu cho minority class

---

# 2. GAN-based Oversampling

## 2.1. Kiến trúc cơ bản

GAN gồm 2 thành phần:

* Generator: ( G(z, y) \rightarrow x )
* Discriminator: ( D(x, y) \rightarrow [0,1] )

Mục tiêu:
[
\min_G \max_D \mathbb{E}*{x \sim p*{data}}[\log D(x,y)] + \mathbb{E}_{z \sim p(z)}[\log(1 - D(G(z,y),y))]
]

---

## 2.2. Conditional GAN (cGAN) cho imbalance

Rất quan trọng:

* Conditioning vào label ( y )

Generator:
[
G(z, y)
]

Discriminator:
[
D(x, y)
]

### Lợi ích:

* Generate đúng class
* Có thể chỉ generate minority class

---

## 2.3. Pipeline oversampling với GAN

1. Train GAN trên dataset imbalance
2. Identify minority classes
3. Sample:
   [
   x_{synthetic} \sim G(z, y_{minor})
   ]
4. Merge:
   [
   D_{new} = D_{original} \cup D_{synthetic}
   ]
5. Train classifier

---

## 2.4. Các biến thể GAN quan trọng

### (1) WGAN / WGAN-GP

* Dùng Wasserstein distance
* Stable hơn

### (2) DCGAN

* CNN-based
* Phù hợp image

### (3) StyleGAN (advanced)

* High-quality images
* Ít dùng cho tabular

### (4) Tabular GAN

* CTGAN
* TVAE

---

## 2.5. Ưu điểm của GAN

* Sinh dữ liệu realistic
* Capture distribution phức tạp
* Tốt cho:

  * image
  * tabular phi tuyến

---

## 2.6. Nhược điểm

* Mode collapse
* Training unstable
* Khó tune
* Overfit minority nếu data quá ít

---

# 3. VAE-based Oversampling

## 3.1. Kiến trúc cơ bản

VAE gồm:

* Encoder: ( q(z|x,y) )
* Decoder: ( p(x|z,y) )

Loss:
[
\mathcal{L} = \mathbb{E}_{q(z|x)}[\log p(x|z)] - KL(q(z|x) || p(z))
]

---

## 3.2. Conditional VAE (CVAE)

Tương tự GAN:

* Encoder:
  [
  q(z|x,y)
  ]

* Decoder:
  [
  p(x|z,y)
  ]

---

## 3.3. Pipeline oversampling với VAE

1. Train CVAE
2. Sample:
   [
   z \sim \mathcal{N}(0, I)
   ]
3. Generate:
   [
   x = Decoder(z, y_{minor})
   ]
4. Merge dataset
5. Train classifier

---

## 3.4. Ưu điểm của VAE

* Training ổn định hơn GAN
* Không bị mode collapse
* Latent space có structure

---

## 3.5. Nhược điểm

* Sample blur (đặc biệt với image)
* Chất lượng thấp hơn GAN
* Over-smoothing

---

# 4. So sánh GAN vs VAE

| Tiêu chí            | GAN            | VAE        |
| ------------------- | -------------- | ---------- |
| Chất lượng mẫu      | Cao            | Trung bình |
| Stability           | Thấp           | Cao        |
| Mode collapse       | Có             | Không      |
| Likelihood modeling | Không explicit | Có         |
| Training difficulty | Cao            | Trung bình |

---

# 5. Các vấn đề quan trọng trong oversampling

## 5.1. Overfitting minority class

* Generator chỉ học vài sample
* Sinh data “duplicate-like”

Giải pháp:

* Regularization
* Data augmentation

---

## 5.2. Distribution shift

Synthetic data không khớp real data

Giải pháp:

* Use discriminator score filtering
* Use classifier feedback

---

## 5.3. Class boundary distortion

* Oversampling làm boundary sai

Giải pháp:

* Combine với undersampling
* Use cost-sensitive loss

---

# 6. Advanced techniques (2024–2026)

## 6.1. GAN + Classifier joint training

* Generator optimize:
  [
  \text{classification loss}
  ]

→ Generate data giúp classifier tốt hơn

---

## 6.2. Feature-space generation

Thay vì generate raw data:

* Generate embedding

Ưu điểm:

* Dễ hơn
* Stable hơn

---

## 6.3. Diffusion-based oversampling

* Thay GAN bằng diffusion model
* Chất lượng cao hơn GAN

---

## 6.4. Meta-learning oversampling

* Learn cách oversample
* Adaptive theo dataset

---

# 7. Khi nào nên dùng GAN/VAE oversampling

## Nên dùng khi:

* Data phức tạp (image, tabular nonlinear)
* Minority rất ít
* SMOTE không hiệu quả

## Không nên dùng khi:

* Dataset nhỏ (< 1k samples)
* Feature đơn giản
* Tabular tuyến tính

---

# 8. Evaluation trong oversampling

Không chỉ accuracy:

* Macro-F1
* Recall (minority)
* G-mean
* AUC

---

# 9. Best practices (rất quan trọng)

1. Không oversample quá mức
2. Luôn giữ validation set nguyên bản
3. So sánh với:

   * baseline (no sampling)
   * SMOTE
4. Kiểm tra:

   * class distribution
   * diversity của synthetic data

---

# 10. Insight quan trọng nhất

* GAN/VAE không chỉ là “data augmentation”
* Mà là:

  > học phân phối dữ liệu để tái cân bằng không gian xác suất

---

# 11. Tóm tắt ngắn gọn

* SMOTE: interpolation
* GAN/VAE: distribution learning
* GAN: realistic nhưng unstable
* VAE: stable nhưng kém sắc nét
* Xu hướng mới:

  * Diffusion > GAN
  * Hybrid methods > single method
