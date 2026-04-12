# Methods: GAN and VAE for Imbalance Learning

## Problem Statement

**Data Imbalance Problem**: In long-tailed distributions, minority classes have extremely few samples, causing:

1. Classifiers overfit or ignore minority classes
2. Standard methods (SMOTE) fail on complex distributions
3. Evaluation metrics become misleading

## Solution: Generative Models for Oversampling

Instead of interpolating (SMOTE), we **learn the data distribution** and generate realistic samples.

## Conditional GAN (cGAN)

### Architecture

**Generator** G(z, y):
```
Input: z ~ N(0,I) [noise], y [class label]
  |
  v
Embedding Layer (embed y to 100-D)
  |
  v
FC Layers: 200 -> 2048 -> flatten(4×4×128)
  |
  v
Deconv2d: 128 -> 64 -> 32
  |
  v
Output: 32×32×3 image, Tanh activation
```

**Discriminator** D(x, y):
```
Input: x [image 32×32×3], y [class label]
  |
  v
Embedding Layer (embed y to spatial 32×32)
  |
  v
Concatenate [x, y_embedded] -> 4 channel input
  |
  v
Conv2d (stride 2): 4 -> 128 -> 256 -> 512
  |
  v
FC Layer + Sigmoid: -> [0,1] real/fake score
```

### Training Objective (Min-Max Game)

**Minimax loss**:
$$\min_G \max_D \mathbb{E}_{x \sim p_{data}}[\log D(x,y)] + \mathbb{E}_{z \sim p(z)}[\log(1 - D(G(z,y),y))]$$

**Practical Training** (alternating):

1. **Critic iterations** (5×):
   - Get real batch `(x_real, y)`
   - Sample `z ~ N(0,I)` and generate `x_fake = G(z, y)`
   - Compute: `L_D = -log(D(x_real,y)) - log(1-D(x_fake,y))`
   - Backprop and update discriminator

2. **Generator update** (1×):
   - Sample `z ~ N(0,I)` and generate `x_fake = G(z, y)`
   - Compute: `L_G = -log(D(x_fake,y))`
   - Backprop and update generator

### Advantages

- Generates **realistic high-quality images** (visual realism)
- Captures **complex distributions** (multimodal data)
- **Conditioned on class labels** ensures correct generation

### Disadvantages

- **Mode collapse**: Generator may ignore some classes/modes
- **Training instability**: Difficult hyperparameter tuning
- **Convergence issues**: May not reach equilibrium

### Hyperparameters (from `configs/gan.yaml`)

```yaml
latent_dim: 100          # Noise vector dimension
learning_rate: 0.0002    # Adam LR
beta1: 0.5               # Adam momentum
epochs: 50               # Training epochs
critic_iterations: 5     # D updates per G update
generator_hidden_dim: 128
discriminator_hidden_dim: 128
```

## Conditional VAE (CVAE)

### Architecture

**Encoder** q(z | x, y):
```
Input: x [image 32×32×3], y [class label]
  |
  v
Embed y to 32×32 spatial representation
  |
  v
Concatenate [x, y_embedded] -> 4 channels
  |
  v
Conv2d (stride 2): 4 -> 128 -> 256 -> 512
  |
  v
FC Layers: -> mean (64-D), log_var (64-D)
```

**Decoder** p(x | z, y):
```
Input: z [latent], y [class label]
  |
  v
Embed y to 64-D
  |
  v
Concatenate [z, y_embedded] -> 128-D
  |
  v
FC Layer: 128 -> 256×4×4
  |
  v
ConvTranspose2d: 256 -> 128 -> 64 -> 3
  |
  v
Output: 32×32×3 image, Sigmoid activation
```

### Training Objective (ELBO)

**Evidence Lower Bound Loss**:

$$\mathcal{L} = \mathbb{E}_{q(z|x,y)}[\log p(x|z,y)] - KL(q(z|x,y) || p(z))$$

**In practice**:

$$\mathcal{L} = \underbrace{BCE(x, \hat{x})}_{\text{Reconstruction}} + \underbrace{\lambda \cdot KL(q||p)}_{\text{Regularization}}$$

Where:
- **Reconstruction Loss** (BCE): How well decoder reconstructs original
- **KL Divergence**: How close learned distribution is to standard Gaussian
- **λ = 0.00025**: Weight balancing two terms (critical hyperparameter)

**Training** (single pass):

1. Encode `(x, y)` -> get `(mean, logvar)`
2. Reparameterize: `z = mean + std * epsilon` where `epsilon ~ N(0,I)`
3. Decode `(z, y)` -> get `x_recon`
4. Compute loss: `L_total = L_recon + lambda * L_kld`
5. Backprop and update all parameters jointly

### Advantages

- **Stable training**: Single loss, no adversarial dynamics
- **No mode collapse**: Latent space is smooth, can sample any `z`
- **Interpretable latent space**: Each dimension has meaning
- **Theoretical guarantees**: Probabilistic framework

### Disadvantages

- **Blurry samples**: Averaging over multiple modes produces blur
- **Mode averaging**: Cannot capture sharp distribution boundaries
- **Lower visual quality**: Compared to GAN samples
- **Posterior collapse**: In some configurations, may ignore latent code

### Hyperparameters (from `configs/vae.yaml`)

```yaml
latent_dim: 64           # Latent space dimension
learning_rate: 0.001     # Adam LR
epochs: 50               # Training epochs
kld_weight: 0.00025      # Beta-VAE weight (critical!)
encoder_hidden_dim: 256
decoder_hidden_dim: 256
```

## Comparison: GAN vs VAE

| Criterion | GAN | VAE |
|-----------|-----|-----|
| **Sample Quality** | High, sharp | Medium, blurry |
| **Training Stability** | Low, adversarial | High, stable |
| **Mode Coverage** | Partial, mode collapse | Full, no collapse |
| **Likelihood** | No explicit model | Explicit ELBO |
| **Latent Space** | Unstructured | Smooth, structured |
| **Training Time** | 5× critic updates per G update | Single pass |
| **Hyperparameter Tuning** | Difficult | Moderate (mainly lambda) |

## Oversampling Pipeline

### Step 1: Identify Minority Classes

```python
from src.data.imbalance import identify_minority_classes

minority = identify_minority_classes(targets, threshold=0.5)
# threshold=0.5: classes below 50th percentile
```

### Step 2: Calculate Samples Needed

```python
needed = get_samples_needed(targets, target_ratio=1.0)
# target_ratio=1.0: achieve exactly balanced distribution
```

For CIFAR-10-LT r-20 example:
- Truck (class 9): has 45 samples, needs 455 more
- Ship (class 8): has 60 samples, needs 440 more
- Horse (class 7): has 80 samples, needs 420 more
- etc.

### Step 3: Generate Synthetic Samples

**GAN Generation**:
```python
with torch.no_grad():
    for class_idx in minority_classes:
        z = torch.randn(num_to_generate, latent_dim)
        y = torch.full((num_to_generate,), class_idx)
        x_synthetic = generator(z, y)
```

**VAE Generation**:
```python
with torch.no_grad():
    for class_idx in minority_classes:
        z = torch.randn(num_to_generate, latent_dim)
        y = torch.full((num_to_generate,), class_idx)
        x_synthetic = vae.sample(z, y)
```

### Step 4: Combine and Retrain

```python
# Combine original + synthetic
combined_images = vstack([original_images, synthetic_images])
combined_labels = hstack([original_labels, synthetic_labels])

# Train classifier on combined data
classifier.fit(combined_images, combined_labels)

# Evaluate on original test set (NOT synthetic)
test_pred = classifier.predict(test_images)
```

### Step 5: Evaluation

**Critical**: Always evaluate on **original test set**, not synthetic!

```python
# Test set is class-balanced in CIFAR-10-LT
# Evaluate minority classes separately
per_class_recall = {}
for class_idx in range(10):
    mask = test_labels == class_idx
    recall = accuracy(pred[mask], test_labels[mask])
    per_class_recall[class_idx] = recall
```

## Key Design Decisions

### 1. Conditional Generation

Both models condition on `y` because:
- **Without conditioning**: Generator may collapse to majority class mode
- **With conditioning**: Explicitly guides each sample to correct class
- **Label embedding**: Maps discrete class to continuous space

### 2. Batch Normalization in generators

Used in GAN generator to:
- Stabilize training
- Reduce internal covariate shift
- Help with gradient flow

**Not used** in VAE decoder to:
- Preserve probabilistic interpretation
- Avoid mode collapse from centering

### 3. Balanced vs Artificially Balanced

We create **truly balanced** data (1000 per class), not `target_ratio=1.2`:
- Simpler evaluation
- Maximum fairness across classes
- Clear theoretical justification

### 4. No Synthetic Mixture

We do NOT mix synthetic samples (e.g., 50% original, 50%synthetic):
- Original approach: 100% original + generated minority only
- Reasoning: Don't need to oversample majority classes

## Mathematical Background

### VAE KL Divergence

$$KL(q(z|x) || p(z)) = \int q(z|x) \log\frac{q(z|x)}{p(z)} dz$$

For Gaussian distributions with **diagonal covariance**:

$$KL = \frac{1}{2}\sum_{j=1}^{J}(1 + \log\sigma_j^2 - \mu_j^2 - \sigma_j^2)$$

Where:
- $\mu_j$: mean of j-th latent dimension
- $\sigma_j^2 = \exp(\log\sigma_j^2)$: variance
- Minimizing: pulls distribution toward standard Gaussian

### GAN Minimax Objective

The generator and discriminator play a game:

- **Discriminator** tries to maximize: `log D(x) + log(1-D(G(z)))`
- **Generator** tries to minimize: `log(1-D(G(z)))`

Equivalently: **Generator minimizes** `log D(G(z))` early in training (stronger signal).

## References

- **Conditional GAN**: Mirza & Osinski (2014) "Conditional Generative Adversarial Nets"
- **VAE**: Kingma & Welling (2013) "Auto-Encoding Variational Bayes"
- **VAE Tutorial**: Doersch (2016) "Understanding Disentangling in β-VAE"
- **Imbalance**: He & Garcia (2009) "Learning from Imbalanced Data"
