# Diffusion-Based Oversampling Methods

## Background: Denoising Diffusion Probabilistic Models

### Diffusion Process Overview

DDPM learns to reverse noise addition through iterative denoising:

#### Forward Process (Noising)
Gradually add Gaussian noise to images:

$$q(x_t | x_0) = \sqrt{\bar{\alpha}_t} \cdot x_0 + \sqrt{1-\bar{\alpha}_t} \cdot \epsilon$$

Where:
- $x_0$: Original clean image
- $x_t$: Noisy image at step $t$
- $\bar{\alpha}_t$: Cumulative noise schedule (decreases from 1 to ~0)
- $\epsilon \sim \mathcal{N}(0, I)$: Gaussian noise

This process gradually transforms images into pure noise over $T=1000$ steps.

#### Reverse Process (Denoising)
Learn to reverse the process:

$$p_\theta(x_{t-1}|x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \sigma_t^2 I)$$

The model learns a neural network $\epsilon_\theta(x_t, t)$ to predict the noise:

$$\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}}(x_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}}\epsilon_\theta(x_t, t))$$

### Training Objective

Directly predict noise with MSE loss:

$$\mathcal{L} = \mathbb{E}_{x_0, t, \epsilon} \left[||\epsilon - \epsilon_\theta(x_t, t)||_2^2\right]$$

This is equivalent to optimizing the variational lower bound.

## Conditional Diffusion for Class-Balanced Generation

### Class-Conditional Extension

To generate samples from specific classes, we condition the model:

$$p_\theta(x_{t-1}|x_t, y) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t, y), \sigma_t^2 I)$$

The network becomes class-aware:

$$\epsilon_\theta(x_t, t, y)$$

### Class Conditioning Strategy

1. **Class Embedding**: Convert class label to embedding vector
   $$e_y = \text{Embedding}(y) \in \mathbb{R}^{d_{embed}}$$

2. **Time Embedding**: Convert timestep to embedding
   $$e_t = \text{SinusoidalTimeEmbedding}(t)$$

3. **Projection**: Combine embeddings
   $$e_{cond} = \text{Linear}([e_t; e_y])$$

4. **Feature Injection**: Add to model features at multiple scales
   $$h = h + e_{cond} \text{ (broadcasted)}$$

This allows the model to learn different denoising patterns for each class.

## Architecture: U-Net for Diffusion

### Design Overview

The U-Net processes images with learned spatial information at multiple scales:

```
Input (32×32)
    ↓
Conv (32×32, 64 channels)
    ↓
[Residual Blocks + Attention]
    ↓
Downsample → 16×16 → 8×8
    ↓
Middle Blocks (8×8)
    ↓
Upsample → 16×16 → 32×32
    ↓
[Residual Blocks + Attention]
    ↓
Conv (32×32, 3 channels)
    ↓
Output: Predicted Noise
```

### Components

#### Residual Blocks
Each block applies:
1. GroupNorm → SiLU activation
2. Conv 3×3
3. Add time+class embeddings
4. GroupNorm → SiLU activation
5. Dropout
6. Conv 3×3
7. Residual connection

Equation:
$$h = h_{in} + \text{Conv2}(\text{silu}(\text{norm}(\text{Conv1}(h_{in})) + e_{cond}))$$

#### Attention Mechanism
Multi-head self-attention for spatial correlations:

1. Compute Q, K, V from features
2. Multi-head attention: $\text{Attn}(Q, K, V)$
3. Project back to feature space
4. Residual connection

#### Time Embedding
Sinusoidal positional encoding:

$$e_t^{(i)} = \begin{cases}
\sin(\frac{t}{10000^{2i/d}}) & \text{if } i \text{ even} \\
\cos(\frac{t}{10000^{2(i-1)/d}}) & \text{if } i \text{ odd}
\end{cases}$$

### Key Features

- **Skip Connections**: Between corresponding encoder/decoder levels
- **Multi-Scale Processing**: Attention at 8×8 and 16×16 resolutions
- **Residual Connections**: Throughout architecture for gradient flow
- **Dropout**: Regularization with p=0.1

## Sampling Strategy

### Inference: Reverse Diffusion

Starting from noise $x_T \sim \mathcal{N}(0, I)$, iteratively denoise:

```
x_T (pure noise) → class y
  ↓
t = T-1
  ↓
Predict noise: ε̂ = ε_θ(x_t, t, y)
  ↓
Denoise: x_{t-1} = (x_t - √(1-ᾱ_t)·ε̂) / √ᾱ_t + noise
  ↓
t = t - 1
  ↓
t > 0? → Loop
  ↓
Return x_0 (generated image)
```

### Algorithm: DDPM Sampling

```python
def sample(model, num_steps=1000, class_label=None):
    x_t = randn([3, 32, 32])  # Start from noise
    
    for t in range(num_steps-1, -1, -1):
        # Predict noise
        noise_pred = model(x_t, t, class_label)
        
        # Compute posterior parameters
        mean = posterior_mean_from_noise(x_t, noise_pred, t)
        
        # Add noise (except at t=0)
        if t > 0:
            x_t = mean + sqrt(posterior_variance[t]) * randn_like(x_t)
        else:
            x_t = mean
    
    return x_t  # Generated image
```

### Fast Sampling Option

Full DDPM uses all 1000 steps. Faster variants (DDIM) use subset:

```
DDPM:  1000 steps → High quality, slow
DDIM:  100 steps →  Good quality, 10× faster
```

## Class Imbalance Solution

### Problem
Classifier trained on imbalanced data:
- Biases toward head classes (more training signals)
- Poor performance on tail classes (few samples)
- Overall accuracy misleading

### Solution: Diffusion-Based Oversampling

#### Pipeline

```
Original (Imbalanced):
  Class 1: 500 samples
  Class 2: 450 samples
  ...
  Class 10: 25 samples

     Train Diffusion Model ↓

Generate from Tail Classes:
  Sample 475 images for Class 10
  Sample 450 images for Class 9
  ...

      Combine with Original ↓

Balanced Dataset:
  Class 1: 500 original
  Class 2: 500 (450 original + 50 synthetic)
  ...
  Class 10: 500 (25 original + 475 synthetic)

     Train Classifier ↓

Better Performance! Balanced classes
```

### Advantages

1. **Realistic Synthetic Samples**: Learned from data distribution
2. **Class-Specific Generation**: Each class generates appropriate samples
3. **Better Generalization**: Balanced training improves all classes
4. **No Information Loss**: Original samples retained
5. **Controlled Generation**: Can adjust number of synthetic samples

### Theoretical Foundation

Why this works:

1. **Distribution Learning**: Diffusion learns true class distributions
2. **Mode Coverage**: Generates diverse samples (not just nearest neighbors)
3. **Smooth Interpolation**: Learned denoising creates smooth transitions
4. **Information Preservation**: Generative models preserve class semantics

## Implementation Details

### Hyperparameter Choices

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| ${T}$ (steps) | 1000 | Standard DDPM choice |
| $\beta_{start}$ | 0.0001 | Small initial noise |
| $\beta_{end}$ | 0.02 | High final noise |
| Schedule | Linear | Simple, effective |
| Model channels | 64 | Balanced size/quality |
| Residual blocks | 2 | Per-level depth |
| Attention resolutions | [8, 16] | Lower resolutions for efficiency |
| EMA decay | 0.9999 | Stable model averaging |
| Learning rate | 1e-4 | Conservative, stable |

### Training Considerations

1. **Gradient Clipping**: Prevents training instability
2. **EMA**: Exponential moving average for model stability
3. **Early Stopping**: Save best models during training
4. **Validation**: Use test set to monitor progress

## Evaluation

### Quality Metrics for Generated Samples

Consider evaluating:

1. **Diversity**: Different samples from same class
2. **Fidelity**: Visual similarity to real images
3. **Semantic Consistency**: Generated samples are recognizable
4. **Class-Conditioning**: Generated samples match requested class

In practice, downstream classifier performance is the best evaluation metric.

## References

1. **DDPM**: Ho et al. (2020) "Denoising Diffusion Probabilistic Models"
2. **Improved DDPM**: Nichol & Dhariwal (2021) "Improved Denoising Diffusion Probabilistic Models"
3. **DDIM**: Song et al. (2021) "Denoising Diffusion Implicit Models"
4. **Classifier Guidance**: Dhariwal & Nichol (2021) "Diffusion Models Beat GANs on Image Synthesis"
5. **Conditional Generation**: Gu et al. (2021) "Vector Quantized Diffusion Model for Text-to-Image Synthesis"
