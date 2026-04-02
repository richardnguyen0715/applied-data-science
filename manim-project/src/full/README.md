# NO3 Animation - Cross-Domain Recommendation Visualization

Complete Manim animation suite for visualizing the **SNO3: Soft Matching for No-Overlap Cross-Domain Recommendation** paper.

This project contains 17 scenes that progressively build understanding of the CDR (Cross-Domain Recommendation) problem, from data fragmentation through hard matching, soft matching, and the Sinkhorn algorithm.

---

## 📋 Table of Contents

- [Scene Overview](#scene-overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Rendering Options](#rendering-options)
- [Color Scheme & Design](#color-scheme--design)
- [Troubleshooting](#troubleshooting)

---

## 🎬 Scene Overview

### Part 1: Problem Definition & Setting

| Scene | Title | Duration | Key Concepts |
|-------|-------|----------|--------------|
| **01** | Data Fragmentation | 60s | Isolated platforms, user fragmentation, cold-start problem |
| **02** | CDR Landscape | 60s | Three overlap scenarios, highlighting no-overlap case |
| **03** | NO3 Setting | 60s | Three constraints: no user/item overlap, no side info |
| **04** | Learning Objective | 60s | Multi-task learning, L = L₁ + L₂ |

### Part 2: Hard Matching (HNO3)

| Scene | Title | Duration | Key Concepts |
|-------|-------|----------|--------------|
| **05** | HNO3 | 90s | Hungarian algorithm, 1-to-1 matching, discrete optimization |
| **06** | HNO3 Limitation | 60s | Dependency on initial embeddings, non-end-to-end, discrete problems |

### Part 3: Soft Matching & Sinkhorn (SNO3)

| Scene | Title | Duration | Key Concepts |
|-------|-------|----------|--------------|
| **07** | SNO3 | 90s | Soft matching, distribution alignment, probabilistic weights |
| **08** | Sinkhorn Intuition | 90s | Optimal transport, mass transport, probability flow |
| **09** | Final Objective | 30s | Combined loss: recommendation + Sinkhorn alignment |
| **10** | Key Insight | 30s | No identity needed, preference is enough, soft wins |

### Part 4: Deep Dive (Advanced Topics)

| Scene | Title | Duration | Key Concepts |
|-------|-------|----------|--------------|
| **11** | Gradient Descent | 60s | Loss landscape, step-by-step convergence |
| **12** | Loss Landscape Comparison | 60s | HNO3 (rough) vs SNO3 (smooth) comparison |
| **13** | 3D Embedding Space | 90s | Cluster separation and alignment in 3D space |
| **14** | Sinkhorn Convergence | 90s | Matrix normalization, iterative algorithm, transport plan |
| **15** | Gradient Vector Field | 90s | Field geometry, embedding updates, space reshaping |
| **16** | KL vs Sinkhorn | 90s | KL divergence limitations, Sinkhorn stability |
| **17** | Wasserstein GAN | 60s | Connection to WGAN, Wasserstein distance |

**Total Duration:** ~15 minutes of animations

---

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- FFmpeg
- LaTeX (optional, for advanced math rendering)

### Step 1: Install Manim

```bash
pip install manim
```

### Step 2: Install System Dependencies

**macOS:**
```bash
brew install py3cairo ffmpeg pango pkg-config
# Optional: LaTeX for advanced math text
brew install mactex
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get install libpango-1.0-0 libpango-gobject-0 libgobject-introspection1
sudo apt-get install python3-cairo python3-pil
sudo apt-get install ffmpeg
# Optional: LaTeX
sudo apt-get install texlive-full
```

**Windows:**
- Use [Chocolatey](https://chocolatey.org/):
  ```bash
  choco install ffmpeg
  choco install miktex  # LaTeX
  ```
- Or follow [Manim installation guide](https://docs.manim.community/en/stable/installation.html)

### Step 3: Install Additional Python Packages

```bash
pip install loguru numpy
```

### Step 4: Verify Installation

```bash
manim --version
```

---

## 🚀 Quick Start

### Render Single Scene

```bash
cd /Users/tgng_mac/Coding/applied-data-science/manim-project/src/full/

# Low quality (fastest, for testing)
manim -pql no3_animation.py Scene01DataFragmentation

# High quality (best for final output)
manim -pqh no3_animation.py Scene01DataFragmentation
```

### Render All Scenes

Use the provided shell script:

```bash
bash render_all.sh
```

Or manually:

```bash
bash render_all.sh -q high  # High quality
bash render_all.sh -q med   # Medium quality (default)
```

---

## 📊 Rendering Options

### Quality Levels

| Flag | Resolution | FPS | Best For | Render Time |
|------|-----------|-----|----------|-------------|
| `-ql` | 480p | 15 | Development/testing | ~2-5s per scene |
| `-qm` | 720p | 30 | General use | ~10-20s per scene |
| `-qh` | 1080p | 60 | Final output | ~30-60s per scene |
| `-qk` | 2160p | 60 | Ultra HD | ~2-5min per scene |

### Common Manim Flags

```bash
manim [OPTIONS] file.py SceneName

Options:
  -p, --preview              Auto-open video after render
  -q, --quality {l,m,h,k}    Render quality (low/medium/high/4k)
  -o, --output_file FILE     Custom output filename
  --disable_caching          Regenerate all animations (slower)
  -n, --from_animation_number N    Start rendering from animation N
  -s, --save_last_frame      Save last frame as image
```

---

## 📁 Output Structure

After rendering, videos are saved to:

```
media/
└── videos/
    └── no3_animation/
        ├── 480p15/
        │   ├── Scene01DataFragmentation.mp4
        │   ├── Scene02CDRLandscape.mp4
        │   └── ...
        ├── 720p30/
        ├── 1080p60/
        └── 2160p60/
```

Example full path:
```
media/videos/no3_animation/1080p60/Scene05HNO3.mp4
```

---

## 🎨 Color Scheme & Design

The animation uses a consistent color palette inspired by modern data visualization:

| Element | Color | Hex |
|---------|-------|-----|
| Background | Dark Navy | `#0F172A` |
| Domain A | Blue | `#3B82F6` |
| Domain B | Green | `#22C55E` |
| Matching/Connection | Yellow | `#EAB308` |
| Flow/Transport | Red | `#EF4444` |
| Main Text | Near-white | `#F1F5F9` |
| Accent | Purple | `#A78BFA` |
| Loss | Orange | `#F97316` |
| Soft Link | Slate Gray | `#475569` |

### Design Principles

1. **Geometry-first thinking**: Embeddings are spaces; matching is mapping
2. **Motion = meaning**: Smooth motion (continuous optimization) vs. jumps (discrete)
3. **Consistency**: Users = dots, Relations = lines/arrows, Probability = opacity
4. **Local vs. Global**: Gradient field (local force) vs. Sinkhorn (global alignment)

---

## 🔧 Customization

### Modify Scene Parameters

Edit timing constants in `no3_animation.py`:

```python
TF: float = 0.4   # Fade duration
TW: float = 0.7   # Write duration
TM: float = 1.2   # Move/create duration
TS: float = 0.5   # Short wait
TM2: float = 1.0  # Medium wait
TL: float = 1.5   # Long wait
```

### Change Colors

Modify the color palette at the top:

```python
# Color palette
C_A: str = "#3B82F6"       # Domain A - Blue
C_B: str = "#22C55E"       # Domain B - Green
C_MATCH: str = "#EAB308"   # Matching - Yellow
C_FLOW: str = "#EF4444"    # Flow - Red
```

### Adjust Animation Speed

Modify `run_time` parameters in individual scenes or change global timing constants.

---

## 📝 Scene Details

### Scene 01: Data Fragmentation
- **Idea**: Introduce the core problem - users exist on multiple platforms
- **Visuals**: Two platforms with isolated user populations
- **Duration**: 60 seconds
- **Render time (1080p)**: ~40s

### Scene 05: HNO3
- **Idea**: Hard matching solves no-overlap via Hungarian algorithm
- **Visuals**: 1-to-1 matching lines between two user columns
- **Duration**: 90 seconds
- **Render time (1080p)**: ~60s

### Scene 08: Sinkhorn Intuition
- **Idea**: Optimal transport perspective on distribution alignment
- **Visuals**: Mass transport with flow arrows and varying dot sizes
- **Duration**: 90 seconds
- **Render time (1080p)**: ~65s

### Scene 13: 3D Embedding Space
- **Idea**: Visualize alignment in high-dimensional space
- **Visuals**: 3D cluster animation with camera rotation
- **Duration**: 90 seconds
- **Render time (1080p)**: ~90s (slower due to 3D rendering)

(See [Scene Overview](#scene-overview) for complete list)

---

## 🐛 Troubleshooting

### Issue: Command not found: `manim`

**Solution**: Ensure manim is in your PATH:
```bash
which manim
pip show manim | grep Location
```

If not found, reinstall:
```bash
pip uninstall manim
pip install manim
```

### Issue: FFmpeg not found

**Solution**: Install FFmpeg using your package manager:

- macOS: `brew install ffmpeg`
- Linux: `sudo apt-get install ffmpeg`
- Windows: `choco install ffmpeg`

### Issue: ModuleNotFoundError: No module named 'loguru'

**Solution**: Install missing package:
```bash
pip install loguru
```

### Issue: Rendering fails with LaTeX errors

**Solution**: Either install LaTeX or use text-only mode:

Option 1 (Install LaTeX):
```bash
brew install mactex  # macOS
# or
sudo apt-get install texlive-full  # Linux
```

Option 2 (Use plain Text instead of MathTex):
Replace `MathTex` with `Text` in the scene code.

### Issue: Output video is blank or corrupted

**Solution**: 
- Try rendering with lower quality first: `-ql`
- Clear cache: `manim --disable_caching -ql no3_animation.py SceneName`
- Check available disk space

### Issue: Rendering is very slow

**Solution**:
- Use lower quality: `-ql` or `-qm` instead of `-qh` or `-qk`
- Reduce total duration by disabling 3D scene (Scene 13)
- Close other applications

---

## 📚 References

- [Manim Documentation](https://docs.manim.community/)
- [Manim Community](https://github.com/ManimCommunity/manim)
- SNO3 Paper: "Soft Matching for No-Overlap Cross-Domain Recommendation"

---

## 📝 License & Attribution

This animation suite was created to visualize the SNO3 Cross-Domain Recommendation paper.

**Colors & Design Inspired By:**
- Tailwind CSS color palette
- Professional data visualization standards
- Modern online learning platforms

---

## 💬 Notes

- Each scene is fully independent and can be rendered individually
- Rendering time varies based on machine specs and quality settings
- First render may be slower due to FFmpeg codec initialization
- Use `-p` flag to automatically open rendered video in default player

---

## 🎯 Next Steps

1. **Quick test**: `manim -pql no3_animation.py Scene01DataFragmentation`
2. **Batch render**: `bash render_all.sh -q med`
3. **Combine scenes**: Use FFmpeg to concatenate MP4 files:
   ```bash
   ffmpeg -f concat -safe 0 -i filelist.txt -c copy output.mp4
   ```
4. **Add voiceover**: Use Audacity or similar to mix audio track

---

**Last Updated**: April 2026  
**Manim Version**: 0.18.0+  
**Python Version**: 3.8+
