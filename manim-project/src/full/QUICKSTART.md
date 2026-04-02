# Quick Start Guide - NO3 Animation

## 📦 Setup (One-time)

```bash
# Install dependencies
pip install manim loguru numpy

# macOS: Install system dependencies
brew install py3cairo ffmpeg pango pkg-config
```

## 🚀 Render Single Scene

```bash
cd /Users/tgng_mac/Coding/applied-data-science/manim-project/src/full/

# Quick preview (480p, ~3 seconds)
manim -pql no3_animation.py Scene01DataFragmentation

# High quality (1080p, ~40 seconds)
manim -pqh no3_animation.py Scene08SinkhornIntuition
```

## 🎬 Render All Scenes

```bash
# Medium quality (default)
bash render_all.sh

# High quality
bash render_all.sh -q high

# Low quality (fastest testing)
bash render_all.sh -q low

# Skip slow 3D scene
bash render_all.sh -q high -s 13

# Only render first 10 scenes
bash render_all.sh -o "01,02,03,04,05,06,07,08,09,10"
```

## ⏱️ Estimated Times

| Quality | Single Scene | All 17 Scenes |
|---------|-------------|---------------|
| Low (480p) | ~3s | ~50s |
| Medium (720p) | ~15s | ~4min |
| High (1080p) | ~45s | ~12min |
| 4K (2160p) | ~180s | ~50min |

*Note: Scene 13 (3D) takes 2-3x longer than others*

## 📁 Output Location

Videos saved to:
```
media/videos/no3_animation/{quality}/Scene{XX}.mp4
```

Example:
```
media/videos/no3_animation/1080p60/Scene01DataFragmentation.mp4
```

## 📚 Full Documentation

See [README.md](README.md) for:
- Complete scene descriptions
- Detailed installation instructions
- Troubleshooting guide
- Color scheme & design principles
- Advanced rendering options

## 🎯 Common Workflows

**1. Test single scene:**
```bash
manim -pql no3_animation.py Scene05HNO3
```

**2. Render high-quality version:**
```bash
bash render_all.sh -q high
```

**3. Develop/debug (skip 3D):**
```bash
bash render_all.sh -q low -s 13
```

**4. Render specific scenes:**
```bash
bash render_all.sh -o "01,05,08,14,16,17"
```

**5. Show script help:**
```bash
bash render_all.sh -h
```

---

**Need help?** See [README.md](README.md) → Troubleshooting section
