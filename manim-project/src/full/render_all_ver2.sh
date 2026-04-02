#!/bin/bash

################################################################################
# render_all_ver2.sh - Simple batch render script for src.py
#
# Usage:
#   bash render_all_ver2.sh                    # Default: medium quality
#   bash render_all_ver2.sh -q high            # High quality (1080p)
#   bash render_all_ver2.sh -q low             # Low quality (480p, fastest)
#   bash render_all_ver2.sh -q 4k              # Ultra HD (2160p)
#   bash render_all_ver2.sh -p                 # Preview after render
#   bash render_all_ver2.sh -q high -p         # High quality with preview
#
# Options:
#   -q, --quality {low,med,high,4k}   Render quality (default: med)
#   -p, --preview                     Auto-open video after render
#   --no-cache, --disable-caching     Disable manim caching
#   -v, --verbose                     Print detailed debug information
#   -h, --help                        Show this help message
#
################################################################################

set -e  # Exit on error

# ============================================================
# Configuration
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_FILE="src.py"
SCENE_NAME="CDRPresentation"
QUALITY="med"
QUALITY_FLAG="-qm"
PREVIEW=false
NO_CACHE=""
VERBOSE=false

# Timing estimates (in seconds)
AVG_RENDER_TIME_LOW=8
AVG_RENDER_TIME_MED=25
AVG_RENDER_TIME_HIGH=80
AVG_RENDER_TIME_4K=300

# ============================================================
# Helper Functions
# ============================================================

print_help() {
    cat << EOF
render_all_ver2.sh - Simple batch render script for CDRPresentation (src.py)

USAGE:
    bash render_all_ver2.sh [OPTIONS]

OPTIONS:
    -q, --quality {low,med,high,4k}
        Render quality (default: med)
        - low  : 480p@15fps  (~8s)
        - med  : 720p@30fps  (~25s)
        - high : 1080p@60fps (~80s)
        - 4k   : 2160p@60fps (~300s)

    -p, --preview
        Auto-open video after render (manim -p flag)

    --no-cache, --disable-caching
        Disable manim output caching (slower, but fresh renders)

    -v, --verbose
        Print detailed debug information

    -h, --help
        Show this help message

EXAMPLES:
    # Default: medium quality
    bash render_all_ver2.sh

    # Quick test: low quality only
    bash render_all_ver2.sh -q low

    # High quality with preview
    bash render_all_ver2.sh -q high -p

    # High quality, fresh render (no cache)
    bash render_all_ver2.sh -q high --no-cache -p

RENDER TIME ESTIMATES:
    -q low  : ~8 seconds
    -q med  : ~25 seconds
    -q high : ~80 seconds
    -q 4k   : ~300 seconds (5 minutes)

NOTES:
    - Scene: CDRPresentation (5 sub-scenes)
    - Output saved to: media/videos/src/{quality}/
    - Use -p flag to auto-open video in default player

EOF
}

log_info() {
    echo "[INFO] $1"
}

log_warn() {
    echo "[WARN] $1"
}

log_error() {
    echo "[ERROR] $1"
}

log_success() {
    echo "[✓] $1"
}

log_debug() {
    if [ "$VERBOSE" = true ]; then
        echo "[DEBUG] $1"
    fi
}

estimate_render_time() {
    case "$QUALITY" in
        low) echo "${AVG_RENDER_TIME_LOW}s" ;;
        med) echo "${AVG_RENDER_TIME_MED}s" ;;
        high) echo "${AVG_RENDER_TIME_HIGH}s (~1min 20s)" ;;
        4k) echo "${AVG_RENDER_TIME_4K}s (~5min)" ;;
    esac
}

# ============================================================
# Parse Arguments
# ============================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        -q|--quality)
            QUALITY="$2"
            case "$QUALITY" in
                low) QUALITY_FLAG="-ql" ;;
                med) QUALITY_FLAG="-qm" ;;
                high) QUALITY_FLAG="-qh" ;;
                4k) QUALITY_FLAG="-qk" ;;
                *)
                    log_error "Unknown quality: $QUALITY"
                    echo "Valid options: low, med, high, 4k"
                    exit 1
                    ;;
            esac
            shift 2
            ;;
        -p|--preview)
            PREVIEW=true
            shift
            ;;
        --no-cache|--disable-caching)
            NO_CACHE="--disable_caching"
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -h|--help)
            print_help
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            print_help
            exit 1
            ;;
    esac
done

# ============================================================
# Main Execution
# ============================================================

cd "$SCRIPT_DIR" || exit 1

log_info "CDR Presentation - Batch Render Script (v2)"
log_info "=========================================="

# Verify Python file exists
if [ ! -f "$PYTHON_FILE" ]; then
    log_error "File not found: $PYTHON_FILE"
    exit 1
fi

log_info "Manim Python file: $PYTHON_FILE"
log_info "Scene name: $SCENE_NAME"
log_info "Quality setting: $QUALITY ($QUALITY_FLAG)"
ESTIMATED_TIME=$(estimate_render_time)
log_info "Estimated render time: $ESTIMATED_TIME"

if [ "$PREVIEW" = true ]; then
    log_info "Preview enabled: Yes (-p)"
fi

if [ -n "$NO_CACHE" ]; then
    log_info "Caching: Disabled ($NO_CACHE)"
fi

log_info ""

# ============================================================
# Build and Execute Manim Command
# ============================================================

MANIM_CMD="manim $QUALITY_FLAG"

if [ "$PREVIEW" = true ]; then
    MANIM_CMD="$MANIM_CMD -p"
fi

if [ -n "$NO_CACHE" ]; then
    MANIM_CMD="$MANIM_CMD $NO_CACHE"
fi

MANIM_CMD="$MANIM_CMD $PYTHON_FILE $SCENE_NAME"

log_debug "Manim command: $MANIM_CMD"

log_info "Rendering: $SCENE_NAME"
log_info "----------------------------------------"

# Execute render
if eval "$MANIM_CMD"; then
    log_success "Render completed successfully!"
else
    log_error "Render failed with exit code $?"
    exit 1
fi

# ============================================================
# Summary
# ============================================================

echo ""
log_info "=========================================="
log_info "Render complete!"

# Find and list output video
OUTPUT_DIR="media/videos/src"
if [ -d "$OUTPUT_DIR" ]; then
    QUALITY_DIR="${QUALITY}p"
    case "$QUALITY" in
        low) QUALITY_DIR="480p15" ;;
        med) QUALITY_DIR="720p30" ;;
        high) QUALITY_DIR="1080p60" ;;
        4k) QUALITY_DIR="2160p60" ;;
    esac
    
    if [ -d "$OUTPUT_DIR/$QUALITY_DIR" ]; then
        VIDEO_FILE="$OUTPUT_DIR/$QUALITY_DIR/CDRPresentation.mp4"
        if [ -f "$VIDEO_FILE" ]; then
            log_success "Output video: $VIDEO_FILE"
            log_info "File size: $(du -h "$VIDEO_FILE" | cut -f1)"
        fi
    fi
fi

log_info "Done!"
exit 0
