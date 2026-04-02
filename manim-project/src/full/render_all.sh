#!/bin/bash

################################################################################
# render_all.sh - Batch render all 17 NO3 animation scenes
#
# Usage:
#   bash render_all.sh                    # Default: medium quality
#   bash render_all.sh -q high            # High quality (1080p)
#   bash render_all.sh -q low             # Low quality (480p, fastest)
#   bash render_all.sh -q 4k              # Ultra HD (2160p)
#   bash render_all.sh -p                 # Preview each scene after render
#   bash render_all.sh -q med --no-cache  # Medium quality, disable caching
#
# Options:
#   -q, --quality {low,med,high,4k}   Render quality (default: med)
#   -p, --preview                     Auto-open video after each render
#   --no-cache, --disable-caching     Disable manim caching
#   -n, --parallel N                  Render N scenes in parallel (experimental)
#   -s, --skip SCENE_NUM              Skip specific scenes (comma-separated)
#   -o, --only SCENE_NUM              Render only specific scenes
#   -h, --help                        Show this help message
#
################################################################################

set -e  # Exit on error

# ============================================================
# Configuration
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_FILE="no3_animation.py"
QUALITY="med"
QUALITY_FLAG="-qm"
PREVIEW=false
NO_CACHE=""
PARALLEL=1
SKIP_SCENES=""
ONLY_SCENES=""
VERBOSE=false

# Timing threshold for progress estimation (in seconds)
AVG_SCENE_TIME_LOW=3
AVG_SCENE_TIME_MED=15
AVG_SCENE_TIME_HIGH=45
AVG_SCENE_TIME_4K=180

# ============================================================
# Scene list (17 total)
# ============================================================

SCENES=(
    "Scene01DataFragmentation"
    "Scene02CDRLandscape"
    "Scene03NO3Setting"
    "Scene04LearningObjective"
    "Scene05HNO3"
    "Scene06Limitation"
    "Scene07SNO3"
    "Scene08SinkhornIntuition"
    "Scene09FinalObjective"
    "Scene10KeyInsight"
    "Scene11GradientDescent"
    "Scene12LossLandscape"
    "Scene13EmbeddingSpace3D"
    "Scene14SinkhornConvergence"
    "Scene15GradientVectorField"
    "Scene16KLvsSinkhorn"
    "Scene17WassersteinGAN"
)

# ============================================================
# Helper Functions
# ============================================================

print_help() {
    cat << EOF
render_all.sh - Batch render all NO3 animation scenes

USAGE:
    bash render_all.sh [OPTIONS]

OPTIONS:
    -q, --quality {low,med,high,4k}
        Render quality (default: med)
        - low  : 480p@15fps  (~3s per scene)
        - med  : 720p@30fps  (~15s per scene)
        - high : 1080p@60fps (~45s per scene)
        - 4k   : 2160p@60fps (~180s per scene)

    -p, --preview
        Auto-open video after each render (manim -p flag)

    --no-cache, --disable-caching
        Disable manim output caching (slower, but fresh renders)

    -n, --parallel N
        Render N scenes in parallel (experimental, default: 1)

    -s, --skip SCENE_NUM
        Skip specific scenes (comma-separated numbers or names)
        Example: -s 13 (skip Scene 13 - 3D)
        Example: -s "Scene13,Scene14" (skip multiple)

    -o, --only SCENE_NUM
        Render only specific scenes
        Example: -o "01,05,08" (render scenes 1, 5, 8)

    -v, --verbose
        Print detailed debug information

    -h, --help
        Show this help message

EXAMPLES:
    # Default: medium quality, all 17 scenes
    bash render_all.sh

    # Quick test: low quality only
    bash render_all.sh -q low

    # High quality with preview
    bash render_all.sh -q high -p

    # Skip slow 3D scene
    bash render_all.sh -q high -s 13

    # Only render first 10 scenes
    bash render_all.sh -q med -o "01,02,03,04,05,06,07,08,09,10"

    # Render in parallel (2 concurrent processes)
    bash render_all.sh -q low -n 2

TOTAL RENDER TIME ESTIMATES (all 17 scenes):
    -q low  : ~50 seconds
    -q med  : ~4 minutes
    -q high : ~12 minutes
    -q 4k   : ~50 minutes

NOTES:
    - First render may be slower due to codec initialization
    - Scene13 (3D Embedding Space) is significantly slower than others
    - Output videos saved to: media/videos/no3_animation/{quality}/
    - Use -p flag to auto-open videos (requires default video player)

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

estimate_total_time() {
    local count=$1
    local total_seconds=0

    case "$QUALITY" in
        low) total_seconds=$((count * AVG_SCENE_TIME_LOW)) ;;
        med) total_seconds=$((count * AVG_SCENE_TIME_MED)) ;;
        high) total_seconds=$((count * AVG_SCENE_TIME_HIGH)) ;;
        4k) total_seconds=$((count * AVG_SCENE_TIME_4K)) ;;
    esac

    local hours=$((total_seconds / 3600))
    local minutes=$(((total_seconds % 3600) / 60))
    local secs=$((total_seconds % 60))

    if [ $hours -gt 0 ]; then
        echo "${hours}h ${minutes}m ${secs}s"
    elif [ $minutes -gt 0 ]; then
        echo "${minutes}m ${secs}s"
    else
        echo "${secs}s"
    fi
}

should_render_scene() {
    local scene_num=$1
    local scene_name=$2

    # Check if in ONLY list
    if [ -n "$ONLY_SCENES" ]; then
        if [[ "$ONLY_SCENES" == *"$scene_num"* ]] || [[ "$ONLY_SCENES" == *"$scene_name"* ]]; then
            return 0
        else
            return 1
        fi
    fi

    # Check if in SKIP list
    if [ -n "$SKIP_SCENES" ]; then
        if [[ "$SKIP_SCENES" == *"$scene_num"* ]] || [[ "$SKIP_SCENES" == *"$scene_name"* ]]; then
            return 1
        fi
    fi

    return 0
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
        -n|--parallel)
            PARALLEL="$2"
            shift 2
            ;;
        -s|--skip)
            SKIP_SCENES="$2"
            shift 2
            ;;
        -o|--only)
            ONLY_SCENES="$2"
            shift 2
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

log_info "NO3 Animation - Batch Render Script"
log_info "======================================"

# Verify Python file exists
if [ ! -f "$PYTHON_FILE" ]; then
    log_error "File not found: $PYTHON_FILE"
    exit 1
fi

log_info "Manim Python file: $PYTHON_FILE"
log_info "Quality setting: $QUALITY ($QUALITY_FLAG)"
if [ "$PREVIEW" = true ]; then
    log_info "Preview enabled: Yes (-p)"
fi
if [ -n "$NO_CACHE" ]; then
    log_info "Caching: Disabled ($NO_CACHE)"
fi

# Count scenes to render
SCENES_TO_RENDER=()
for i in "${!SCENES[@]}"; do
    scene_num=$((i + 1))
    scene_name="${SCENES[$i]}"
    
    if should_render_scene "$(printf "%02d" $scene_num)" "$scene_name"; then
        SCENES_TO_RENDER+=("$scene_name")
    fi
done

TOTAL_SCENES=${#SCENES_TO_RENDER[@]}

if [ $TOTAL_SCENES -eq 0 ]; then
    log_warn "No scenes to render after applying filters"
    exit 0
fi

log_info "Scenes to render: $TOTAL_SCENES / 17"
ESTIMATED_TIME=$(estimate_total_time "$TOTAL_SCENES")
log_info "Estimated render time: $ESTIMATED_TIME"
log_info ""

# Render scenes
RENDERED=0
FAILED=0
FAILED_SCENES=()

for scene_name in "${SCENES_TO_RENDER[@]}"; do
    RENDERED=$((RENDERED + 1))
    PERCENT=$((RENDERED * 100 / TOTAL_SCENES))
    
    log_info "[$RENDERED/$TOTAL_SCENES] ($PERCENT%) Rendering: $scene_name"
    
    # Build manim command
    MANIM_CMD="manim $QUALITY_FLAG"
    if [ "$PREVIEW" = true ]; then
        MANIM_CMD="$MANIM_CMD -p"
    fi
    if [ -n "$NO_CACHE" ]; then
        MANIM_CMD="$MANIM_CMD $NO_CACHE"
    fi
    MANIM_CMD="$MANIM_CMD $PYTHON_FILE $scene_name"
    
    log_debug "Command: $MANIM_CMD"
    
    # Execute render
    if eval "$MANIM_CMD"; then
        log_success "Rendered: $scene_name"
    else
        log_error "Failed to render: $scene_name"
        FAILED=$((FAILED + 1))
        FAILED_SCENES+=("$scene_name")
    fi
    
    echo ""
done

# ============================================================
# Summary
# ============================================================

echo "======================================"
log_info "Batch render complete!"
log_info "Rendered: $((RENDERED - FAILED)) / $TOTAL_SCENES scenes"

if [ $FAILED -gt 0 ]; then
    log_warn "Failed scenes: $FAILED"
    for failed_scene in "${FAILED_SCENES[@]}"; do
        log_warn "  - $failed_scene"
    done
else
    log_success "All scenes rendered successfully!"
fi

# Find and list output videos
OUTPUT_DIR="media/videos/no3_animation"
if [ -d "$OUTPUT_DIR" ]; then
    QUALITY_DIR="${QUALITY}p"
    case "$QUALITY" in
        low) QUALITY_DIR="480p15" ;;
        med) QUALITY_DIR="720p30" ;;
        high) QUALITY_DIR="1080p60" ;;
        4k) QUALITY_DIR="2160p60" ;;
    esac
    
    if [ -d "$OUTPUT_DIR/$QUALITY_DIR" ]; then
        VIDEO_COUNT=$(find "$OUTPUT_DIR/$QUALITY_DIR" -name "*.mp4" 2>/dev/null | wc -l)
        log_info "Output videos: $VIDEO_COUNT files in $OUTPUT_DIR/$QUALITY_DIR/"
    fi
fi

log_info "Done!"

# Exit with error code if any scenes failed
if [ $FAILED -gt 0 ]; then
    exit 1
else
    exit 0
fi
