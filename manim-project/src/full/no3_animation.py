#!/usr/bin/env python3
"""
NO3 Cross-Domain Recommendation - Full Manim Animation
17 Scenes based on provided screenplay for CDR paper visualization.

Requirements:
    pip install manim loguru numpy

Usage (render single scene):
    manim -pql no3_animation.py Scene01DataFragmentation
    manim -pqh no3_animation.py Scene01DataFragmentation   # high quality

Usage (render all - see render_all.sh):
    bash render_all.sh

Color coding (from design spec):
    Domain A   -> Blue   (#3B82F6)
    Domain B   -> Green  (#22C55E)
    Matching   -> Yellow (#EAB308)
    Flow       -> Red    (#EF4444)
"""

from __future__ import annotations

import math
import sys
from typing import List, Tuple

import numpy as np
from loguru import logger
from manim import (
    Scene,
    ThreeDScene,
    Circle,
    Dot,
    Dot3D,
    Line,
    Arrow,
    DashedLine,
    DoubleArrow,
    Text,
    MathTex,
    VGroup,
    Axes,
    ThreeDAxes,
    ParametricFunction,
    FadeIn,
    FadeOut,
    Write,
    Create,
    GrowFromCenter,
    Transform,
    ReplacementTransform,
    AnimationGroup,
    LaggedStart,
    Rotate,
    Flash,
    UP,
    DOWN,
    LEFT,
    RIGHT,
    ORIGIN,
    DEGREES,
    PI,
    TAU,
    config,
    Rectangle,
    SurroundingRectangle,
    Brace,
    Sphere,
    BLUE,
    GREEN,
    YELLOW,
    RED,
    WHITE,
    GRAY,
    ORANGE,
    PURPLE,
)

# ============================================================
# LOGGING CONFIGURATION
# ============================================================
logger.remove()
logger.add(
    sys.stderr,
    level="DEBUG",
    colorize=True,
    format=(
        "<green>{time:HH:mm:ss}</green> | "
        "<level>{level:<8}</level> | "
        "<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    ),
)
logger.add(
    "no3_animation_debug.log",
    level="DEBUG",
    rotation="5 MB",
    retention="7 days",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} - {message}",
)

# ============================================================
# COLOR PALETTE (matches design spec)
# ============================================================
BG: str = "#0F172A"        # Dark navy background
C_A: str = "#3B82F6"       # Domain A  - Blue
C_B: str = "#22C55E"       # Domain B  - Green
C_MATCH: str = "#EAB308"   # Matching  - Yellow
C_FLOW: str = "#EF4444"    # Flow      - Red
C_TEXT: str = "#F1F5F9"    # Main text - near white
C_ACC: str = "#A78BFA"     # Accent    - purple
C_LOSS: str = "#F97316"    # Loss      - orange
C_SOFT: str = "#475569"    # Soft link - slate gray

# ============================================================
# TIMING CONSTANTS (seconds)
# ============================================================
TF: float = 0.4   # Fade
TW: float = 0.7   # Write
TM: float = 1.2   # Move / create
TS: float = 0.5   # Short wait
TM2: float = 1.0  # Medium wait
TL: float = 1.5   # Long wait

# ============================================================
# SHARED HELPER FUNCTIONS
# ============================================================


def set_background(scene: Scene) -> None:
    """
    Set scene background to the shared dark navy color.

    Args:
        scene: The Manim Scene whose background will be set.

    Returns:
        None
    """
    logger.debug(f"set_background called for {type(scene).__name__}")
    scene.camera.background_color = BG


def make_title(title: str, subtitle: str = "") -> VGroup:
    """
    Create a bold title with an optional subtitle, anchored to the top edge.

    Args:
        title:    Main title string.
        subtitle: Optional subtitle string (smaller, accent color).

    Returns:
        VGroup containing the title (and subtitle if provided).
    """
    logger.debug(f"make_title: '{title}'")
    t = Text(title, font_size=30, color=C_TEXT, weight="BOLD")
    t.to_edge(UP, buff=0.25)
    if subtitle:
        s = Text(subtitle, font_size=20, color=C_ACC)
        s.next_to(t, DOWN, buff=0.1)
        return VGroup(t, s)
    return VGroup(t)


def make_user_dots(
    center: np.ndarray,
    radius: float,
    count: int,
    color: str,
    seed: int = 0,
) -> List[Dot]:
    """
    Generate user dots scattered randomly inside a circle boundary.

    Args:
        center: Center position [x, y, 0] as numpy array.
        radius: Bounding circle radius.
        count:  Number of dots.
        color:  Hex color string for dot fill.
        seed:   Random seed for reproducibility.

    Returns:
        List of Dot mobjects positioned inside the circle.
    """
    logger.debug(f"make_user_dots: count={count}, radius={radius}, color={color}")
    rng = np.random.default_rng(seed)
    dots: List[Dot] = []
    for _ in range(count):
        angle: float = float(rng.uniform(0.0, 2.0 * math.pi))
        r: float = float(rng.uniform(0.25, radius * 0.70))
        pos = center + np.array([r * math.cos(angle), r * math.sin(angle), 0.0])
        dot = Dot(point=pos, radius=0.10, color=color, fill_opacity=0.95)
        dots.append(dot)
    return dots


def make_platform(
    label: str,
    color: str,
    position: np.ndarray,
    radius: float = 1.7,
) -> Tuple[Circle, Text]:
    """
    Create a platform visualization (circle + label text).

    Args:
        label:    Name of the platform.
        color:    Platform border and label color.
        position: Center of the circle.
        radius:   Circle radius.

    Returns:
        Tuple of (Circle, Text) mobjects.
    """
    logger.debug(f"make_platform: '{label}' at {position}")
    circle = Circle(
        radius=radius, color=color, fill_opacity=0.07, stroke_width=2.5
    )
    circle.move_to(position)
    text = Text(label, font_size=20, color=color)
    text.next_to(circle, UP, buff=0.15)
    return circle, text


def make_insight_box(text: str, width: float = 7.0) -> VGroup:
    """
    Create a highlighted insight box at the bottom of the screen.

    Args:
        text:  Insight text content.
        width: Width of the bounding rectangle.

    Returns:
        VGroup of (Rectangle, Text).
    """
    logger.debug(f"make_insight_box: '{text}'")
    label = Text(
        text,
        font="Comic Sans MS",
        font_size=18,
        color=C_ACC
    )
    box = SurroundingRectangle(label, color=C_ACC, buff=0.2, corner_radius=0.1)
    group = VGroup(box, label)
    group.to_edge(DOWN, buff=0.3)
    return group


# ============================================================
# SCENE 01 - DATA FRAGMENTATION
# ============================================================


class Scene01DataFragmentation(Scene):
    """
    Scene 1: Two isolated platforms, users trapped inside each.
    Shows why data fragmentation is a problem.
    Duration: ~60s
    """

    def construct(self) -> None:
        """Build and animate Scene 01."""
        try:
            logger.info("Scene01DataFragmentation.construct: start")
            set_background(self)

            # --- Title ---
            title = make_title("Problem: Data Fragmentation")
            self.play(FadeIn(title), run_time=TF)

            # --- Platforms ---
            pos_a = np.array([-3.5, 0.0, 0.0])
            pos_b = np.array([3.5, 0.0, 0.0])
            circle_a, label_a = make_platform("Music Platform", C_A, pos_a)
            circle_b, label_b = make_platform("Movie Platform", C_B, pos_b)

            self.play(
                Create(circle_a), Create(circle_b),
                run_time=TM,
            )
            self.play(
                Write(label_a), Write(label_b),
                run_time=TW,
            )
            self.wait(TS)

            # --- User dots inside each platform ---
            dots_a: List[Dot] = make_user_dots(pos_a, 1.7, 6, C_A, seed=1)
            dots_b: List[Dot] = make_user_dots(pos_b, 1.7, 6, C_B, seed=2)

            self.play(
                LaggedStart(
                    *[GrowFromCenter(d) for d in dots_a + dots_b],
                    lag_ratio=0.15,
                ),
                run_time=TM,
            )
            self.wait(TM2)

            # --- Highlight separation: show dashed barrier in the middle ---
            barrier = DashedLine(
                start=np.array([0.0, 2.5, 0.0]),
                end=np.array([0.0, -2.5, 0.0]),
                color=C_FLOW,
                dash_length=0.15,
                stroke_width=2,
            )
            barrier_label = Text("No Connection", font_size=18, color=C_FLOW)
            barrier_label.next_to(barrier, RIGHT, buff=0.15)

            self.play(Create(barrier), Write(barrier_label), run_time=TW)
            self.wait(TM2)

            # --- Insight ---
            insight = make_insight_box(
                "Result: data sparsity + cold-start problem"
            )
            self.play(FadeIn(insight), run_time=TF)
            self.wait(TL)
            logger.info("Scene01DataFragmentation.construct: done")

        except Exception as exc:
            logger.error(f"Scene01 failed: {exc}")
            raise


# ============================================================
# SCENE 02 - CDR LANDSCAPE
# ============================================================


class Scene02CDRLandscape(Scene):
    """
    Scene 2: Three CDR overlap cases.
    Highlights Case 3 (no overlap) as the target setting.
    Duration: ~60s
    """

    def construct(self) -> None:
        """Build and animate Scene 02."""
        try:
            logger.info("Scene02CDRLandscape.construct: start")
            set_background(self)

            title = make_title("Cross-Domain Recommendation Landscape")
            self.play(FadeIn(title), run_time=TF)

            # --- Three cases laid out horizontally ---
            cases: List[VGroup] = []
            case_data = [
                ("User Overlap", C_A, True, False),
                ("Item Overlap", C_B, False, True),
                ("No Overlap", C_FLOW, False, False),
            ]
            x_positions: List[float] = [-4.2, 0.0, 4.2]

            for i, ((case_label, case_color, user_ov, item_ov), x_pos) in enumerate(
                zip(case_data, x_positions)
            ):
                case_group = self._build_case_visual(
                    case_label, case_color, x_pos, user_ov, item_ov
                )
                cases.append(case_group)

            self.play(
                LaggedStart(*[FadeIn(c) for c in cases], lag_ratio=0.35),
                run_time=TM,
            )
            self.wait(TM2)

            # --- Zoom emphasis on Case 3 (no overlap) ---
            highlight_box = SurroundingRectangle(
                cases[2], color=C_FLOW, buff=0.15, stroke_width=3
            )
            focus_label = Text("Our target setting", font_size=20, color=C_FLOW)
            focus_label.next_to(highlight_box, DOWN, buff=0.2)

            self.play(Create(highlight_box), Write(focus_label), run_time=TW)
            self.wait(TM2)

            insight = make_insight_box("Most methods assume overlap - this paper tackles the hardest case")
            self.play(FadeIn(insight), run_time=TF)
            self.wait(TL)
            logger.info("Scene02CDRLandscape.construct: done")

        except Exception as exc:
            logger.error(f"Scene02 failed: {exc}")
            raise

    def _build_case_visual(
        self,
        label: str,
        color: str,
        x_center: float,
        user_overlap: bool,
        item_overlap: bool,
    ) -> VGroup:
        """
        Build a single CDR case visualization column.

        Args:
            label:        Case label text.
            color:        Primary color for this case.
            x_center:     Horizontal center position.
            user_overlap: Whether to draw user-overlap lines.
            item_overlap: Whether to draw item-overlap lines.

        Returns:
            VGroup with all elements for this case.
        """
        logger.debug(f"_build_case_visual: '{label}' at x={x_center}")
        elements: List = []

        # Case title
        case_title = Text(label, font_size=16, color=color)
        case_title.move_to(np.array([x_center, 2.0, 0.0]))
        elements.append(case_title)

        # Left mini-circle (Domain A)
        c_left = Circle(radius=0.6, color=C_A, fill_opacity=0.08, stroke_width=1.5)
        c_left.move_to(np.array([x_center - 0.75, 0.5, 0.0]))
        elements.append(c_left)

        # Right mini-circle (Domain B)
        c_right = Circle(radius=0.6, color=C_B, fill_opacity=0.08, stroke_width=1.5)
        c_right.move_to(np.array([x_center + 0.75, 0.5, 0.0]))
        elements.append(c_right)

        # Dots inside circles
        for dx in [-0.25, 0.1]:
            d = Dot(np.array([x_center - 0.75 + dx, 0.5, 0.0]), radius=0.07, color=C_A)
            elements.append(d)
        for dx in [-0.1, 0.25]:
            d = Dot(np.array([x_center + 0.75 + dx, 0.5, 0.0]), radius=0.07, color=C_B)
            elements.append(d)

        # Overlap connections if required
        if user_overlap:
            line = Line(
                np.array([x_center - 0.1, 0.5, 0.0]),
                np.array([x_center + 0.1, 0.5, 0.0]),
                color=C_MATCH, stroke_width=2,
            )
            elements.append(line)

        if item_overlap:
            line = Line(
                np.array([x_center - 0.75, 0.1, 0.0]),
                np.array([x_center + 0.75, 0.1, 0.0]),
                color=C_MATCH, stroke_width=2, stroke_opacity=0.6,
            )
            elements.append(line)

        return VGroup(*elements)


# ============================================================
# SCENE 03 - NO3 SETTING
# ============================================================


class Scene03NO3Setting(Scene):
    """
    Scene 3: Three constraints that define the NO3 (No Overlap x3) setting.
    Bullets appear one by one, then dissolve into two separated clouds.
    Duration: ~60s
    """

    def construct(self) -> None:
        """Build and animate Scene 03."""
        try:
            logger.info("Scene03NO3Setting.construct: start")
            set_background(self)

            title = make_title("The NO3 Setting")
            self.play(FadeIn(title), run_time=TF)
            self.wait(TS)

            # --- Three constraint bullets ---
            constraints: List[Text] = [
                Text("No user overlap", font_size=32, color=C_TEXT),
                Text("No item overlap", font_size=32, color=C_TEXT),
                Text("No side information", font_size=32, color=C_TEXT),
            ]
            for i, c in enumerate(constraints):
                c.move_to(np.array([0.0, 1.0 - i * 1.2, 0.0]))

            for c in constraints:
                self.play(Write(c), run_time=TW)
                self.wait(TS)

            self.wait(TM2)

            # --- Fade constraints out, show two separated cloud shapes ---
            cloud_a_label = Text("Domain A", font_size=22, color=C_A)
            cloud_a_label.move_to(np.array([-3.5, 0.0, 0.0]))
            cloud_b_label = Text("Domain B", font_size=22, color=C_B)
            cloud_b_label.move_to(np.array([3.5, 0.0, 0.0]))

            ellipse_a = Circle(radius=1.3, color=C_A, fill_opacity=0.10, stroke_width=2)
            ellipse_a.move_to(np.array([-3.5, 0.0, 0.0]))
            ellipse_b = Circle(radius=1.3, color=C_B, fill_opacity=0.10, stroke_width=2)
            ellipse_b.move_to(np.array([3.5, 0.0, 0.0]))

            self.play(
                *[FadeOut(c) for c in constraints],
                run_time=TF,
            )
            self.play(
                FadeIn(ellipse_a), FadeIn(ellipse_b),
                FadeIn(cloud_a_label), FadeIn(cloud_b_label),
                run_time=TM,
            )
            self.wait(TM2)

            insight = make_insight_box("Closest to real-world privacy constraints")
            self.play(FadeIn(insight), run_time=TF)
            self.wait(TL)
            logger.info("Scene03NO3Setting.construct: done")

        except Exception as exc:
            logger.error(f"Scene03 failed: {exc}")
            raise


# ============================================================
# SCENE 04 - LEARNING OBJECTIVE
# ============================================================


class Scene04LearningObjective(Scene):
    """
    Scene 4: Multi-task learning objective.
    Shows L = L1 + L2 being constructed from two domain losses.
    Duration: ~60s
    """

    def construct(self) -> None:
        """Build and animate Scene 04."""
        try:
            logger.info("Scene04LearningObjective.construct: start")
            set_background(self)

            title = make_title("Learning Objective", "Multi-task Learning")
            self.play(FadeIn(title), run_time=TF)
            self.wait(TS)

            # --- Domain boxes ---
            d1_box = Rectangle(width=1.6, height=0.8, color=C_A, fill_opacity=0.15)
            d1_box.move_to(np.array([-4.0, 0.5, 0.0]))
            d1_label = Text("D1", color=C_A, font_size=36, weight="BOLD")
            d1_label.move_to(d1_box.get_center())

            d2_box = Rectangle(width=1.6, height=0.8, color=C_B, fill_opacity=0.15)
            d2_box.move_to(np.array([-4.0, -0.8, 0.0]))
            d2_label = Text("D2", color=C_B, font_size=36, weight="BOLD")
            d2_label.move_to(d2_box.get_center())

            # --- Loss boxes ---
            l1_box = Rectangle(width=1.6, height=0.8, color=C_A, fill_opacity=0.15)
            l1_box.move_to(np.array([0.0, 0.5, 0.0]))
            l1_label = Text("L1", color=C_A, font_size=36, weight="BOLD")
            l1_label.move_to(l1_box.get_center())

            l2_box = Rectangle(width=1.6, height=0.8, color=C_B, fill_opacity=0.15)
            l2_box.move_to(np.array([0.0, -0.8, 0.0]))
            l2_label = Text("L2", color=C_B, font_size=36, weight="BOLD")
            l2_label.move_to(l2_box.get_center())

            # --- Arrows D -> L ---
            arr1 = Arrow(
                start=np.array([-3.2, 0.5, 0.0]),
                end=np.array([-0.8, 0.5, 0.0]),
                color=C_A, buff=0,
            )
            arr2 = Arrow(
                start=np.array([-3.2, -0.8, 0.0]),
                end=np.array([-0.8, -0.8, 0.0]),
                color=C_B, buff=0,
            )

            # --- Final combined loss ---
            l_total = Text(
                "L = L1 + L2",
                color=C_TEXT, font_size=42, weight="BOLD",
            )
            l_total.move_to(np.array([3.5, -0.15, 0.0]))

            arr_combine = Arrow(
                start=np.array([0.8, -0.15, 0.0]),
                end=np.array([2.3, -0.15, 0.0]),
                color=C_MATCH, buff=0,
            )

            # --- Animation sequence ---
            self.play(
                FadeIn(d1_box), FadeIn(d1_label),
                FadeIn(d2_box), FadeIn(d2_label),
                run_time=TM,
            )
            self.play(
                Create(arr1), Create(arr2), run_time=TW
            )
            self.play(
                FadeIn(l1_box), FadeIn(l1_label),
                FadeIn(l2_box), FadeIn(l2_label),
                run_time=TM,
            )
            self.wait(TS)
            self.play(Create(arr_combine), run_time=TW)
            self.play(Write(l_total), run_time=TW)
            self.wait(TM2)

            insight = make_insight_box("Multi-task learning: optimize both domains jointly")
            self.play(FadeIn(insight), run_time=TF)
            self.wait(TL)
            logger.info("Scene04LearningObjective.construct: done")

        except Exception as exc:
            logger.error(f"Scene04 failed: {exc}")
            raise


# ============================================================
# SCENE 05 - HNO3 HARD MATCHING
# ============================================================


class Scene05HNO3(Scene):
    """
    Scene 5: Hard matching (HNO3) via Hungarian Algorithm.
    Each user on Domain A is matched 1-to-1 with a user on Domain B.
    Duration: ~90s
    """

    def construct(self) -> None:
        """Build and animate Scene 05."""
        try:
            logger.info("Scene05HNO3.construct: start")
            set_background(self)

            title = make_title("HNO3: Hard Matching", "Hungarian Algorithm")
            self.play(FadeIn(title), run_time=TF)

            # --- Two columns of user dots ---
            n_users: int = 5
            left_x: float = -2.8
            right_x: float = 2.8
            y_positions: List[float] = [1.6, 0.8, 0.0, -0.8, -1.6]

            dots_a: List[Dot] = []
            dots_b: List[Dot] = []
            for y in y_positions:
                dots_a.append(Dot(np.array([left_x, y, 0.0]), radius=0.13, color=C_A))
                dots_b.append(Dot(np.array([right_x, y, 0.0]), radius=0.13, color=C_B))

            label_a = Text("Domain A", font_size=20, color=C_A)
            label_a.move_to(np.array([left_x, 2.3, 0.0]))
            label_b = Text("Domain B", font_size=20, color=C_B)
            label_b.move_to(np.array([right_x, 2.3, 0.0]))

            self.play(
                LaggedStart(*[GrowFromCenter(d) for d in dots_a + dots_b], lag_ratio=0.1),
                FadeIn(label_a), FadeIn(label_b),
                run_time=TM,
            )
            self.wait(TS)

            # --- Draw 1-to-1 matching lines (yellow) one by one ---
            match_lines: List[Line] = []
            for da, db in zip(dots_a, dots_b):
                line = Line(
                    da.get_center(), db.get_center(),
                    color=C_MATCH, stroke_width=2.5,
                )
                match_lines.append(line)

            for line in match_lines:
                self.play(Create(line), run_time=0.25)
            self.wait(TM2)

            algo_label = Text("Hungarian Algorithm: O(n*3)", font_size=18, color=C_ACC)
            algo_label.move_to(np.array([0.0, -2.5, 0.0]))
            self.play(Write(algo_label), run_time=TW)
            self.wait(TM2)

            label = MathTex(r"\text{Converts no-overlap} \rightarrow \text{overlap via 1-to-1 matching}", font_size=18, color=C_ACC)
            box = SurroundingRectangle(label, color=C_ACC, buff=0.2, corner_radius=0.1)
            insight = VGroup(box, label)
            insight.to_edge(DOWN, buff=0.3)
            self.play(FadeIn(insight), run_time=TF)
            self.wait(TL)
            logger.info("Scene05HNO3.construct: done")

        except Exception as exc:
            logger.error(f"Scene05 failed: {exc}")
            raise


# ============================================================
# SCENE 06 - HNO3 LIMITATION
# ============================================================


class Scene06Limitation(Scene):
    """
    Scene 6: Limitations of hard matching.
    Shows mismatched connections and broken lines.
    Duration: ~60s
    """

    def construct(self) -> None:
        """Build and animate Scene 06."""
        try:
            logger.info("Scene06Limitation.construct: start")
            set_background(self)

            title = make_title("HNO3 Limitation", "Discrete Optimization Problem")
            self.play(FadeIn(title), run_time=TF)

            # --- User columns ---
            n_users: int = 5
            left_x: float = -2.8
            right_x: float = 2.8
            y_positions: List[float] = [1.6, 0.8, 0.0, -0.8, -1.6]

            dots_a: List[Dot] = [
                Dot(np.array([left_x, y, 0.0]), radius=0.13, color=C_A)
                for y in y_positions
            ]
            dots_b: List[Dot] = [
                Dot(np.array([right_x, y, 0.0]), radius=0.13, color=C_B)
                for y in y_positions
            ]

            # Wrong matching permutation (to show errors)
            wrong_perm: List[int] = [2, 4, 0, 3, 1]

            wrong_lines: List[Line] = []
            for i, j in enumerate(wrong_perm):
                line = Line(
                    dots_a[i].get_center(),
                    dots_b[j].get_center(),
                    color=C_FLOW,
                    stroke_width=2.0,
                    stroke_opacity=0.85,
                )
                wrong_lines.append(line)

            self.play(
                LaggedStart(*[GrowFromCenter(d) for d in dots_a + dots_b], lag_ratio=0.08),
                run_time=TM,
            )
            self.play(
                LaggedStart(*[Create(l) for l in wrong_lines], lag_ratio=0.12),
                run_time=TM,
            )
            self.wait(TS)

            # --- Flash to indicate errors ---
            for line in wrong_lines[1:3]:
                self.play(Flash(line.get_center(), color=C_FLOW, flash_radius=0.35), run_time=0.3)

            # --- Problem labels ---
            problems: List[Text] = [
                Text("Highly dependent on initial embedding quality", font_size=16, color=C_FLOW),
                Text("Cannot be trained end-to-end", font_size=16, color=C_FLOW),
                Text("Discrete optimization - lacks robustness", font_size=16, color=C_FLOW),
            ]
            for i, prob in enumerate(problems):
                prob.move_to(np.array([0.0, -2.0 - i * 0.45, 0.0]))
            # Place them below the scene
            prob_group = VGroup(*problems)
            prob_group.move_to(np.array([0.0, -2.3, 0.0]))
            # Stack vertically
            for i, p in enumerate(problems):
                p.move_to(np.array([0.0, -1.9 - i * 0.5, 0.0]))

            self.play(
                LaggedStart(*[FadeIn(p) for p in problems], lag_ratio=0.4),
                run_time=TM,
            )
            self.wait(TL)
            logger.info("Scene06Limitation.construct: done")

        except Exception as exc:
            logger.error(f"Scene06 failed: {exc}")
            raise


# ============================================================
# SCENE 07 - SNO3 SOFT MATCHING
# ============================================================


class Scene07SNO3(Scene):
    """
    Scene 7: Soft matching (SNO3) - distribution alignment.
    Each user connects softly to all users on the other side.
    Duration: ~90s
    """

    def construct(self) -> None:
        """Build and animate Scene 07."""
        try:
            logger.info("Scene07SNO3.construct: start")
            set_background(self)

            title = make_title("SNO3: Soft Matching", "Distribution Alignment")
            self.play(FadeIn(title), run_time=TF)

            # --- User columns ---
            n_users: int = 5
            left_x: float = -2.8
            right_x: float = 2.8
            y_positions: List[float] = [1.5, 0.75, 0.0, -0.75, -1.5]

            dots_a: List[Dot] = [
                Dot(np.array([left_x, y, 0.0]), radius=0.13, color=C_A)
                for y in y_positions
            ]
            dots_b: List[Dot] = [
                Dot(np.array([right_x, y, 0.0]), radius=0.13, color=C_B)
                for y in y_positions
            ]

            label_a = Text("Domain A", font_size=20, color=C_A)
            label_a.move_to(np.array([left_x, 2.3, 0.0]))
            label_b = Text("Domain B", font_size=20, color=C_B)
            label_b.move_to(np.array([right_x, 2.3, 0.0]))

            self.play(
                LaggedStart(*[GrowFromCenter(d) for d in dots_a + dots_b], lag_ratio=0.08),
                FadeIn(label_a), FadeIn(label_b),
                run_time=TM,
            )
            self.wait(TS)

            # --- Draw all soft (semi-transparent) connections ---
            soft_lines: List[Line] = []
            for da in dots_a:
                for db in dots_b:
                    # Opacity encodes probability / soft weight
                    opacity: float = float(
                        np.random.default_rng(
                            abs(int(da.get_y() * 100)) + abs(int(db.get_y() * 100))
                        ).uniform(0.08, 0.35)
                    )
                    line = Line(
                        da.get_center(),
                        db.get_center(),
                        color=C_SOFT,
                        stroke_width=1.2,
                        stroke_opacity=opacity,
                    )
                    soft_lines.append(line)

            self.play(
                LaggedStart(*[Create(l) for l in soft_lines], lag_ratio=0.02),
                run_time=TM,
            )
            self.wait(TM2)

            # --- Caption explaining opacity = probability ---
            opacity_note = Text("Opacity = connection probability", font_size=18, color=C_SOFT)
            opacity_note.move_to(np.array([0.0, -2.5, 0.0]))
            self.play(Write(opacity_note), run_time=TW)
            self.wait(TM2)

            label = MathTex(r"\text{Soft matching: from 1-to-1 mapping } \rightarrow \text{ distribution alignment}", font_size=18, color=C_ACC)
            box = SurroundingRectangle(label, color=C_ACC, buff=0.2, corner_radius=0.1)
            insight = VGroup(box, label)
            insight.to_edge(DOWN, buff=0.3)
            self.play(FadeIn(insight), run_time=TF)
            self.wait(TL)
            logger.info("Scene07SNO3.construct: done")

        except Exception as exc:
            logger.error(f"Scene07 failed: {exc}")
            raise


# ============================================================
# SCENE 08 - SINKHORN INTUITION
# ============================================================


class Scene08SinkhornIntuition(Scene):
    """
    Scene 8: Optimal Transport / Sinkhorn intuition.
    Shows mass (dot size) being transported from source to target distribution.
    Duration: ~90s
    """

    def construct(self) -> None:
        """Build and animate Scene 08."""
        try:
            logger.info("Scene08SinkhornIntuition.construct: start")
            set_background(self)

            title = make_title("Sinkhorn: Optimal Transport Intuition")
            self.play(FadeIn(title), run_time=TF)

            # --- Source distribution (left) - varying dot sizes = mass ---
            source_pos: List[Tuple[float, float, float]] = [
                (-4.0, 1.2, 0.0), (-4.0, 0.0, 0.0), (-4.0, -1.2, 0.0),
            ]
            source_radii: List[float] = [0.25, 0.40, 0.18]  # mass = size

            # --- Target distribution (right) ---
            target_pos: List[Tuple[float, float, float]] = [
                (4.0, 1.5, 0.0), (4.0, 0.3, 0.0),
                (4.0, -0.7, 0.0), (4.0, -1.8, 0.0),
            ]
            target_radii: List[float] = [0.20, 0.32, 0.15, 0.28]

            source_dots: List[Dot] = [
                Dot(np.array(p), radius=r, color=C_A, fill_opacity=0.85)
                for p, r in zip(source_pos, source_radii)
            ]
            target_dots: List[Dot] = [
                Dot(np.array(p), radius=r, color=C_B, fill_opacity=0.85)
                for p, r in zip(target_pos, target_radii)
            ]

            src_label = Text("Source Distribution", font_size=18, color=C_A)
            src_label.move_to(np.array([-4.0, -2.5, 0.0]))
            tgt_label = Text("Target Distribution", font_size=18, color=C_B)
            tgt_label.move_to(np.array([4.0, -2.5, 0.0]))

            size_note = Text("Size = probability mass", font_size=16, color=C_ACC)
            size_note.move_to(np.array([0.0, -2.5, 0.0]))

            self.play(
                LaggedStart(*[GrowFromCenter(d) for d in source_dots], lag_ratio=0.2),
                FadeIn(src_label),
                run_time=TM,
            )
            self.play(
                LaggedStart(*[GrowFromCenter(d) for d in target_dots], lag_ratio=0.2),
                FadeIn(tgt_label),
                run_time=TM,
            )
            self.play(Write(size_note), run_time=TW)
            self.wait(TS)

            # --- Phase 2: Draw transport flow arrows ---
            flow_arrows: List[Arrow] = [
                Arrow(np.array(source_pos[0]), np.array(target_pos[0]),
                      color=C_FLOW, stroke_width=3, buff=0.3),
                Arrow(np.array(source_pos[0]), np.array(target_pos[1]),
                      color=C_FLOW, stroke_width=1.5, buff=0.3, stroke_opacity=0.5),
                Arrow(np.array(source_pos[1]), np.array(target_pos[2]),
                      color=C_FLOW, stroke_width=3, buff=0.3),
                Arrow(np.array(source_pos[2]), np.array(target_pos[3]),
                      color=C_FLOW, stroke_width=2, buff=0.3),
            ]

            flow_label = Text("Transport plan (arrows = flow)", font_size=16, color=C_FLOW)
            flow_label.move_to(np.array([0.0, 2.5, 0.0]))

            self.play(Write(flow_label), run_time=TW)
            self.play(
                LaggedStart(*[Create(a) for a in flow_arrows], lag_ratio=0.2),
                run_time=TM,
            )
            self.wait(TM2)

            label = MathTex(r"\text{Sinkhorn: differentiable OT } \rightarrow \text{ trains end-to-end in deep learning}", font_size=18, color=C_ACC)
            box = SurroundingRectangle(label, color=C_ACC, buff=0.2, corner_radius=0.1)
            insight = VGroup(box, label)
            insight.to_edge(DOWN, buff=0.3)
            self.play(FadeIn(insight), run_time=TF)
            self.wait(TL)
            logger.info("Scene08SinkhornIntuition.construct: done")

        except Exception as exc:
            logger.error(f"Scene08 failed: {exc}")
            raise


# ============================================================
# SCENE 09 - FINAL OBJECTIVE
# ============================================================


class Scene09FinalObjective(Scene):
    """
    Scene 9: Combined final loss = recommendation loss + Sinkhorn loss.
    Both components appear then merge into one equation.
    Duration: ~30s
    """

    def construct(self) -> None:
        """Build and animate Scene 09."""
        try:
            logger.info("Scene09FinalObjective.construct: start")
            set_background(self)

            title = make_title("Final Objective")
            self.play(FadeIn(title), run_time=TF)

            # --- Component 1: Recommendation loss ---
            rec_loss = Text("L rec", font_size=48, color=C_A, weight="BOLD")
            rec_loss.move_to(np.array([-3.0, 0.5, 0.0]))
            rec_desc = Text("Recommendation accuracy", font_size=18, color=C_A)
            rec_desc.next_to(rec_loss, DOWN, buff=0.3)

            # --- Component 2: Sinkhorn loss ---
            sink_loss = Text("lambda * L sink", font_size=48, color=C_B, weight="BOLD")
            sink_loss.move_to(np.array([3.0, 0.5, 0.0]))
            sink_desc = Text("Distribution alignment (Sinkhorn)", font_size=18, color=C_B)
            sink_desc.next_to(sink_loss, DOWN, buff=0.3)

            self.play(
                FadeIn(rec_loss), FadeIn(rec_desc),
                run_time=TM,
            )
            self.wait(TS)
            self.play(
                FadeIn(sink_loss), FadeIn(sink_desc),
                run_time=TM,
            )
            self.wait(TS)

            # --- Merged final equation ---
            final_eq = Text(
                "L = L rec + lambda * L sink",
                font_size=44,
                color=C_TEXT,
                weight="BOLD",
            )
            final_eq.move_to(np.array([0.0, -1.2, 0.0]))

            plus_sign = Text("+", font_size=52, color=C_MATCH, weight="BOLD")
            plus_sign.move_to(np.array([0.0, 0.5, 0.0]))

            self.play(Write(plus_sign), run_time=TW)
            self.wait(TS)
            self.play(Write(final_eq), run_time=TW)
            self.wait(TM2)

            insight = make_insight_box("Representation learning + distribution alignment")
            self.play(FadeIn(insight), run_time=TF)
            self.wait(TL)
            logger.info("Scene09FinalObjective.construct: done")

        except Exception as exc:
            logger.error(f"Scene09 failed: {exc}")
            raise


# ============================================================
# SCENE 10 - KEY INSIGHT
# ============================================================


class Scene10KeyInsight(Scene):
    """
    Scene 10: Three core takeaways from the paper.
    Duration: ~30s
    """

    def construct(self) -> None:
        """Build and animate Scene 10."""
        try:
            logger.info("Scene10KeyInsight.construct: start")
            set_background(self)

            title = make_title("Key Insight")
            self.play(FadeIn(title), run_time=TF)

            # --- Three insight cards ---
            insights: List[Tuple[str, str]] = [
                ("No identity needed", C_A),
                ("Preference is enough", C_B),
                ("Soft matching wins", C_MATCH),
            ]
            cards: List[VGroup] = []
            for i, (text, color) in enumerate(insights):
                t = Text(text, font_size=30, color=color, weight="BOLD")
                t.move_to(np.array([0.0, 1.0 - i * 1.3, 0.0]))
                box = SurroundingRectangle(t, color=color, buff=0.2, corner_radius=0.1)
                cards.append(VGroup(box, t))

            for card in cards:
                self.play(FadeIn(card), run_time=TW)
                self.wait(TS)

            self.wait(TL)
            logger.info("Scene10KeyInsight.construct: done")

        except Exception as exc:
            logger.error(f"Scene10 failed: {exc}")
            raise


# ============================================================
# SCENE 11 - GRADIENT DESCENT
# ============================================================


class Scene11GradientDescent(Scene):
    """
    Scene 11: Gradient descent on a parabolic loss curve.
    A point descends step by step, with gradient arrow shown.
    Duration: ~60s
    """

    def construct(self) -> None:
        """Build and animate Scene 11."""
        try:
            logger.info("Scene11GradientDescent.construct: start")
            set_background(self)

            title = make_title("Gradient Descent", "How the model learns")
            self.play(FadeIn(title), run_time=TF)

            # --- Axes ---
            axes = Axes(
                x_range=[-3.0, 3.0, 1.0],
                y_range=[0.0, 9.0, 2.0],
                x_length=7,
                y_length=4,
                axis_config={"color": C_TEXT, "stroke_width": 1.5},
                tips=False,
            )
            axes.shift(DOWN * 0.5)

            x_label = Text("Parameter", font_size=18, color=C_TEXT)
            x_label.next_to(axes.x_axis, DOWN, buff=0.2)
            y_label = Text("Loss", font_size=18, color=C_TEXT)
            y_label.next_to(axes.y_axis, LEFT, buff=0.2)

            # --- Loss curve: simple parabola ---
            curve = axes.plot(
                lambda x: x ** 2,
                x_range=[-3.0, 3.0],
                color=C_A,
                stroke_width=3,
            )
            curve_label = Text("L(theta) = theta^2", font_size=24, color=C_A, weight="BOLD")
            curve_label.next_to(curve, UP + RIGHT, buff=0.2)

            self.play(Create(axes), FadeIn(x_label), FadeIn(y_label), run_time=TM)
            self.play(Create(curve), Write(curve_label), run_time=TM)
            self.wait(TS)

            # --- Descending point ---
            start_x: float = -2.7
            point = Dot(axes.c2p(start_x, start_x ** 2), radius=0.14, color=C_LOSS)
            self.play(GrowFromCenter(point), run_time=TF)

            # --- Step-by-step descent ---
            x_steps: List[float] = [-2.7, -2.0, -1.4, -0.9, -0.5, -0.2, 0.0]
            for x_val in x_steps[1:]:
                y_val: float = x_val ** 2
                # Gradient arrow at current position
                grad_arrow = Arrow(
                    start=axes.c2p(x_val + 0.5, (x_val + 0.5) ** 2),
                    end=axes.c2p(x_val, y_val),
                    color=C_FLOW,
                    stroke_width=2.5,
                    buff=0.0,
                    max_tip_length_to_length_ratio=0.25,
                )
                self.play(
                    point.animate.move_to(axes.c2p(x_val, y_val)),
                    Create(grad_arrow),
                    run_time=0.4,
                )
                self.play(FadeOut(grad_arrow), run_time=0.15)

            self.wait(TS)

            # --- Convergence indicator ---
            conv_label = Text("Converged at minimum", font_size=18, color=C_MATCH)
            conv_label.next_to(point, RIGHT, buff=0.3)
            self.play(Write(conv_label), run_time=TW)
            self.wait(TM2)

            insight = make_insight_box("HNO3 vs SNO3 differ primarily in their loss surface geometry")
            self.play(FadeIn(insight), run_time=TF)
            self.wait(TL)
            logger.info("Scene11GradientDescent.construct: done")

        except Exception as exc:
            logger.error(f"Scene11 failed: {exc}")
            raise


# ============================================================
# SCENE 12 - LOSS LANDSCAPE COMPARISON
# ============================================================


class Scene12LossLandscape(Scene):
    """
    Scene 12: Split-screen comparison of HNO3 (rough) vs SNO3 (smooth) loss surfaces.
    Duration: ~60s
    """

    def construct(self) -> None:
        """Build and animate Scene 12."""
        try:
            logger.info("Scene12LossLandscape.construct: start")
            set_background(self)

            title = make_title("Loss Landscape Comparison")
            self.play(FadeIn(title), run_time=TF)

            # --- Left axes: HNO3 rough landscape ---
            axes_l = Axes(
                x_range=[-3.0, 3.0, 1.0],
                y_range=[0.0, 10.0, 2.0],
                x_length=5.0,
                y_length=3.5,
                axis_config={"color": C_TEXT, "stroke_width": 1},
                tips=False,
            )
            axes_l.shift(LEFT * 3.2 + DOWN * 0.3)

            # HNO3: rough, non-convex with local minima
            rough_curve = axes_l.plot(
                lambda x: (x ** 2) + 2.0 * np.sin(4.0 * x) + 0.8 * np.cos(7.0 * x) + 2.0,
                x_range=[-3.0, 3.0],
                color=C_FLOW,
                stroke_width=2.5,
            )
            hno3_label = Text("HNO3 - Hard Matching", font_size=18, color=C_FLOW)
            hno3_label.next_to(axes_l, DOWN, buff=0.2)

            # --- Right axes: SNO3 smooth landscape ---
            axes_r = Axes(
                x_range=[-3.0, 3.0, 1.0],
                y_range=[0.0, 10.0, 2.0],
                x_length=5.0,
                y_length=3.5,
                axis_config={"color": C_TEXT, "stroke_width": 1},
                tips=False,
            )
            axes_r.shift(RIGHT * 3.2 + DOWN * 0.3)

            # SNO3: smooth, convex - easy to optimize
            smooth_curve = axes_r.plot(
                lambda x: (x ** 2) + 0.5,
                x_range=[-3.0, 3.0],
                color=C_B,
                stroke_width=2.5,
            )
            sno3_label = Text("SNO3 - Soft Matching", font_size=18, color=C_B)
            sno3_label.next_to(axes_r, DOWN, buff=0.2)

            # --- Divider ---
            divider = DashedLine(
                start=np.array([0.0, 2.0, 0.0]),
                end=np.array([0.0, -2.5, 0.0]),
                color=C_SOFT,
                stroke_width=1.5,
            )

            self.play(
                Create(axes_l), Create(axes_r),
                FadeIn(divider),
                run_time=TM,
            )
            self.play(
                Create(rough_curve), Create(smooth_curve),
                run_time=TM,
            )
            self.play(
                Write(hno3_label), Write(sno3_label),
                run_time=TW,
            )
            self.wait(TS)

            # --- Drop a point on each landscape and show path ---
            point_l = Dot(axes_l.c2p(-2.5, (-2.5) ** 2 + 2.0 * np.sin(-10.0) + 0.8 * np.cos(-17.5) + 2.0),
                          radius=0.13, color=C_LOSS)
            point_r = Dot(axes_r.c2p(-2.5, (-2.5) ** 2 + 0.5),
                          radius=0.13, color=C_MATCH)

            self.play(GrowFromCenter(point_l), GrowFromCenter(point_r), run_time=TF)

            # Rough path: zigzag descent
            rough_x_steps: List[float] = [-2.5, -1.8, -2.1, -1.3, -1.6, -0.8, -0.4, 0.0]
            for x_val in rough_x_steps[1:]:
                y_val = x_val ** 2 + 2.0 * np.sin(4.0 * x_val) + 0.8 * np.cos(7.0 * x_val) + 2.0
                self.play(point_l.animate.move_to(axes_l.c2p(x_val, y_val)), run_time=0.25)

            # Smooth path: direct descent
            smooth_x_steps: List[float] = [-2.5, -1.8, -1.2, -0.6, -0.2, 0.0]
            for x_val in smooth_x_steps[1:]:
                y_val = x_val ** 2 + 0.5
                self.play(point_r.animate.move_to(axes_r.c2p(x_val, y_val)), run_time=0.25)

            self.wait(TM2)

            insight = make_insight_box("Core reason SNO3 outperforms HNO3: smoother optimization landscape")
            self.play(FadeIn(insight), run_time=TF)
            self.wait(TL)
            logger.info("Scene12LossLandscape.construct: done")

        except Exception as exc:
            logger.error(f"Scene12 failed: {exc}")
            raise


# ============================================================
# SCENE 13 - 3D EMBEDDING SPACE
# ============================================================


class Scene13EmbeddingSpace3D(ThreeDScene):
    """
    Scene 13: 3D embedding space with two domain clusters.
    Shows initial separation, HNO3 forced alignment, then SNO3 smooth alignment.
    Duration: ~90s
    """

    def construct(self) -> None:
        """Build and animate Scene 13 in 3D."""
        try:
            logger.info("Scene13EmbeddingSpace3D.construct: start")
            self.camera.background_color = BG

            # --- Camera orientation ---
            self.set_camera_orientation(phi=70 * DEGREES, theta=-45 * DEGREES)
            self.begin_ambient_camera_rotation(rate=0.08)

            # --- 3D Axes ---
            axes = ThreeDAxes(
                x_range=[-4.0, 4.0, 1.0],
                y_range=[-4.0, 4.0, 1.0],
                z_range=[-3.0, 3.0, 1.0],
                x_length=6,
                y_length=6,
                z_length=4,
                axis_config={"color": C_TEXT, "stroke_width": 1},
                tips=False,
            )
            self.play(Create(axes), run_time=TM)
            self.wait(TS)

            # --- Domain A cluster (top-left region) ---
            rng_a = np.random.default_rng(10)
            cluster_a_positions: List[np.ndarray] = [
                np.array([
                    float(rng_a.uniform(-3.5, -1.5)),
                    float(rng_a.uniform(1.0, 3.0)),
                    float(rng_a.uniform(-1.0, 1.0)),
                ])
                for _ in range(8)
            ]
            dots_a: List[Dot3D] = [
                Dot3D(point=pos, radius=0.10, color=C_A)
                for pos in cluster_a_positions
            ]

            # --- Domain B cluster (bottom-right region) ---
            rng_b = np.random.default_rng(20)
            cluster_b_positions: List[np.ndarray] = [
                np.array([
                    float(rng_b.uniform(1.5, 3.5)),
                    float(rng_b.uniform(-3.0, -1.0)),
                    float(rng_b.uniform(-1.0, 1.0)),
                ])
                for _ in range(8)
            ]
            dots_b: List[Dot3D] = [
                Dot3D(point=pos, radius=0.10, color=C_B)
                for pos in cluster_b_positions
            ]

            self.play(
                LaggedStart(*[GrowFromCenter(d) for d in dots_a + dots_b], lag_ratio=0.08),
                run_time=TM,
            )
            self.wait(TM2)

            # --- Phase 2: Soft alignment - move clusters closer together ---
            aligned_a_positions: List[np.ndarray] = [
                np.array([
                    pos[0] * 0.4,
                    pos[1] * 0.4,
                    pos[2] * 0.7,
                ])
                for pos in cluster_a_positions
            ]
            aligned_b_positions: List[np.ndarray] = [
                np.array([
                    pos[0] * 0.4,
                    pos[1] * 0.4,
                    pos[2] * 0.7,
                ])
                for pos in cluster_b_positions
            ]

            self.play(
                *[
                    da.animate.move_to(aligned_a)
                    for da, aligned_a in zip(dots_a, aligned_a_positions)
                ],
                *[
                    db.animate.move_to(aligned_b)
                    for db, aligned_b in zip(dots_b, aligned_b_positions)
                ],
                run_time=TM * 1.5,
            )
            self.wait(TM2)
            self.stop_ambient_camera_rotation()
            self.wait(TL)
            logger.info("Scene13EmbeddingSpace3D.construct: done")

        except Exception as exc:
            logger.error(f"Scene13 failed: {exc}")
            raise


# ============================================================
# SCENE 14 - SINKHORN CONVERGENCE
# ============================================================


class Scene14SinkhornConvergence(Scene):
    """
    Scene 14: Matrix heatmap showing Sinkhorn iterative convergence.
    Row normalization -> column normalization -> repeat -> stable transport plan.
    Duration: ~90s
    """

    # Number of users per domain shown in the matrix
    N_USERS: int = 5

    def construct(self) -> None:
        """Build and animate Scene 14."""
        try:
            logger.info("Scene14SinkhornConvergence.construct: start")
            set_background(self)

            title = make_title("Sinkhorn Algorithm: Convergence", "Iterative row/column normalization")
            self.play(FadeIn(title), run_time=TF)

            # --- Phase 1: Random initial matrix ---
            rng = np.random.default_rng(99)
            initial_matrix: np.ndarray = rng.uniform(0.1, 1.0, (self.N_USERS, self.N_USERS))
            grid_initial = self._build_heatmap(initial_matrix, np.array([-3.0, 0.0, 0.0]))

            iter_label = Text("Iteration 0: Random initialization", font_size=18, color=C_TEXT)
            iter_label.move_to(np.array([-3.0, -2.0, 0.0]))

            self.play(FadeIn(grid_initial), Write(iter_label), run_time=TM)
            self.wait(TM2)

            # --- Phase 2: After row normalization ---
            row_norm: np.ndarray = initial_matrix / initial_matrix.sum(axis=1, keepdims=True)
            grid_row = self._build_heatmap(row_norm, np.array([-3.0, 0.0, 0.0]))

            new_label = Text("After row normalization", font_size=18, color=C_A)
            new_label.move_to(np.array([-3.0, -2.0, 0.0]))

            self.play(
                ReplacementTransform(grid_initial, grid_row),
                ReplacementTransform(iter_label, new_label),
                run_time=TM,
            )
            self.wait(TS)

            # --- Phase 3: After column normalization ---
            col_norm: np.ndarray = row_norm / row_norm.sum(axis=0, keepdims=True)
            grid_col = self._build_heatmap(col_norm, np.array([-3.0, 0.0, 0.0]))

            col_label = Text("After column normalization", font_size=18, color=C_B)
            col_label.move_to(np.array([-3.0, -2.0, 0.0]))

            self.play(
                ReplacementTransform(grid_row, grid_col),
                ReplacementTransform(new_label, col_label),
                run_time=TM,
            )
            self.wait(TS)

            # --- Phase 4: Converged (simulate several more iterations) ---
            converged: np.ndarray = col_norm.copy()
            for _ in range(8):
                converged = converged / converged.sum(axis=1, keepdims=True)
                converged = converged / converged.sum(axis=0, keepdims=True)

            grid_conv = self._build_heatmap(converged, np.array([-3.0, 0.0, 0.0]))
            conv_label = Text("Converged: stable transport plan", font_size=18, color=C_MATCH)
            conv_label.move_to(np.array([-3.0, -2.0, 0.0]))

            self.play(
                ReplacementTransform(grid_col, grid_conv),
                ReplacementTransform(col_label, conv_label),
                run_time=TM,
            )
            self.wait(TM2)

            # --- Sidebar: matching arrows proportional to transport plan ---
            self._show_transport_arrows(converged)
            self.wait(TM2)

            label = MathTex(r"\text{Sinkhorn = differentiable OT } \rightarrow \text{ enables end-to-end gradient flow}", font_size=18, color=C_ACC)
            box = SurroundingRectangle(label, color=C_ACC, buff=0.2, corner_radius=0.1)
            insight = VGroup(box, label)
            insight.to_edge(DOWN, buff=0.3)
            self.play(FadeIn(insight), run_time=TF)
            self.wait(TL)
            logger.info("Scene14SinkhornConvergence.construct: done")

        except Exception as exc:
            logger.error(f"Scene14 failed: {exc}")
            raise

    def _build_heatmap(
        self,
        matrix: np.ndarray,
        center: np.ndarray,
    ) -> VGroup:
        """
        Build a grid of colored rectangles representing a matrix heatmap.

        Args:
            matrix: 2D numpy array with values in [0, 1] range.
            center: Center position for the heatmap VGroup.

        Returns:
            VGroup of Rectangle mobjects colored by matrix value.
        """
        logger.debug(f"_build_heatmap: matrix shape={matrix.shape}")
        n: int = matrix.shape[0]
        cell_size: float = 0.55
        cells: List[Rectangle] = []
        m_max: float = float(matrix.max())
        for row in range(n):
            for col in range(n):
                value: float = float(matrix[row, col]) / (m_max + 1e-8)
                # Interpolate color from dark (low) to accent (high)
                opacity: float = 0.15 + 0.80 * value
                cell = Rectangle(
                    width=cell_size,
                    height=cell_size,
                    color=C_ACC,
                    fill_opacity=opacity,
                    stroke_width=0.5,
                    stroke_color=C_SOFT,
                )
                x_pos: float = center[0] + (col - n / 2.0 + 0.5) * cell_size
                y_pos: float = center[1] + (n / 2.0 - row - 0.5) * cell_size
                cell.move_to(np.array([x_pos, y_pos, 0.0]))
                cells.append(cell)
        return VGroup(*cells)

    def _show_transport_arrows(self, transport_plan: np.ndarray) -> None:
        """
        Show arrows on the right side of screen proportional to transport plan values.

        Args:
            transport_plan: Converged doubly-stochastic transport matrix.

        Returns:
            None
        """
        logger.debug("_show_transport_arrows: drawing arrows")
        n: int = transport_plan.shape[0]
        left_x: float = 1.8
        right_x: float = 4.5
        y_positions: List[float] = [1.5 - i * 0.7 for i in range(n)]

        arrows: List[Arrow] = []
        threshold: float = 0.15  # Only draw arrows above this weight
        for i in range(n):
            for j in range(n):
                weight: float = float(transport_plan[i, j])
                if weight > threshold:
                    arr = Arrow(
                        start=np.array([left_x, y_positions[i], 0.0]),
                        end=np.array([right_x, y_positions[j], 0.0]),
                        color=C_FLOW,
                        stroke_width=1.0 + 3.0 * weight,
                        stroke_opacity=0.4 + 0.6 * weight,
                        buff=0.1,
                        max_tip_length_to_length_ratio=0.2,
                    )
                    arrows.append(arr)

        if arrows:
            self.play(
                LaggedStart(*[Create(a) for a in arrows], lag_ratio=0.05),
                run_time=TM,
            )


# ============================================================
# SCENE 15 - GRADIENT VECTOR FIELD IN EMBEDDING SPACE
# ============================================================


class Scene15GradientVectorField(Scene):
    """
    Scene 15: Gradient vector field showing how embeddings are pushed.
    Simulated in 2D with a grid of arrows representing the gradient field.
    Duration: ~90s
    """

    def construct(self) -> None:
        """Build and animate Scene 15."""
        try:
            logger.info("Scene15GradientVectorField.construct: start")
            set_background(self)

            title = make_title("Gradient Vector Field in Embedding Space")
            self.play(FadeIn(title), run_time=TF)

            # --- Initial separated clusters ---
            rng_a = np.random.default_rng(5)
            rng_b = np.random.default_rng(6)

            pos_a: List[np.ndarray] = [
                np.array([float(rng_a.uniform(-4.0, -2.0)),
                          float(rng_a.uniform(-0.5, 1.5)), 0.0])
                for _ in range(7)
            ]
            pos_b: List[np.ndarray] = [
                np.array([float(rng_b.uniform(2.0, 4.0)),
                          float(rng_b.uniform(-1.5, 0.5)), 0.0])
                for _ in range(7)
            ]

            dots_a: List[Dot] = [
                Dot(p, radius=0.12, color=C_A) for p in pos_a
            ]
            dots_b: List[Dot] = [
                Dot(p, radius=0.12, color=C_B) for p in pos_b
            ]

            self.play(
                LaggedStart(*[GrowFromCenter(d) for d in dots_a + dots_b], lag_ratio=0.08),
                run_time=TM,
            )
            self.wait(TS)

            # --- Phase 2: Gradient field arrows (grid of small arrows) ---
            field_arrows: List[Arrow] = self._build_gradient_field()
            field_label = Text("Gradient field pushes embeddings toward alignment", font_size=16, color=C_FLOW)
            field_label.move_to(np.array([0.0, -2.5, 0.0]))

            self.play(
                LaggedStart(*[Create(a) for a in field_arrows], lag_ratio=0.02),
                Write(field_label),
                run_time=TM,
            )
            self.wait(TS)

            # --- Phase 3: Embeddings move along the gradient field ---
            target_a: List[np.ndarray] = [
                np.array([p[0] * 0.35 - 0.5, p[1] * 0.35, 0.0])
                for p in pos_a
            ]
            target_b: List[np.ndarray] = [
                np.array([p[0] * 0.35 + 0.5, p[1] * 0.35, 0.0])
                for p in pos_b
            ]

            self.play(
                *[da.animate.move_to(ta) for da, ta in zip(dots_a, target_a)],
                *[db.animate.move_to(tb) for db, tb in zip(dots_b, target_b)],
                run_time=TM * 1.5,
            )
            self.wait(TM2)

            insight = make_insight_box(
                "Gradient field does not just update parameters - it reshapes embedding geometry"
            )
            self.play(FadeIn(insight), run_time=TF)
            self.wait(TL)
            logger.info("Scene15GradientVectorField.construct: done")

        except Exception as exc:
            logger.error(f"Scene15 failed: {exc}")
            raise

    def _build_gradient_field(self) -> List[Arrow]:
        """
        Build a sparse grid of small arrows pointing from left region to right region.
        Simulates the gradient vector field pushing Domain A embeddings toward Domain B.

        Returns:
            List of Arrow mobjects representing the vector field.
        """
        logger.debug("_build_gradient_field: building grid")
        arrows: List[Arrow] = []
        # Grid from x=-4 to 4, y=-2 to 2, step=1.5
        for gx in np.arange(-3.5, 4.0, 1.5):
            for gy in np.arange(-1.5, 2.0, 1.0):
                # Direction: arrows point right (toward alignment)
                dx: float = 0.35 if gx < 0 else -0.35
                dy: float = -gy * 0.1  # slight vertical pull toward center
                arr = Arrow(
                    start=np.array([gx, gy, 0.0]),
                    end=np.array([gx + dx, gy + dy, 0.0]),
                    color=C_FLOW,
                    stroke_width=1.2,
                    stroke_opacity=0.35,
                    buff=0.0,
                    max_tip_length_to_length_ratio=0.3,
                )
                arrows.append(arr)
        return arrows


# ============================================================
# SCENE 16 - KL DIVERGENCE vs SINKHORN DISTANCE
# ============================================================


class Scene16KLvsSinkhorn(Scene):
    """
    Scene 16: Side-by-side comparison of KL divergence and Sinkhorn distance.
    Shows KL fails when distributions don't overlap; Sinkhorn stays stable.
    Duration: ~90s
    """

    def construct(self) -> None:
        """Build and animate Scene 16."""
        try:
            logger.info("Scene16KLvsSinkhorn.construct: start")
            set_background(self)

            title = make_title("KL Divergence vs Sinkhorn Distance", "Handling non-overlapping distributions")
            self.play(FadeIn(title), run_time=TF)

            # --- Left panel: KL divergence ---
            axes_l = Axes(
                x_range=[-6.0, 6.0, 2.0],
                y_range=[0.0, 0.55, 0.1],
                x_length=5.5,
                y_length=3.5,
                axis_config={"color": C_TEXT, "stroke_width": 1},
                tips=False,
            )
            axes_l.shift(LEFT * 3.3 + DOWN * 0.3)

            kl_header = Text("KL Divergence (Failed)", font_size=20, color=C_FLOW, weight="BOLD")
            kl_header.next_to(axes_l, UP, buff=0.2)

            # Two Gaussians with NO overlap
            source_kl = axes_l.plot(
                lambda x: (1.0 / (0.8 * math.sqrt(2 * PI))) * math.exp(-0.5 * ((x + 3.5) / 0.8) ** 2),
                x_range=[-6.0, 0.0],
                color=C_A, stroke_width=2.5,
            )
            target_kl = axes_l.plot(
                lambda x: (1.0 / (0.8 * math.sqrt(2 * PI))) * math.exp(-0.5 * ((x - 3.5) / 0.8) ** 2),
                x_range=[0.0, 6.0],
                color=C_B, stroke_width=2.5,
            )
            kl_infinity = Text("KL = infinity", font_size=30, color=C_FLOW, weight="BOLD")
            kl_infinity.next_to(axes_l, DOWN, buff=0.2)

            # --- Right panel: Sinkhorn distance ---
            axes_r = Axes(
                x_range=[-6.0, 6.0, 2.0],
                y_range=[0.0, 0.55, 0.1],
                x_length=5.5,
                y_length=3.5,
                axis_config={"color": C_TEXT, "stroke_width": 1},
                tips=False,
            )
            axes_r.shift(RIGHT * 3.3 + DOWN * 0.3)

            sink_header = Text("Sinkhorn Distance (Works)", font_size=20, color=C_B, weight="BOLD")
            sink_header.next_to(axes_r, UP, buff=0.2)

            source_sink = axes_r.plot(
                lambda x: (1.0 / (0.8 * math.sqrt(2 * PI))) * math.exp(-0.5 * ((x + 3.5) / 0.8) ** 2),
                x_range=[-6.0, 0.0],
                color=C_A, stroke_width=2.5,
            )
            target_sink = axes_r.plot(
                lambda x: (1.0 / (0.8 * math.sqrt(2 * PI))) * math.exp(-0.5 * ((x - 3.5) / 0.8) ** 2),
                x_range=[0.0, 6.0],
                color=C_B, stroke_width=2.5,
            )

            # Transport arrows from source to target (Sinkhorn shows flow)
            transport_arrows: List[Arrow] = [
                Arrow(
                    start=axes_r.c2p(-3.5, 0.0),
                    end=axes_r.c2p(3.5, 0.0),
                    color=C_FLOW,
                    stroke_width=2.5,
                    buff=0,
                )
            ]
            sink_value = Text("W = finite (Sinkhorn stable)", font_size=26, color=C_B, weight="BOLD")
            sink_value.next_to(axes_r, DOWN, buff=0.2)

            divider = DashedLine(
                start=np.array([0.0, 2.0, 0.0]),
                end=np.array([0.0, -2.5, 0.0]),
                color=C_SOFT, stroke_width=1.5,
            )

            self.play(
                Create(axes_l), Write(kl_header),
                Create(axes_r), Write(sink_header),
                FadeIn(divider),
                run_time=TM,
            )
            self.play(
                Create(source_kl), Create(target_kl),
                Create(source_sink), Create(target_sink),
                run_time=TM,
            )
            self.wait(TS)

            # Highlight KL infinity issue
            self.play(Write(kl_infinity), run_time=TW)
            self.play(
                LaggedStart(*[Create(a) for a in transport_arrows], lag_ratio=0.2),
                Write(sink_value),
                run_time=TM,
            )
            self.wait(TM2)

            # --- Comparison table ---
            table_items: List[Tuple[str, str, str]] = [
                ("KL Divergence", "Diverges when distributions don't overlap", C_FLOW),
                ("Sinkhorn Distance", "Remains finite and stable", C_B),
                ("KL Divergence", "Density-based measure", C_FLOW),
                ("Sinkhorn Distance", "Geometry-based optimal transport", C_B),
            ]
            table_group: List[Text] = []
            for i, (name, desc, color) in enumerate(table_items):
                row = Text(f"{name}: {desc}", font_size=15, color=color)
                row.move_to(np.array([0.0, -2.0 - i * 0.35, 0.0]))
                table_group.append(row)

            self.play(
                LaggedStart(*[FadeIn(r) for r in table_group], lag_ratio=0.2),
                run_time=TM,
            )
            self.wait(TL)
            logger.info("Scene16KLvsSinkhorn.construct: done")

        except Exception as exc:
            logger.error(f"Scene16 failed: {exc}")
            raise


# ============================================================
# SCENE 17 - WASSERSTEIN GAN COMPARISON
# ============================================================


class Scene17WassersteinGAN(Scene):
    """
    Scene 17: Conceptual mapping between WGAN and the CDR paper.
    Generator -> Domain A, Real data -> Domain B, Wasserstein -> Sinkhorn.
    Duration: ~60s
    """

    def construct(self) -> None:
        """Build and animate Scene 17."""
        try:
            logger.info("Scene17WassersteinGAN.construct: start")
            set_background(self)

            title = make_title("Connection to Wasserstein GAN")
            self.play(FadeIn(title), run_time=TF)

            # --- WGAN side (left column) ---
            wgan_label = Text("Wasserstein GAN", font_size=22, color=C_TEXT, weight="BOLD")
            wgan_label.move_to(np.array([-3.5, 2.2, 0.0]))
            self.play(Write(wgan_label), run_time=TW)

            wgan_items: List[Tuple[str, str]] = [
                ("Generator", C_A),
                ("Real data", C_B),
                ("Discriminator", C_ACC),
                ("Wasserstein distance", C_MATCH),
            ]
            wgan_mobjects: List[VGroup] = []
            for i, (item_text, color) in enumerate(wgan_items):
                t = Text(item_text, font_size=20, color=color)
                box = SurroundingRectangle(t, color=color, buff=0.15, corner_radius=0.08)
                group = VGroup(box, t)
                group.move_to(np.array([-3.5, 1.2 - i * 0.95, 0.0]))
                wgan_mobjects.append(group)

            self.play(
                LaggedStart(*[FadeIn(m) for m in wgan_mobjects], lag_ratio=0.2),
                run_time=TM,
            )
            self.wait(TS)

            # --- Mapping arrows (center) ---
            map_arrows: List[Arrow] = []
            for wg_mob in wgan_mobjects:
                arr = Arrow(
                    start=np.array([-1.8, wg_mob.get_center()[1], 0.0]),
                    end=np.array([-0.2, wg_mob.get_center()[1], 0.0]),
                    color=C_SOFT,
                    stroke_width=1.5,
                    buff=0.0,
                )
                map_arrows.append(arr)

            self.play(
                LaggedStart(*[Create(a) for a in map_arrows], lag_ratio=0.2),
                run_time=TM,
            )

            # --- Paper CDR side (right column) ---
            cdr_label = Text("SNO3 Paper (CDR)", font_size=22, color=C_TEXT, weight="BOLD")
            cdr_label.move_to(np.array([3.5, 2.2, 0.0]))
            self.play(Write(cdr_label), run_time=TW)

            cdr_items: List[Tuple[str, str]] = [
                ("Domain A", C_A),
                ("Domain B", C_B),
                ("Sinkhorn loss", C_ACC),
                ("Sinkhorn distance", C_MATCH),
            ]
            cdr_mobjects: List[VGroup] = []
            for i, (item_text, color) in enumerate(cdr_items):
                t = Text(item_text, font_size=20, color=color)
                box = SurroundingRectangle(t, color=color, buff=0.15, corner_radius=0.08)
                group = VGroup(box, t)
                group.move_to(np.array([3.5, 1.2 - i * 0.95, 0.0]))
                cdr_mobjects.append(group)

            self.play(
                LaggedStart(*[FadeIn(m) for m in cdr_mobjects], lag_ratio=0.2),
                run_time=TM,
            )
            self.wait(TM2)

            # --- Concluding note ---
            conclusion = MathTex(
                r"\text{SNO3 = implicit Wasserstein alignment } \rightarrow \text{ stable training}",
                font_size=18, color=C_MATCH,
            )
            conclusion.move_to(np.array([0.0, -2.6, 0.0]))
            self.play(Write(conclusion), run_time=TW)
            self.wait(TL)
            logger.info("Scene17WassersteinGAN.construct: done")

        except Exception as exc:
            logger.error(f"Scene17 failed: {exc}")
            raise


# ============================================================
# ENTRY POINT - list all scenes for reference
# ============================================================
ALL_SCENES: List[str] = [
    "Scene01DataFragmentation",
    "Scene02CDRLandscape",
    "Scene03NO3Setting",
    "Scene04LearningObjective",
    "Scene05HNO3",
    "Scene06Limitation",
    "Scene07SNO3",
    "Scene08SinkhornIntuition",
    "Scene09FinalObjective",
    "Scene10KeyInsight",
    "Scene11GradientDescent",
    "Scene12LossLandscape",
    "Scene13EmbeddingSpace3D",
    "Scene14SinkhornConvergence",
    "Scene15GradientVectorField",
    "Scene16KLvsSinkhorn",
    "Scene17WassersteinGAN",
]

if __name__ == "__main__":
    logger.info("no3_animation.py loaded. Use manim CLI to render scenes.")
    logger.info(f"Available scenes: {ALL_SCENES}")