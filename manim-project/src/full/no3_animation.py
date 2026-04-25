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
    ChangingDecimal,
    Integer,
    ImageMobject,
    MathTex,
    Scene,
    ThreeDScene,
    Circle,
    Square,
    Dot,
    Dot3D,
    Line,
    Arrow,
    DashedLine,
    DoubleArrow,
    Text,
    Tex,
    MathTex,
    Group,
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
    UL,
    UR,
    DL,    
    DR,
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
C_SOFT: str = "#6C819F"    # Soft link - slate gray

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
    t = Text(title, font_size=30, color=C_TEXT, weight="BOLD", font="Serif")
    t.to_edge(UP, buff=0.4)
    if subtitle:
        s = Text(subtitle, font_size=24, color=C_ACC, font="Calibri")
        s.next_to(t, DOWN, buff=0.15)
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
    text = Text(label, font_size=20, color=color, font="Sans")
    text.next_to(circle, UP, buff=0.25)
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
        font="Sans",
        font_size=20,
        color="#000000",
        weight="BOLD",
    )
    box = SurroundingRectangle(label, color=C_LOSS, buff=0.25, corner_radius=0.1)
    box.set_fill(
        color=C_LOSS,   
        opacity=1.0,
    )

    group = VGroup(box, label)
    group.to_edge(DOWN, buff=0.5)
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
            dots_a: List[Dot] = make_user_dots(pos_a, 1.8, 8, C_A, seed=5)
            dots_b: List[Dot] = make_user_dots(pos_b, 1.8, 8, C_B, seed=7)

            self.play(
                LaggedStart(*[GrowFromCenter(d) for d in dots_a], lag_ratio=0.5),
                LaggedStart(*[GrowFromCenter(d) for d in dots_b], lag_ratio=0.5),
                run_time=TM,
            )
            self.wait(TM2)

            # --- Highlight separation: show dashed barrier in the middle ---
            barrier = DashedLine(
                start=np.array([0.0, 2.2, 0.0]),
                end=np.array([0.0, -2.2, 0.0]),
                color=C_FLOW,
                dash_length=0.15,
                stroke_width=3,
            )
            barrier_label = Text("No Connection", font_size=20, color=C_FLOW, font="Sans")
            barrier_label.next_to(barrier, UP, buff=0.25)

            self.play(Create(barrier), Write(barrier_label), run_time=TW)
            self.wait(TM2)

            # --- Insight ---
            insight = make_insight_box(
                "Result: Data sparsity & Cold-start problem"
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
        try:
            logger.info("Scene02CDRLandscape.construct: start")
            set_background(self)

            title = make_title("Cross-Domain Recommendation Landscape")
            self.play(FadeIn(title), run_time=TF)

            # Case specifications (LaTeX formatted)
            case_specs = [
                ("User Overlap", C_A, True, False, r"U^1 \cap U^2 \neq \emptyset"),
                ("Item Overlap", C_B, False, True, r"I^1 \cap I^2 \neq \emptyset"),
                ("User + Item Overlap", C_MATCH, True, True,
                r"U^1 \cap U^2 \neq \emptyset \land I^1 \cap I^2 \neq \emptyset"),
                ("No Overlap", C_FLOW, False, False,
                r"U^1 \cap U^2 = \emptyset \land I^1 \cap I^2 = \emptyset"),
            ]

            # Stage 1: Show three overlap cases
            x_pos_3 = [-4.8, 0, 4.8]
            first_three = VGroup()
            for i in range(3):
                s = case_specs[i]
                first_three.add(
                    self._build_case_visual(s[0], s[1], x_pos_3[i], s[2], s[3], s[4])
                )

            self.play(LaggedStart(*[FadeIn(c) for c in first_three], lag_ratio=1), run_time=3)
            self.wait(2)

            # Stage 2: Fade out to focus on No Overlap
            self.play(FadeOut(first_three), run_time=1.5)

            # Stage 3: Show No Overlap at center
            s_no = case_specs[3]
            no_overlap_case = self._build_case_visual(
                s_no[0], s_no[1], 0, s_no[2], s_no[3], s_no[4]
            )

            self.play(FadeIn(no_overlap_case), run_time=1.5)

            # Highlight box
            h_box = SurroundingRectangle(
                no_overlap_case, color=C_FLOW, buff=0.3, stroke_width=3
            )

            # Target label
            target_label = Text(
                "Our target setting",
                font_size=20,
                color=C_FLOW,
                weight="BOLD",
                font="Sans"
            )

            shift_vec = LEFT * 1.8

            # Position label based on final shifted layout
            target_label.next_to(
                no_overlap_case.copy().shift(shift_vec), RIGHT, buff=1
            )

            group_left = VGroup(no_overlap_case, h_box)

            self.play(
                group_left.animate.shift(shift_vec),
                FadeIn(target_label),
                run_time=1.5
            )

            self.wait(1.5)

            insight = make_insight_box(
                "Hardest Scenario: No shared information between domains"
            )
            self.play(FadeIn(insight))
            self.wait(3)

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
        formula: str = ""
    ) -> VGroup:

        elements = []

        # Case title
        case_title = Text(
            label, font_size=20, color=color, weight="BOLD", font="Sans"
        )
        case_title.move_to(np.array([x_center, 2.4, 0.0]))
        elements.append(case_title)

        # Origin and axes
        origin = np.array([x_center - 1.2, 1.55, 0.0])
        size = 1.2

        ax_y = Arrow(
            origin,
            origin + DOWN * (size * 2 + 0.5),
            buff=0,
            stroke_width=2,
            tip_length=0.15
        )
        ax_x = Arrow(
            origin,
            origin + RIGHT * (size * 2 + 0.5),
            buff=0,
            stroke_width=2,
            tip_length=0.15
        )
        elements.extend([ax_y, ax_x])

        # Domain 1
        d1 = Square(
            side_length=size, color=C_A, fill_opacity=0.2, stroke_width=2.5
        )
        d1.move_to(origin, aligned_edge=UL)

        d1_text = MathTex(
            r"\mathcal{D}^1", color=C_A, font_size=34
        ).move_to(d1.get_center())

        u1_lab = MathTex(r"U^1", font_size=34).next_to(d1, LEFT, buff=0.15)
        i1_lab = MathTex(r"I^1", font_size=34).next_to(d1, UP, buff=0.15)

        elements.extend([d1, d1_text, u1_lab, i1_lab])

        # Domain 2 position
        if user_overlap and not item_overlap:
            d2_offset = RIGHT * size + DOWN * (size * 0.7)
        elif item_overlap and not user_overlap:
            d2_offset = DOWN * size + RIGHT * (size * 0.7)
        elif user_overlap and item_overlap:
            d2_offset = RIGHT * (size * 0.7) + DOWN * (size * 0.7)
        else:
            d2_offset = RIGHT * (size * 1.1) + DOWN * (size * 1.1)

        d2_origin = origin + d2_offset

        d2 = Square(
            side_length=size, color=C_B, fill_opacity=0.2, stroke_width=2.5
        )
        d2.move_to(d2_origin, aligned_edge=UL)

        d2_text = MathTex(
            r"\mathcal{D}^2", color=C_B, font_size=34
        ).move_to(d2.get_center())

        elements.extend([d2, d2_text])

        # Labels for Domain 2
        if user_overlap and not item_overlap:
            u2_lab = MathTex(r"U^2", font_size=34).next_to(d2, RIGHT, buff=0.15)
            i2_lab = MathTex(r"I^2", font_size=34).next_to(
                d2, UP, buff=0.15 + size * 0.7
            )

        elif item_overlap and not user_overlap:
            u2_lab = MathTex(r"U^2", font_size=34).next_to(
                d2, LEFT, buff=0.15 + size * 0.7
            )
            i2_lab = MathTex(r"I^2", font_size=34).next_to(d2, DOWN, buff=0.15)

        else:
            u2_lab = MathTex(r"U^2", font_size=34).next_to(d2, RIGHT, buff=0.15)
            i2_lab = MathTex(r"I^2", font_size=34).next_to(d2, DOWN, buff=0.15)

        elements.extend([u2_lab, i2_lab])

        # Formula
        if formula:
            f_math = MathTex(formula, font_size=34, color=color)
            f_math.move_to(np.array([x_center, -1.9, 0.0]))
            elements.append(f_math)

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
            icon_path = "assets/remove.png" 

            constraints: List[Group] = []

            texts = [
                "No user overlap",
                "No item overlap",
                "No side information",
            ]

            for i, txt in enumerate(texts):
                text = Text(txt, font_size=32, color=C_TEXT)

                icon = ImageMobject(icon_path).scale(0.15)
                icon.next_to(text, LEFT, buff=0.3)

                group = Group(icon, text)

                group.move_to(np.array([0.0, 1.0 - i * 1.2, 0.0]))
                constraints.append(group)

            # Animate
            for g in constraints:
                self.play(FadeIn(g, shift=RIGHT * 0.3), run_time=TW)
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
            d1_box.move_to(np.array([-4.5, 0.6, 0.0]))

            d2_box = Rectangle(width=1.6, height=0.8, color=C_B, fill_opacity=0.15)
            d2_box.move_to(np.array([-4.5, -0.8, 0.0]))

            # --- Loss boxes ---
            l1_box = Rectangle(width=1.6, height=0.8, color=C_A, fill_opacity=0.15)
            l1_box.move_to(np.array([-1, 0.6, 0.0]))

            l2_box = Rectangle(width=1.6, height=0.8, color=C_B, fill_opacity=0.15)
            l2_box.move_to(np.array([-1, -0.8, 0.0]))
            
            # --- Domain labels (MathTex) ---
            d1_label = MathTex("D_1", color=C_A)
            d1_label.move_to(d1_box.get_center())

            d2_label = MathTex("D_2", color=C_B)
            d2_label.move_to(d2_box.get_center())

            # --- Loss labels (MathTex) ---
            l1_label = MathTex("L_1", color=C_A)
            l1_label.move_to(l1_box.get_center())

            l2_label = MathTex("L_2", color=C_B)
            l2_label.move_to(l2_box.get_center())

            # --- Total box and label ---
            l_total_box = Rectangle(
                width=3.2, height=1,
                color=C_MATCH,
                fill_opacity=0.15,
            )
            l_total_box.move_to(np.array([3.5, -0.1, 0.0]))

            l_total = MathTex("L = L_1 + L_2", color=C_MATCH)
            l_total.move_to(l_total_box.get_center())

            # --- Arrows D -> L ---
            arr1 = Arrow(
                start=np.array([-3.7, 0.6, 0.0]),
                end=np.array([-1.8, 0.6, 0.0]),
                color=C_A, buff=0,
            )
            arr2 = Arrow(
                start=np.array([-3.7, -0.8, 0.0]),
                end=np.array([-1.8, -0.8, 0.0]),
                color=C_B, buff=0,
            )

            # --- Arrows from both L1 and L2 → L_total ---
            arr_l1 = Arrow(
                start=l1_box.get_right(),
                end=l_total.get_left() + UP * 0.2 + 0.25 * LEFT,
                color=C_MATCH,
                buff=0,
            )

            arr_l2 = Arrow(
                start=l2_box.get_right(),
                end=l_total.get_left() + DOWN * 0.2 + 0.25 * LEFT,
                color=C_MATCH,
                buff=0,
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

            self.play(
                Create(arr_l1),
                Create(arr_l2),
                run_time=TW
            )
            self.play(
                FadeIn(l_total),
                Create(l_total_box),
                run_time=TW
            )
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
            y_positions: List[float] = [1.2, 0.4, -0.4, -1.2]

            dots_a: List[Dot] = []
            dots_b: List[Dot] = []
            for y in y_positions:
                dots_a.append(Dot(np.array([left_x, y, 0.0]), radius=0.13, color=C_A))
                dots_b.append(Dot(np.array([right_x, y, 0.0]), radius=0.13, color=C_B))

            label_a = Text("Domain A", font_size=20, color=C_A, font="Sans")
            label_a.move_to(np.array([left_x, 1.9, 0.0]))
            label_b = Text("Domain B", font_size=20, color=C_B, font="Sans")
            label_b.move_to(np.array([right_x, 1.9, 0.0]))

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

            algo_label = MathTex(
                r"\text{Time Complexity: } O(n^3)",
                font_size=32,
                color=C_ACC
            )
            algo_label.move_to(np.array([0.0, -2, 0.0]))

            self.play(Write(algo_label), run_time=TW)
            self.wait(TM2)

            label = "Hard matching: Converts no-overlap to overlap via 1-to-1 matching"
            insight = make_insight_box(label)

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
            y_positions: List[float] = [1.2, 0.4, -0.4, -1.2, -2]

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

            graph_group = VGroup(
                *dots_a,
                *dots_b,
                *wrong_lines
            )

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
                Text("Highly dependent on initial embedding quality", font_size=20, color=C_FLOW),
                Text("Cannot be trained end-to-end", font_size=20, color=C_FLOW),
                Text("Discrete optimization - lacks robustness", font_size=20, color=C_FLOW),
            ]

            for i, p in enumerate(problems):
                p.move_to(np.array([3.3, 0.2 - i * 0.6, 0.0])) 

            self.play(
                graph_group.animate.shift(LEFT * 3.5),
                run_time=TM
            )

            for p in problems:
                self.play(FadeIn(p, shift=RIGHT * 0.3), run_time=0.8)
                self.wait(0.5)

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

            title = make_title("SNO3: Soft Matching")
            subtitle = Text("Distribution Alignment", font_size=24, color=C_ACC, font="Sans")
            subtitle.next_to(title[0], DOWN, buff=0.15)
            title_group = VGroup(title, subtitle)
            self.play(FadeIn(title_group), run_time=TF)

            # --- User columns ---
            left_x: float = -2.8
            right_x: float = 2.8
            y_positions: List[float] = [1.2, 0.4, -0.4, -1.2]

            dots_a: List[Dot] = [
                Dot(np.array([left_x, y, 0.0]), radius=0.13, color=C_A)
                for y in y_positions
            ]
            dots_b: List[Dot] = [
                Dot(np.array([right_x, y, 0.0]), radius=0.13, color=C_B)
                for y in y_positions
            ]

            label_a = Text("Domain A", font_size=20, color=C_A)
            label_a.move_to(np.array([left_x, 1.9, 0.0]))
            label_b = Text("Domain B", font_size=20, color=C_B)
            label_b.move_to(np.array([right_x, 1.9, 0.0]))

            self.play(
                LaggedStart(*[GrowFromCenter(d) for d in dots_a + dots_b], lag_ratio=0.08),
                FadeIn(label_a), FadeIn(label_b),
                run_time=TM,
            )
            self.wait(TS)

            # --- Phase A: hard 1-to-1 matching recap ---
            wrong_perm: List[int] = [2, 0, 3, 1]
            hard_lines: List[Line] = []
            for i, j in enumerate(wrong_perm):
                hard_lines.append(
                    Line(
                        dots_a[i].get_center(),
                        dots_b[j].get_center(),
                        color=C_FLOW,
                        stroke_width=3.0,
                        stroke_opacity=0.95,
                    )
                )

            hard_note = Text("Hard constraint: 1 user <-> 1 user", font_size=20, color=C_FLOW, font="Sans")
            hard_note.move_to(np.array([0.0, -2.0, 0.0]))

            self.play(
                LaggedStart(*[Create(l) for l in hard_lines], lag_ratio=0.12),
                FadeIn(hard_note),
                run_time=TM,
            )
            self.play(
                dots_a[1].animate.shift(UP * 0.08),
                dots_a[1].animate.shift(DOWN * 0.08),
                dots_b[2].animate.shift(DOWN * 0.08),
                dots_b[2].animate.shift(UP * 0.08),
                run_time=TF,
            )
            self.wait(TS)

            # --- Phase B/C: transition to soft bipartite matching with probabilities ---
            rng = np.random.default_rng(77)
            weights: np.ndarray = rng.uniform(0.1, 1.0, (len(dots_a), len(dots_b)))
            weights = weights / weights.sum(axis=1, keepdims=True)

            soft_lines: List[Line] = []
            for i, da in enumerate(dots_a):
                for j, db in enumerate(dots_b):
                    w_ij: float = float(weights[i, j])
                    soft_lines.append(
                        Line(
                            da.get_center(),
                            db.get_center(),
                            color=C_SOFT,
                            stroke_width=1.0 + 2.0 * w_ij,
                            stroke_opacity=0.25 + 0.75 * w_ij,
                        )
                    )

            prob_note = Text("Soft matching: each user connects to many users", font_size=20, color=C_MATCH, font="Sans")
            prob_note.move_to(np.array([0.0, -2.0, 0.0]))

            self.play(
                FadeOut(VGroup(*hard_lines), run_time=TF),
                FadeOut(hard_note, run_time=TF),
                LaggedStart(*[Create(l) for l in soft_lines], lag_ratio=0.02),
                FadeIn(prob_note),
                run_time=TM,
            )
            self.wait(TM2)

            outgoing_labels: List[MathTex] = []
            for j in range(len(dots_b)):
                lab = MathTex(f"{weights[0, j]:.2f}", font_size=24, color=C_MATCH)
                lab.move_to((dots_a[0].get_center() + dots_b[j].get_center()) / 2 + UP * 0.15)
                outgoing_labels.append(lab)

            row_sum = MathTex(r"\sum_j w_{1j} = 1", font_size=34, color=C_MATCH)
            row_sum.move_to(np.array([0.0, -1.95, 0.0]))

            self.play(
                Write(VGroup(*outgoing_labels)),
                Transform(prob_note, row_sum),
                run_time=TM,
            )
            self.wait(TM2)

            # --- Phase D: mass transport bridge ---
            flow_arrows: List[Arrow] = []
            for i in range(2):
                for j in range(len(dots_b)):
                    w_ij: float = float(weights[i, j])
                    flow_arrows.append(
                        Arrow(
                            start=dots_a[i].get_center(),
                            end=dots_b[j].get_center(),
                            color=C_LOSS,
                            stroke_width=1.2 + 3.0 * w_ij,
                            stroke_opacity=0.15 + 0.80 * w_ij,
                            buff=0.13,
                            max_tip_length_to_length_ratio=0.12,
                        )
                    )

            flow_note = Text("Preference mass is transported across domains", font_size=20, color=C_LOSS, font="Sans")
            flow_note.move_to(np.array([0.0, -2.0, 0.0]))

            self.play(
                FadeOut(VGroup(*outgoing_labels), run_time=TF),
                Transform(prob_note, flow_note),
                LaggedStart(*[Create(a) for a in flow_arrows], lag_ratio=0.05),
                run_time=TM,
            )
            self.wait(TS)

            # --- Phase E: optimization view ---
            rec_loss = MathTex(r"\mathcal{L}_{rec}", font_size=38, color=C_A)
            rec_loss.move_to(np.array([-1.8, 2.2, 0.0]))
            sink_loss = MathTex(r"\mathcal{L}_{S}", font_size=38, color=C_B)
            sink_loss.move_to(np.array([1.8, 2.2, 0.0]))
            total_loss = MathTex(
                r"\mathcal{L} = \mathcal{L}_{rec} + \lambda\mathcal{L}_{S}",
                font_size=40,
                color=C_MATCH,
            )
            total_loss.move_to(np.array([0.0, 1.35, 0.0]))

            self.play(
                Write(rec_loss),
                Write(sink_loss),
                run_time=TW,
            )
            self.play(
                Write(total_loss),
                run_time=TM,
            )
            self.wait(TM2)

            self.play(
                FadeOut(VGroup(*soft_lines), run_time=TF),
                FadeOut(VGroup(*flow_arrows), run_time=TF),
                FadeOut(prob_note, run_time=TF),
                run_time=TF,
            )

            label = (
                "Hard matching is discrete and brittle. "
                "Soft matching is continuous, differentiable, and aligns distributions."
            )
            insight = make_insight_box(label)
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
                (-4.0, 1.4, 0.0), (-4.0, 0.25, 0.0), (-4.0, -0.85, 0.0),
            ]
            source_radii: List[float] = [0.25, 0.40, 0.18]  # mass = size

            # --- Target distribution (right) ---
            target_pos: List[Tuple[float, float, float]] = [
                (4.0, 1.95, 0.0), (4.0, 0.8, 0.0),
                (4.0, -0.25, 0.0), (4.0, -1.2, 0.0),
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

            src_label = Text("Source Distribution", font_size=20, color=C_A)
            src_label.move_to(np.array([-4.0, -2.08, 0.0]))
            tgt_label = Text("Target Distribution", font_size=20, color=C_B)
            tgt_label.move_to(np.array([4.0, -2.1, 0.0]))

            size_note = Text("Size = Mass Probability", font_size=20, color=C_ACC)
            size_note.move_to(np.array([0.0, -2.1, 0.0]))

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
                Arrow(np.array(source_pos[0] + + RIGHT * 0.01), np.array(target_pos[0]),
                      color=C_FLOW, stroke_width=3, buff=0.21),
                Arrow(np.array(source_pos[0]), np.array(target_pos[1] + LEFT * 0.1),
                      color=C_FLOW, stroke_width=1.5, buff=0.2, stroke_opacity=0.5),
                Arrow(np.array(source_pos[1] + RIGHT * 0.2), np.array(target_pos[2]) + RIGHT * 0.05,
                      color=C_FLOW, stroke_width=3, buff=0.2),
                Arrow(np.array(source_pos[2] + LEFT * 0.05), np.array(target_pos[3]),
                      color=C_FLOW, stroke_width=2, buff=0.25),
            ]

            flow_label = Text("Transport plan (Opacity = contribution)", font_size=20, color=C_FLOW)
            flow_label.move_to(np.array([0.0, 2.65, 0.0]))

            self.play(Write(flow_label), run_time=TW)
            self.play(
                LaggedStart(*[Create(a) for a in flow_arrows], lag_ratio=0.2),
                run_time=TM,
            )
            self.wait(TM2)

            label = "Sinkhorn: Differentiable OT for end-to-end training in deep learning"
            insight = make_insight_box(label)
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
            rec_loss = MathTex(r"\mathcal{L}_{\text{rec}}", color=C_A, font_size=36).scale(1.5)
            rec_loss.move_to(np.array([-3.4, 1.5, 0.0]))

            rec_desc = MathTex(r"\text{Recommendation}", font_size=36, color=C_A)
            rec_desc.next_to(rec_loss, DOWN, buff=0.3)

            # --- Component 2: Sinkhorn loss ---
            sink_loss = MathTex(r"\lambda \times \mathcal{L}_{\text{sink}}", color=C_B, font_size=36).scale(1.5)
            sink_loss.move_to(np.array([3.4, 1.5, 0.0]))

            sink_desc = MathTex(r"\text{Distribution alignment (Sinkhorn)}", font_size=36, color=C_B)
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

            # --- Final equation ---
            final_eq = MathTex(
                r"\mathcal{L} = \mathcal{L}_{\text{rec}} + \lambda \times \mathcal{L}_{\text{sink}}",
                color=C_MATCH,
                font_size=36
            ).scale(1.5)
            final_eq.move_to(np.array([0.0, -0.8, 0.0]))

            final_desc = MathTex(r"\text{Combined loss function}", font_size=36, color=C_MATCH)
            final_desc.next_to(final_eq, DOWN, buff=0.3)

            plus_sign = MathTex("+", color=C_MATCH).scale(1.5)
            plus_sign.move_to(np.array([0.0, 1.5, 0.0]))

            self.play(Write(plus_sign), run_time=TW)
            self.wait(TS)
            self.play(
                FadeIn(final_eq), FadeIn(final_desc),
                run_time=TM,
            )
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
            insights: List[Tuple[str, str, str]] = [
                ("No identity needed", C_A, "assets/target.png" ),
                ("Preference is enough", C_MATCH, "assets/target.png" ),
                ("Soft matching wins", C_B, "assets/accept.png" ),
            ]

            cards: List[Group] = []

            for (text, color, icon_path) in insights:
                t = Text(text, font_size=32, color=color)

                icon = ImageMobject(icon_path).scale(0.15)

                # Group icon and text, arrange horizontally
                row = Group(icon, t).arrange(RIGHT, buff=0.3)

                # Create a colored box behind the row
                box = Rectangle(
                    width=6.5,
                    height=1.2,
                    color=color,
                ).set_fill(opacity=0.15)

                row.align_to(box, LEFT)
                row.shift(RIGHT * 0.4)  # Left padding 

                group = Group(box, row)
                cards.append(group)

            # Arrange cards
            cards = Group(*cards).arrange(DOWN, aligned_edge=LEFT, buff=0.5)
            cards.to_edge(LEFT, buff=3.8)

            for card in cards:
                self.play(FadeIn(card, shift=RIGHT * 0.3), run_time=TW)
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
                x_range=[-3.0, 2.9, 1.0],
                y_range=[0.0, 8.9, 1.8],
                x_length=7,
                y_length=5,
                axis_config={"color": C_TEXT, "stroke_width": 1.5},
                tips=False,
            )
            
            axes.x_axis.add_tip(tip_length=0.15)
            axes.y_axis.add_tip(tip_length=0.15)

            axes.shift(DOWN * 0.5)

            x_label = Text("Parameter", font_size=20, color=C_TEXT)
            x_label.next_to(axes.x_axis, RIGHT, buff=0.2)
            y_label = Text("Loss", font_size=20, color=C_TEXT)
            y_label.next_to(axes.y_axis, UP, buff=0.2)

            # --- Loss curve: simple parabola ---
            curve = axes.plot(
                lambda x: x ** 2,
                x_range=[-3.0, 3.0],
                color=C_A,
                stroke_width=3,
            )
            curve_label = MathTex("L(\\theta) = \\theta^2", font_size=36, color=C_A)
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
            conv_label = Text("Converged at minimum", font_size=20, color=C_MATCH)
            conv_label.next_to(point, DOWN, buff=0.3)
            self.play(Write(conv_label), run_time=TW)
            self.wait(TM2)

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
                x_range=[-3.0, 2.9, 1.0],
                y_range=[0.0, 9.9, 2.0],
                x_length=5.0,
                y_length=3.5,
                axis_config={"color": C_TEXT, "stroke_width": 1},
                tips=False,
            )

            axes_l.x_axis.add_tip(tip_length=0.15)
            axes_l.y_axis.add_tip(tip_length=0.15)

            axes_l.shift(LEFT * 3.2 + UP * 0.2)

            # HNO3: rough, non-convex with local minima
            rough_curve = axes_l.plot(
                lambda x: (x ** 2) + 1.5 * np.sin(4.0 * x) + 0.6 * np.cos(7.0 * x) + 3.0,
                x_range=[-3.0, 3.0],
                color=C_FLOW,
                stroke_width=2.5,
            )
            hno3_label = Text("HNO3 - Hard Matching", font_size=20, color=C_FLOW)
            hno3_label.next_to(axes_l, DOWN, buff=0.2)

            # --- Right axes: SNO3 smooth landscape ---
            axes_r = Axes(
                x_range=[-3.0, 2.9, 1.0],
                y_range=[0.0, 9.9, 2.0],
                x_length=5.0,
                y_length=3.5,
                axis_config={"color": C_TEXT, "stroke_width": 1},
                tips=False,
            )
            axes_r.x_axis.add_tip(tip_length=0.15)
            axes_r.y_axis.add_tip(tip_length=0.15)

            axes_r.shift(RIGHT * 3.2 + UP * 0.2)

            # SNO3: smooth, convex - easy to optimize
            smooth_curve = axes_r.plot(
                lambda x: (x ** 2) + 0.5,
                x_range=[-3.0, 3.0],
                color=C_B,
                stroke_width=2.5,
            )
            sno3_label = Text("SNO3 - Soft Matching", font_size=20, color=C_B)
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
            point_l = Dot(axes_l.c2p(-2.5, (-2.5) ** 2 + 1.5 * np.sin(-10.0) + 0.6 * np.cos(-17.5) + 3.0),
                          radius=0.13, color=C_LOSS)
            point_r = Dot(axes_r.c2p(-2.5, (-2.5) ** 2 + 0.5),
                          radius=0.13, color=C_MATCH)

            self.play(GrowFromCenter(point_l), GrowFromCenter(point_r), run_time=TF)

            # Rough path: zigzag descent
            rough_x_steps: List[float] = [-2.5, -1.8, -2.1, -1.3, -1.6, -0.8, -0.4, 0.0]

            # Smooth path: direct descent
            smooth_x_steps: List[float] = [-2.5, -1.8, -1.2, -0.6, -0.2, 0.0]
            
            max_len = max(len(rough_x_steps), len(smooth_x_steps))

            # Animate simultaneously
            for i in range(max_len):
                anims = []

                if i < len(rough_x_steps):
                    x_l = rough_x_steps[i]
                    y_l = x_l ** 2 + 1.5 * np.sin(4.0 * x_l) + 0.6 * np.cos(7.0 * x_l) + 3.0
                    anims.append(point_l.animate.move_to(axes_l.c2p(x_l, y_l)))

                if i < len(smooth_x_steps):
                    x_r = smooth_x_steps[i]
                    y_r = x_r ** 2 + 0.5
                    anims.append(point_r.animate.move_to(axes_r.c2p(x_r, y_r)))

                self.play(*anims, run_time=0.25)

            self.wait(TM2)

            insight = make_insight_box("SNO3 outperforms HNO3: Smoother optimization landscape")
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

            title = make_title("3D Embedding Space Visualization")

            # --- Camera orientation ---
            self.set_camera_orientation(phi=70 * DEGREES, theta=-45 * DEGREES)
            self.begin_ambient_camera_rotation(rate=0.08)

            # Title
            self.add_fixed_in_frame_mobjects(title)
            self.add(title)

            self.play(FadeIn(title), run_time=TF)

            # --- 3D Axes ---
            axes = ThreeDAxes(
                x_range=[-4.0, 3.9, 1.0],
                y_range=[-4.0, 3.9, 1.0],
                z_range=[-3.0, 2.9, 1.0],
                x_length=6,
                y_length=6,
                z_length=4,
                axis_config={"color": C_TEXT, "stroke_width": 1},
                tips=False,
            )
            axes.x_axis.add_tip(tip_length=0.15)
            axes.y_axis.add_tip(tip_length=0.15)
            axes.z_axis.add_tip(tip_length=0.15)

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

            # --- Phase 1: Hard alignment (forced collapse) ---
            hard_a_positions: List[np.ndarray] = [
                np.array([
                    float(rng_a.uniform(-1, 1)),
                    float(rng_a.uniform(-1, 1)),
                    float(rng_a.uniform(-1, 1)),
                ])
                for _ in range(8)
            ]

            hard_b_positions: List[np.ndarray] = [
                np.array([
                    float(rng_a.uniform(-1, 1)),
                    float(rng_a.uniform(-1, 1)),
                    float(rng_a.uniform(-1, 1)),
                ])
                for _ in range(8)
            ]

            hard_text = Text("HNO3: Forced alignment", font_size=20, color=C_FLOW)

            box = SurroundingRectangle(
                hard_text,
                color=C_FLOW,
                buff=0.2,
                corner_radius=0.1,
                fill_opacity=0.15
            )

            hard_label = VGroup(box, hard_text)
            hard_label.next_to(ORIGIN, DOWN).to_edge(DOWN, buff=0.8)

            self.add_fixed_in_frame_mobjects(hard_label)

            self.play(
                FadeIn(hard_label, run_time=TF),
                *[
                    da.animate.move_to(hard_a)
                    for da, hard_a in zip(dots_a, hard_a_positions)
                ],
                *[
                    db.animate.move_to(hard_b)
                    for db, hard_b in zip(dots_b, hard_b_positions)
                ],
                run_time=TM,
            )

            self.wait(TL)
            self.play(FadeOut(hard_label))
            self.wait(TS)

            # --- Phase 2: Soft alignment - move clusters closer together ---
            aligned_a_positions: List[np.ndarray] = [
                np.array([
                    float(rng_a.uniform(-0.5, 0)),
                    float(rng_a.uniform(0.5, 1)),
                    float(rng_a.uniform(-0.5, 0.5)),
                ])
                for _ in range(8)
            ]

            aligned_b_positions: List[np.ndarray] = [
                np.array([
                    float(rng_b.uniform(0.5, 1)),
                    float(rng_b.uniform(-1, -0.5)),
                    float(rng_b.uniform(-0.5, 0.5)),
                ])
                for _ in range(8)
            ]

            soft_text = Text("SNO3: Smooth alignment", font_size=20, color=C_MATCH)

            box = SurroundingRectangle(
                soft_text,
                color=C_MATCH,
                buff=0.2,
                corner_radius=0.1,
                fill_opacity=0.15
            )

            soft_label = VGroup(box, soft_text)
            soft_label.next_to(ORIGIN, DOWN).to_edge(DOWN, buff=0.8)

            self.add_fixed_in_frame_mobjects(soft_label)

            self.play(
                FadeIn(soft_label, run_time=TF),
                *[
                    da.animate.move_to(aligned_a)
                    for da, aligned_a in zip(dots_a, aligned_a_positions)
                ],
                *[
                    db.animate.move_to(aligned_b)
                    for db, aligned_b in zip(dots_b, aligned_b_positions)
                ],
                run_time=TM,
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
            grid_initial = self._build_heatmap(initial_matrix, np.array([0.0, 0.0, 0.0]))

            iter_label = Text("Random initialization", font_size=22, color=C_MATCH)
            iter_label.move_to(np.array([0.0, -2.0, 0.0]))

            legend = MathTex(r"arr[i, j] = \text{transport weight from user } i \text{ to user } j", font_size=36, color=C_LOSS)
            legend.next_to(grid_initial, UP, buff=0.3)

            self.play(FadeIn(grid_initial), Write(iter_label), FadeIn(legend), run_time=TM)
            self.wait(TM2)

            # --- Phase 2: After row normalization ---
            row_norm: np.ndarray = initial_matrix / initial_matrix.sum(axis=1, keepdims=True)
            grid_row = self._build_heatmap(row_norm, np.array([0.0, 0.0, 0.0]))

            new_label = Text("After row normalization", font_size=22, color=C_MATCH)
            new_label.move_to(np.array([0.0, -2.0, 0.0]))

            self.play(
                ReplacementTransform(grid_initial, grid_row),
                ReplacementTransform(iter_label, new_label),
                run_time=TM,
            )
            self.wait(TM2)

            # --- Phase 3: After column normalization ---
            col_norm: np.ndarray = row_norm / row_norm.sum(axis=0, keepdims=True)
            grid_col = self._build_heatmap(col_norm, np.array([0.0, 0.0, 0.0]))

            col_label = Text("After column normalization", font_size=22, color=C_MATCH)
            col_label.move_to(np.array([0.0, -2.0, 0.0]))

            self.play(
                ReplacementTransform(grid_row, grid_col),
                ReplacementTransform(new_label, col_label),
                run_time=TM,
            )
            self.wait(TM2)

            # --- Phase 4: Converged (simulate several more iterations) ---
            converged: np.ndarray = col_norm.copy()
            for _ in range(8):
                converged = converged / converged.sum(axis=1, keepdims=True)
                converged = converged / converged.sum(axis=0, keepdims=True)

            grid_conv = self._build_heatmap(converged, np.array([0.0, 0.0, 0.0]))

            conv_label = Text("Converged: Stable transport plan", font_size=22, color=C_MATCH)
            conv_label.move_to(np.array([0.0, -2.0, 0.0]))

            self.play(
                ReplacementTransform(grid_col, grid_conv),
                ReplacementTransform(col_label, conv_label),
                run_time=TM,
            )
            self.wait(TM2)

            label = "Sinkhorn: Iterative Optimal Transport"
            insight = make_insight_box(label)

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
        cell_size: float = 0.6
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
                          float(rng_a.uniform(0, 2)), 0.0])
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
            field_label = Text("Gradient field pushes embeddings toward alignment", font_size=20, color=C_FLOW)
            field_label.move_to(np.array([0.0, -1.8, 0.0]))

            self.play(
                LaggedStart(*[Create(a) for a in field_arrows], lag_ratio=0.02),
                Write(field_label),
                run_time=TM,
            )
            self.wait(TS)

            # --- Phase 3: Embeddings move along the gradient field ---
            target_a: List[np.ndarray] = [
                np.array([p[0] * 0.35 - 0.5, p[1] * 0.35 + 0.8, 0.0])
                for p in pos_a
            ]
            target_b: List[np.ndarray] = [
                np.array([p[0] * 0.35, p[1] * 0.35, 0.0])
                for p in pos_b
            ]

            self.play(
                *[da.animate.move_to(ta) for da, ta in zip(dots_a, target_a)],
                *[db.animate.move_to(tb) for db, tb in zip(dots_b, target_b)],
                run_time=TM * 1.5,
            )
            self.wait(TM2)

            insight = make_insight_box(
                "Gradients sculpt the geometry, not just the weights"
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
        for gx in np.arange(-3, 4.5, 1.5):
            for gy in np.arange(-0.8, 3.0, 1.0):
                # Direction: arrows point right (toward alignment)
                dx: float = 0.35 if gx < 0 else -0.35
                dy: float = -gy * 0.1  # slight vertical pull toward center
                arr = Arrow(
                    start=np.array([gx, gy, 0.0]),
                    end=np.array([gx + dx, gy + dy, 0.0]),
                    color=C_FLOW,
                    stroke_width=3,
                    stroke_opacity=0.6,
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

            title = make_title("KL Divergence vs Sinkhorn Distance")
            subtitle = Text("Handling non-overlapping distributions", font_size=24, color=C_ACC, font="Sans")
            subtitle.next_to(title[0], DOWN, buff=0.15)
            title_group = VGroup(title, subtitle)
            self.play(FadeIn(title_group), run_time=TF)

            # --- Shared setup: two side-by-side panels ---
            axes_l = Axes(
                x_range=[-6.0, 6.0, 2.0],
                y_range=[0.0, 0.59, 0.1],
                x_length=5.5,
                y_length=3.5,
                axis_config={"color": C_TEXT, "stroke_width": 1},
                tips=False,
            )
            axes_l.x_axis.add_tip(tip_length=0.15)
            axes_l.y_axis.add_tip(tip_length=0.15)

            axes_l.shift(LEFT * 3.3 + DOWN * 0.5)

            kl_header = Text("KL Divergence (Failed)", font_size=22, color=C_FLOW, weight="BOLD", font="Sans")
            kl_header.next_to(axes_l, UP, buff=0.5)

            axes_r = Axes(
                x_range=[-6.0, 6.0, 2.0],
                y_range=[0.0, 0.59, 0.1],
                x_length=5.5,
                y_length=3.5,
                axis_config={"color": C_TEXT, "stroke_width": 1},
                tips=False,
            )
            axes_r.x_axis.add_tip(tip_length=0.15)
            axes_r.y_axis.add_tip(tip_length=0.15)

            axes_r.shift(RIGHT * 3.3 + DOWN * 0.5)

            sink_header = Text("Sinkhorn Distance (Works)", font_size=22, color=C_B, weight="BOLD", font="Sans")
            sink_header.next_to(axes_r, UP, buff=0.55)

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

            # --- Phase A1: KL overlap baseline ---
            source_kl = axes_l.plot(
                lambda x: (1.0 / (1.1 * math.sqrt(2 * PI))) * math.exp(-0.5 * ((x + 1.0) / 1.1) ** 2),
                x_range=[-6.0, 6.0],
                color=C_A, stroke_width=2.5,
            )
            target_kl = axes_l.plot(
                lambda x: (1.0 / (1.1 * math.sqrt(2 * PI))) * math.exp(-0.5 * ((x - 1.0) / 1.1) ** 2),
                x_range=[-6.0, 6.0],
                color=C_B, stroke_width=2.5,
            )

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

            self.play(
                Create(source_kl), Create(target_kl),
                Create(source_sink), Create(target_sink),
                run_time=TM,
            )
            overlap_note = Text("Overlap: KL is well-defined", font_size=18, color=C_MATCH, font="Sans")
            overlap_note.next_to(axes_l, DOWN, buff=0.15)
            self.play(FadeIn(overlap_note), run_time=TF)
            self.wait(TS)

            # --- Phase A2/A3: KL non-overlap and instability ---
            source_kl_no = axes_l.plot(
                lambda x: (1.0 / (0.8 * math.sqrt(2 * PI))) * math.exp(-0.5 * ((x + 3.5) / 0.8) ** 2),
                x_range=[-6.0, 0.0],
                color=C_A,
                stroke_width=2.5,
            )
            target_kl_no = axes_l.plot(
                lambda x: (1.0 / (0.8 * math.sqrt(2 * PI))) * math.exp(-0.5 * ((x - 3.5) / 0.8) ** 2),
                x_range=[0.0, 6.0],
                color=C_B,
                stroke_width=2.5,
            )

            self.play(
                Transform(source_kl, source_kl_no),
                Transform(target_kl, target_kl_no),
                FadeOut(overlap_note),
                run_time=TM,
            )

            zero_region = Rectangle(
                width=2.1,
                height=1.3,
                color=C_FLOW,
                stroke_width=2.0,
                fill_opacity=0.18,
            )
            zero_region.move_to(axes_l.c2p(-3.4, 0.25))
            log_zero = MathTex(r"\log(0) \Rightarrow \infty", font_size=34, color=C_FLOW)
            log_zero.next_to(axes_l, DOWN, buff=0.15)

            kl_counter = Integer(0, font_size=30, color=C_FLOW)
            kl_counter.next_to(log_zero, DOWN, buff=0.12)
            counter_label = Text("KL value", font_size=17, color=C_FLOW, font="Sans")
            counter_label.next_to(kl_counter, LEFT, buff=0.12)

            self.play(
                FadeIn(zero_region),
                Write(log_zero),
                FadeIn(counter_label),
                FadeIn(kl_counter),
                run_time=TM,
            )
            self.play(ChangingDecimal(kl_counter, lambda t: t * 100000), run_time=TM)

            kl_infinity = MathTex(r"KL(P\|Q) = \infty", font_size=36, color=C_FLOW)
            kl_infinity.next_to(axes_l, DOWN, buff=0.2)
            self.play(
                FadeOut(VGroup(log_zero, kl_counter, counter_label), run_time=TF),
                Write(kl_infinity),
                run_time=TF,
            )

            kl_insight = VGroup(
                Text("Density matching", font_size=18, color=C_FLOW, font="Sans"),
                Text("Needs support overlap", font_size=18, color=C_FLOW, font="Sans"),
                Text("Diverges on zero-density regions", font_size=18, color=C_FLOW, font="Sans"),
            ).arrange(DOWN, aligned_edge=LEFT, buff=0.16)
            kl_insight.move_to(np.array([-3.3, -0.1, 0.0]))

            self.play(FadeIn(kl_insight, shift=LEFT * 0.2), run_time=TM)
            self.wait(TM2)

            # --- Phase B: Sinkhorn transport mechanism on same non-overlap setup ---
            transport_arrows: List[Arrow] = []
            for sx in [-4.0, -3.3, -2.6]:
                for tx in [2.8, 3.5, 4.2]:
                    dist = abs(tx - sx)
                    transport_arrows.append(
                        Arrow(
                            start=axes_r.c2p(sx, 0.02),
                            end=axes_r.c2p(tx, 0.02),
                            color=C_LOSS,
                            stroke_width=max(1.2, 4.6 - 0.45 * dist),
                            stroke_opacity=max(0.30, 1.0 - 0.06 * dist),
                            buff=0.0,
                            tip_length=0.18,
                        )
                    )

            smooth_arrows: List[Arrow] = []
            for arr in transport_arrows:
                smooth_arrows.append(
                    Arrow(
                        start=arr.get_start(),
                        end=arr.get_end(),
                        color=C_MATCH,
                        stroke_width=arr.get_stroke_width() + 1.2,
                        stroke_opacity=0.12,
                        buff=0.0,
                        tip_length=0.16,
                    )
                )

            sink_value = MathTex(r"W_\varepsilon(P, Q) < \infty", font_size=36, color=C_B)
            sink_value.next_to(axes_r, DOWN, buff=0.2)
            sink_entropy = Text("Entropic smoothing -> stable gradients", font_size=18, color=C_MATCH, font="Sans")
            sink_entropy.next_to(sink_value, DOWN, buff=0.1)

            self.play(
                LaggedStart(*[Create(a) for a in transport_arrows], lag_ratio=0.08),
                Write(sink_value),
                run_time=TM,
            )
            self.play(
                LaggedStart(*[FadeIn(a) for a in smooth_arrows], lag_ratio=0.04),
                FadeIn(sink_entropy),
                run_time=TM,
            )

            sink_insight = VGroup(
                Text("No overlap required", font_size=18, color=C_B, font="Sans"),
                Text("Uses geometry + transport cost", font_size=18, color=C_B, font="Sans"),
                Text("Smooth and differentiable", font_size=18, color=C_B, font="Sans"),
            ).arrange(DOWN, aligned_edge=LEFT, buff=0.16)
            sink_insight.move_to(np.array([3.3, -0.1, 0.0]))
            self.play(FadeIn(sink_insight, shift=RIGHT * 0.2), run_time=TM)
            self.wait(TS)

            # --- Phase C: side-by-side comparison table ---
            self.play(
                FadeOut(axes_l), FadeOut(source_kl), FadeOut(target_kl), FadeOut(kl_infinity),
                FadeOut(axes_r), FadeOut(source_sink), FadeOut(target_sink), FadeOut(sink_value),
                FadeOut(divider),
                FadeOut(zero_region),
                FadeOut(VGroup(*transport_arrows)),
                FadeOut(VGroup(*smooth_arrows)),
                FadeOut(kl_header), FadeOut(sink_header),
                FadeOut(kl_insight), FadeOut(sink_insight), FadeOut(sink_entropy),
                run_time=TM
            )

            table_title = Text("Deep Comparison", font_size=28, color=C_TEXT, weight="BOLD", font="Sans")
            table_title.move_to(np.array([0.0, 2.0, 0.0]))

            rows = [
                ("Requirement", "Overlap needed", "No overlap needed"),
                ("Signal", "Density", "Geometry + cost"),
                ("Behavior", "Diverges", "Stable"),
                ("Matching", "Pointwise", "Global transport"),
                ("Optimization", "Unstable", "Differentiable"),
            ]

            table_items: List[VGroup] = []
            y0 = 1.2
            for idx, (aspect, kl_v, sink_v) in enumerate(rows):
                y = y0 - idx * 0.55
                aspect_t = Text(aspect, font_size=19, color=C_TEXT, font="Sans")
                aspect_t.move_to(np.array([-2.6, y, 0.0]))
                kl_t = Text(kl_v, font_size=19, color=C_FLOW, font="Sans")
                kl_t.move_to(np.array([0.2, y, 0.0]))
                sink_t = Text(sink_v, font_size=19, color=C_B, font="Sans")
                sink_t.move_to(np.array([3.3, y, 0.0]))
                table_items.append(VGroup(aspect_t, kl_t, sink_t))

            self.play(
                FadeIn(table_title),
                LaggedStart(*[FadeIn(row) for row in table_items], lag_ratio=0.14),
                run_time=TM
            )

            final_insight = Tex(
                r"KL = Compare probabilities \\\ Sinkhorn = Move probability mass",
                font_size=44,
                color=C_MATCH,
            )
            final_insight.move_to(np.array([0.0, -2.0, 0.0]))
            self.play(Write(final_insight), run_time=TM)
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
                box = SurroundingRectangle(t, color=color, buff=0.2, corner_radius=0.1)
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
            for i, (wg_mob, color) in enumerate(wgan_items):
                arr = Arrow(
                    start=np.array([-1.5, wgan_mobjects[i].get_center()[1], 0.0]),
                    end=np.array([1.5, wgan_mobjects[i].get_center()[1], 0.0]),
                    color=color,
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
                box = SurroundingRectangle(t, color=color, buff=0.2, corner_radius=0.1)
                group = VGroup(box, t)
                group.move_to(np.array([3.5, 1.2 - i * 0.95, 0.0]))
                cdr_mobjects.append(group)

            self.play(
                LaggedStart(*[FadeIn(m) for m in cdr_mobjects], lag_ratio=0.2),
                run_time=TM,
            )
            self.wait(TM2)

            # --- Concluding note ---
            conclusion = "SNO3 induces implicit Wasserstein alignment, leading to stable training"
            insight = make_insight_box(conclusion)
            self.play(FadeIn(insight), run_time=TW)
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