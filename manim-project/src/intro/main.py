from manim import *
import numpy as np
from typing import List, Tuple
import random

random.seed(42)
np.random.seed(42)


class CrossDomainRecommendation(Scene):
    def construct(self):
        self.camera.background_color = "#ffffff"
        self.scenes = [
            self.scene_1_isolated_domains,
            self.scene_2_no3_constraints,
            self.scene_3_embedding_intuition,
            self.scene_4_hno3_matching,
            self.scene_5_hno3_limitation,
            self.scene_6_sno3_distribution,
            self.scene_7_comparison,
            self.scene_8_real_world,
            self.scene_9_final_insight,
        ]
        
        for scene_func in self.scenes:
            scene_func()
            self.wait(0.5)

    def scene_1_isolated_domains(self):
        """Two isolated domains with distinct clusters"""
        domain_a = self.create_cluster(
            n=12, 
            center=np.array([-4, 0, 0]), 
            color="#1f77b4", 
            radius=0.6
        )
        domain_b = self.create_cluster(
            n=12, 
            center=np.array([4, 0, 0]), 
            color="#2ca02c", 
            radius=0.6
        )
        
        label_a = Text("Physical Books", font_size=24, color="#1f77b4").next_to(domain_a, DOWN, buff=0.8)
        label_b = Text("Kindle", font_size=24, color="#2ca02c").next_to(domain_b, DOWN, buff=0.8)
        
        group = VGroup(domain_a, domain_b, label_a, label_b)
        
        self.play(FadeIn(domain_a), FadeIn(domain_b), run_time=1.5)
        self.play(FadeIn(label_a), FadeIn(label_b), run_time=1)
        self.wait(2)
        
        self.add_to_scene(group)

    def scene_2_no3_constraints(self):
        """Visualize NO3 constraints sequentially"""
        self.clear()
        
        domain_a = self.create_cluster(
            n=12, 
            center=np.array([-4, 0, 0]), 
            color="#1f77b4", 
            radius=0.6
        )
        domain_b = self.create_cluster(
            n=12, 
            center=np.array([4, 0, 0]), 
            color="#2ca02c", 
            radius=0.6
        )
        
        self.play(FadeIn(domain_a), FadeIn(domain_b), run_time=0.5)
        
        # Constraint 1: No shared users
        x1 = Text("No shared users", font_size=20, color="#d62728").move_to(np.array([0, 2.5, 0]))
        self.play(Write(x1), run_time=1)
        self.wait(1)
        
        # Deepen separation slightly
        new_a = self.create_cluster(n=12, center=np.array([-5, 0, 0]), color="#1f77b4", radius=0.6)
        new_b = self.create_cluster(n=12, center=np.array([5, 0, 0]), color="#2ca02c", radius=0.6)
        self.play(Transform(domain_a, new_a), Transform(domain_b, new_b), run_time=1)
        self.wait(0.5)
        
        # Constraint 2: No shared items
        x2 = Text("No shared items", font_size=20, color="#d62728").next_to(x1, DOWN, buff=0.3)
        self.play(Write(x2), run_time=1)
        self.wait(1)
        
        new_a2 = self.create_cluster(n=12, center=np.array([-6, 0, 0]), color="#1f77b4", radius=0.6)
        new_b2 = self.create_cluster(n=12, center=np.array([6, 0, 0]), color="#2ca02c", radius=0.6)
        self.play(Transform(new_a, new_a2), Transform(new_b, new_b2), run_time=1)
        self.wait(0.5)
        
        # Constraint 3: No side information
        x3 = Text("No auxiliary data", font_size=20, color="#d62728").next_to(x2, DOWN, buff=0.3)
        self.play(Write(x3), run_time=1)
        self.wait(1)
        
        self.play(FadeOut(x1), FadeOut(x2), FadeOut(x3), run_time=0.5)
        self.wait(1)

    def scene_3_embedding_intuition(self):
        """Transform islands to embedding space point clouds"""
        self.clear()
        
        # Create initial island-like clusters
        domain_a = self.create_cluster(
            n=15, 
            center=np.array([-3.5, 0, 0]), 
            color="#1f77b4", 
            radius=0.5
        )
        domain_b = self.create_cluster(
            n=15, 
            center=np.array([3.5, 0, 0]), 
            color="#2ca02c", 
            radius=0.5
        )
        
        self.play(FadeIn(domain_a), FadeIn(domain_b), run_time=0.8)
        
        # Transition to embedding space (slightly dispersed)
        domain_a_embedded = self.create_cluster(
            n=15, 
            center=np.array([-3.5, 0, 0]), 
            color="#1f77b4", 
            radius=0.8
        )
        domain_b_embedded = self.create_cluster(
            n=15, 
            center=np.array([3.5, 0, 0]), 
            color="#2ca02c", 
            radius=0.8
        )
        
        self.play(
            Transform(domain_a, domain_a_embedded),
            Transform(domain_b, domain_b_embedded),
            run_time=2
        )
        
        title = Text("Preference Embedding Space", font_size=26, color="#000000", font="sans-serif").to_edge(UP, buff=0.3)
        self.play(Write(title), run_time=1)
        self.wait(2)
        
        self.play(FadeOut(title), run_time=0.5)

    def scene_4_hno3_matching(self):
        """HNO3: Hungarian 1-to-1 matching"""
        self.clear()
        
        n_users = 8
        domain_a_pts = self.create_cluster(
            n=n_users, 
            center=np.array([-3.5, 0, 0]), 
            color="#1f77b4", 
            radius=0.7,
            return_points=True
        )
        domain_b_pts = self.create_cluster(
            n=n_users, 
            center=np.array([3.5, 0, 0]), 
            color="#2ca02c", 
            radius=0.7,
            return_points=True
        )
        
        domain_a = self.create_cluster(
            n=n_users, 
            center=np.array([-3.5, 0, 0]), 
            color="#1f77b4", 
            radius=0.7
        )
        domain_b = self.create_cluster(
            n=n_users, 
            center=np.array([3.5, 0, 0]), 
            color="#2ca02c", 
            radius=0.7
        )
        
        self.play(FadeIn(domain_a), FadeIn(domain_b), run_time=0.8)
        
        label = Text("Optimal 1-to-1 Matching (Hungarian)", font_size=22, color="#000000").to_edge(UP, buff=0.3)
        self.play(Write(label), run_time=0.8)
        
        # Generate matching with shortest distances
        matching = self.compute_greedy_matching(domain_a_pts, domain_b_pts)
        
        lines = VGroup()
        for i, j in matching:
            line = Line(
                domain_a_pts[i], 
                domain_b_pts[j], 
                stroke_width=2,
                stroke_color="#ff7f0e",
                stroke_opacity=0.6
            )
            lines.add(line)
        
        self.play(
            LaggedStart(
                *[GrowFromPoint(line, line.get_start()) for line in lines],
                lag_ratio=0.1,
                run_time=2
            )
        )
        
        self.wait(2)
        self.play(FadeOut(label), FadeOut(lines), run_time=0.5)

    def scene_5_hno3_limitation(self):
        """Show limitation of HNO3: suboptimal/stretched matches"""
        self.clear()
        
        n_users = 8
        domain_a_pts = self.create_cluster(
            n=n_users, 
            center=np.array([-3.5, 0, 0]), 
            color="#1f77b4", 
            radius=0.7,
            return_points=True
        )
        domain_b_pts = self.create_cluster(
            n=n_users, 
            center=np.array([3.5, 0, 0]), 
            color="#2ca02c", 
            radius=0.7,
            return_points=True
        )
        
        domain_a = self.create_cluster(
            n=n_users, 
            center=np.array([-3.5, 0, 0]), 
            color="#1f77b4", 
            radius=0.7
        )
        domain_b = self.create_cluster(
            n=n_users, 
            center=np.array([3.5, 0, 0]), 
            color="#2ca02c", 
            radius=0.7
        )
        
        self.play(FadeIn(domain_a), FadeIn(domain_b), run_time=0.5)
        
        # Compute matching
        matching = self.compute_greedy_matching(domain_a_pts, domain_b_pts)
        
        # Highlight longest lines (poor matches)
        distances = []
        for i, j in matching:
            dist = np.linalg.norm(domain_a_pts[i] - domain_b_pts[j])
            distances.append((dist, i, j))
        
        distances.sort(reverse=True)
        worst = distances[:3]  # Top 3 longest
        
        lines = []
        for dist, i, j in worst:
            line = Line(
                domain_a_pts[i], 
                domain_b_pts[j], 
                stroke_width=3,
                stroke_color="#d62728"
            )
            lines.append(line)
            self.play(GrowFromPoint(line, line.get_start()), run_time=0.4)
        
        # Animate instability (vibrate)
        for line in lines:
            self.play(
                Wiggle(line, scale_value=0.1),
                run_time=0.6
            )
        
        label = Text("Over-constrained matching", font_size=20, color="#d62728").to_edge(UP, buff=0.3)
        self.play(Write(label), run_time=0.8)
        
        self.wait(1.5)
        self.play(FadeOut(label), run_time=0.5)

    def scene_6_sno3_distribution(self):
        """SNO3: Sinkhorn distribution alignment"""
        self.clear()
        
        # Create initial clusters
        domain_a = self.create_cluster(
            n=20, 
            center=np.array([-4, 0, 0]), 
            color="#1f77b4", 
            radius=0.8
        )
        domain_b = self.create_cluster(
            n=20, 
            center=np.array([4, 0, 0]), 
            color="#2ca02c", 
            radius=0.8
        )
        
        self.play(FadeIn(domain_a), FadeIn(domain_b), run_time=0.8)
        
        label = Text("Optimal Transport Alignment", font_size=22, color="#000000").to_edge(UP, buff=0.3)
        self.play(Write(label), run_time=0.8)
        
        # Apply blur/glow to simulate distribution
        domain_a_glow = domain_a.copy()
        domain_b_glow = domain_b.copy()
        
        # Move clusters closer
        domain_a_moved = self.create_cluster(
            n=20, 
            center=np.array([-1.5, 0, 0]), 
            color="#1f77b4", 
            radius=0.8
        )
        domain_b_moved = self.create_cluster(
            n=20, 
            center=np.array([1.5, 0, 0]), 
            color="#2ca02c", 
            radius=0.8
        )
        
        self.play(
            Transform(domain_a, domain_a_moved),
            Transform(domain_b, domain_b_moved),
            run_time=2
        )
        
        self.wait(1.5)
        
        # Create overlap region
        overlap = Circle(radius=1.2, color="#8B4513", fill_opacity=0.15).move_to(np.array([0, 0, 0]))
        self.play(FadeIn(overlap), run_time=0.8)
        
        self.wait(1.5)
        self.play(FadeOut(label), FadeOut(overlap), run_time=0.5)

    def scene_7_comparison(self):
        """Split view: HNO3 vs SNO3 with different metrics highlighted"""
        self.clear()
        
        # Left side: HNO3 (emphasize accuracy/rating)
        domain_a_left = self.create_cluster(
            n=8, 
            center=np.array([-5, 1.5, 0]), 
            color="#1f77b4", 
            radius=0.5
        )
        domain_b_left = self.create_cluster(
            n=8, 
            center=np.array([-1, 1.5, 0]), 
            color="#2ca02c", 
            radius=0.5
        )
        
        # Right side: SNO3 (emphasize ranking)
        domain_a_right = self.create_cluster(
            n=8, 
            center=np.array([1, 1.5, 0]), 
            color="#1f77b4", 
            radius=0.5
        )
        domain_b_right = self.create_cluster(
            n=8, 
            center=np.array([5, 1.5, 0]), 
            color="#2ca02c", 
            radius=0.5
        )
        
        self.play(
            FadeIn(domain_a_left), FadeIn(domain_b_left),
            FadeIn(domain_a_right), FadeIn(domain_b_right),
            run_time=1
        )
        
        # HNO3 matching
        pts_a_left = self.cluster_to_points(domain_a_left)
        pts_b_left = self.cluster_to_points(domain_b_left)
        matching_left = self.compute_greedy_matching(pts_a_left, pts_b_left)
        
        lines_left = VGroup()
        for i, j in matching_left:
            line = Line(pts_a_left[i], pts_b_left[j], stroke_width=2, stroke_color="#ff7f0e", stroke_opacity=0.7)
            lines_left.add(line)
        
        self.play(LaggedStart(*[GrowFromPoint(l, l.get_start()) for l in lines_left], lag_ratio=0.05, run_time=1))
        
        # SNO3: cloud/distribution
        pts_a_right = self.cluster_to_points(domain_a_right)
        pts_b_right = self.cluster_to_points(domain_b_right)
        
        # Move closer to visualize alignment
        domain_a_right_moved = self.create_cluster(n=8, center=np.array([1.8, 1.5, 0]), color="#1f77b4", radius=0.5)
        domain_b_right_moved = self.create_cluster(n=8, center=np.array([4.2, 1.5, 0]), color="#2ca02c", radius=0.5)
        
        self.play(
            Transform(domain_a_right, domain_a_right_moved),
            Transform(domain_b_right, domain_b_right_moved),
            run_time=1
        )
        
        overlap = Circle(radius=0.9, color="#8B4513", fill_opacity=0.15).move_to(np.array([3, 1.5, 0]))
        self.play(FadeIn(overlap), run_time=0.6)
        
        # Labels
        label_hno3 = Text("HNO3", font_size=20, color="#000000").move_to(np.array([-3, 0.5, 0]))
        label_sno3 = Text("SNO3", font_size=20, color="#000000").move_to(np.array([3, 0.5, 0]))
        
        metric_hno3 = Text("Rating Accuracy", font_size=16, color="#1f77b4").next_to(label_hno3, DOWN, buff=0.2)
        metric_sno3 = Text("Ranking Quality", font_size=16, color="#2ca02c").next_to(label_sno3, DOWN, buff=0.2)
        
        self.play(Write(label_hno3), Write(label_sno3), run_time=0.6)
        self.play(Write(metric_hno3), Write(metric_sno3), run_time=0.8)
        
        self.wait(2)
        self.play(FadeOut(label_hno3), FadeOut(label_sno3), FadeOut(metric_hno3), FadeOut(metric_sno3), FadeOut(overlap), FadeOut(lines_left), run_time=0.5)

    def scene_8_real_world(self):
        """Real-world example: CD vs Digital Music"""
        self.clear()
        
        # CD cluster
        cd_cluster = self.create_cluster(
            n=12, 
            center=np.array([-3, 0, 0]), 
            color="#9467bd", 
            radius=0.6
        )
        
        # Digital music cluster
        digital_cluster = self.create_cluster(
            n=12, 
            center=np.array([3, 0, 0]), 
            color="#e377c2", 
            radius=0.6
        )
        
        self.play(FadeIn(cd_cluster), FadeIn(digital_cluster), run_time=1)
        
        # Icons/labels
        label_cd = Text("CD Buyers", font_size=20, color="#9467bd").next_to(cd_cluster, DOWN, buff=0.8)
        label_digital = Text("Digital Listeners", font_size=20, color="#e377c2").next_to(digital_cluster, DOWN, buff=0.8)
        
        self.play(Write(label_cd), Write(label_digital), run_time=1)
        
        # Apply HNO3
        pts_cd = self.cluster_to_points(cd_cluster)
        pts_digital = self.cluster_to_points(digital_cluster)
        matching = self.compute_greedy_matching(pts_cd, pts_digital)
        
        lines = VGroup()
        for i, j in matching[:6]:  # Show top 6 matches
            line = Line(pts_cd[i], pts_digital[j], stroke_width=1.5, stroke_color="#17becf", stroke_opacity=0.5)
            lines.add(line)
        
        self.play(LaggedStart(*[GrowFromPoint(l, l.get_start()) for l in lines], lag_ratio=0.1, run_time=1.5))
        
        self.wait(2)
        self.play(FadeOut(label_cd), FadeOut(label_digital), FadeOut(lines), run_time=0.5)

    def scene_9_final_insight(self):
        """Bridge between domains: similarity without identity"""
        self.clear()
        
        domain_a = self.create_cluster(
            n=15, 
            center=np.array([-2.5, 0, 0]), 
            color="#1f77b4", 
            radius=0.7
        )
        domain_b = self.create_cluster(
            n=15, 
            center=np.array([2.5, 0, 0]), 
            color="#2ca02c", 
            radius=0.7
        )
        
        self.play(FadeIn(domain_a), FadeIn(domain_b), run_time=0.8)
        
        # Create a bridge/overlap region
        bridge = VMobject()
        bridge.set_points_as_corners([
            np.array([-1, -0.3, 0]),
            np.array([-0.5, 0.3, 0]),
            np.array([0.5, 0.3, 0]),
            np.array([1, -0.3, 0]),
        ])
        bridge.set_stroke(color="#ff7f0e", width=4, opacity=0)
        bridge.set_fill(color="#ff7f0e", opacity=0.1)
        
        self.play(FadeIn(bridge), run_time=1)
        
        # Create text insight
        insight = Text(
            "Similarity without identity",
            font_size=24,
            color="#000000"
        ).to_edge(UP, buff=0.4)
        
        self.play(Write(insight), run_time=1)
        
        self.wait(2)
        self.play(FadeOut(insight), run_time=0.5)

    # ========== Helper Methods ==========

    def create_cluster(
        self, 
        n: int, 
        center: np.ndarray, 
        color: str, 
        radius: float,
        return_points: bool = False
    ):
        """Generate a cluster of dots with Gaussian distribution"""
        np.random.seed(hash(tuple(center)) % 2**32)
        points = np.random.randn(n, 3) * radius
        points[:, 2] = 0  # Keep in z=0 plane
        points += center
        
        dots = VGroup()
        for pt in points:
            dot = Dot(pt, radius=0.12, color=color, fill_opacity=0.9)
            dots.add(dot)
        
        if return_points:
            return points
        return dots

    def cluster_to_points(self, cluster_vgroup: VGroup) -> List[np.ndarray]:
        """Extract point positions from a VGroup of dots"""
        points = []
        for dot in cluster_vgroup:
            points.append(dot.get_center())
        return points

    def compute_greedy_matching(
        self, 
        pts_a: List[np.ndarray], 
        pts_b: List[np.ndarray]
    ) -> List[Tuple[int, int]]:
        """Greedy 1-to-1 matching based on minimum distance"""
        n = len(pts_a)
        distances = []
        for i, pt_a in enumerate(pts_a):
            for j, pt_b in enumerate(pts_b):
                dist = np.linalg.norm(pt_a - pt_b)
                distances.append((dist, i, j))
        
        distances.sort()
        
        used_a = set()
        used_b = set()
        matching = []
        
        for dist, i, j in distances:
            if i not in used_a and j not in used_b:
                matching.append((i, j))
                used_a.add(i)
                used_b.add(j)
                if len(matching) == n:
                    break
        
        return matching

    def add_to_scene(self, mobject):
        """Add mobject but keep it in scene context"""
        self.add(mobject)


if __name__ == "__main__":
    pass