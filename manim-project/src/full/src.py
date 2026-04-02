from manim import *
import numpy as np

class CDRPresentation(Scene):
    def construct(self):
        # Thiết lập màu sắc để đảm bảo tính nhất quán
        self.color_a = BLUE
        self.color_b = GREEN
        self.color_match = YELLOW
        self.color_flow = RED

        self.scene_1_problem()
        self.scene_3_no3_setting()
        self.scene_5_hno3()
        self.scene_7_sno3()
        self.scene_8_sinkhorn()
        
        # Có thể thêm các scene khác theo kịch bản ở đây
        self.wait(2)

    def scene_1_problem(self):
        title = Text("Vấn đề: Dữ liệu bị phân mảnh", font_size=40)
        self.play(FadeIn(title, shift=UP))
        self.play(title.animate.to_edge(UP))

        platform_a = Circle(radius=2, color=self.color_a, fill_opacity=0.1).shift(LEFT * 3)
        platform_b = Circle(radius=2, color=self.color_b, fill_opacity=0.1).shift(RIGHT * 3)

        label_a = Text("Music Platform", font_size=30, color=self.color_a).next_to(platform_a, UP)
        label_b = Text("Movie Platform", font_size=30, color=self.color_b).next_to(platform_b, UP)

        self.play(Create(platform_a), Create(platform_b))
        self.play(Write(label_a), Write(label_b))

        # Users in Platform A
        dots_a = VGroup(*[Dot(point=platform_a.get_center() + np.array([np.random.uniform(-1, 1), np.random.uniform(-1, 1), 0]), color=self.color_a) for _ in range(6)])
        # Users in Platform B
        dots_b = VGroup(*[Dot(point=platform_b.get_center() + np.array([np.random.uniform(-1, 1), np.random.uniform(-1, 1), 0]), color=self.color_b) for _ in range(6)])

        self.play(FadeIn(dots_a, shift=UP*0.5), FadeIn(dots_b, shift=UP*0.5))
        self.wait(1)
        
        # Zoom nhẹ (bằng cách scale)
        group = VGroup(platform_a, platform_b, label_a, label_b, dots_a, dots_b)
        self.play(group.animate.scale(1.1))
        self.wait(2)

        self.play(FadeOut(group), FadeOut(title))

    def scene_3_no3_setting(self):
        setting_text = VGroup(
            Text("No user overlap", font_size=36),
            Text("No item overlap", font_size=36),
            Text("No side information", font_size=36)
        ).arrange(DOWN, aligned_edge=LEFT).center()

        for text in setting_text:
            self.play(FadeIn(text, shift=RIGHT))
            self.wait(0.5)

        self.play(setting_text.animate.to_edge(UP).scale(0.7))

        # Hiển thị lại 2 domain tách biệt dưới dạng cloud (các dải dot)
        cloud_a = VGroup(*[Dot(point=np.array([np.random.normal(-3, 0.8), np.random.normal(0, 1), 0]), color=self.color_a) for _ in range(20)])
        cloud_b = VGroup(*[Dot(point=np.array([np.random.normal(3, 0.8), np.random.normal(0, 1), 0]), color=self.color_b) for _ in range(20)])

        self.play(LaggedStart(FadeIn(cloud_a), FadeIn(cloud_b), lag_ratio=0.1))
        self.wait(2)
        
        self.play(FadeOut(setting_text), FadeOut(cloud_a), FadeOut(cloud_b))

    def scene_5_hno3(self):
        title = Text("Hard Matching (HNO3)", font_size=40).to_edge(UP)
        self.play(FadeIn(title))

        cloud_a = VGroup(*[Dot(point=np.array([-3, y, 0]), color=self.color_a) for y in np.linspace(-2, 2, 5)])
        cloud_b = VGroup(*[Dot(point=np.array([3, y, 0]), color=self.color_b) for y in np.linspace(-2, 2, 5)])

        self.play(FadeIn(cloud_a), FadeIn(cloud_b))
        self.wait(1)

        lines = VGroup()
        for i in range(5):
            line = Line(cloud_a[i].get_center(), cloud_b[4-i].get_center(), color=self.color_match)
            lines.add(line)

        self.play(Create(lines))
        self.wait(1)

        # Merge
        self.play(
            cloud_a.animate.shift(RIGHT * 2),
            cloud_b.animate.shift(LEFT * 2),
            lines.animate.scale(0.33)
        )
        self.wait(2)
        
        self.play(FadeOut(VGroup(title, cloud_a, cloud_b, lines)))

    def scene_7_sno3(self):
        title = Text("Soft Matching (SNO3)", font_size=40).to_edge(UP)
        self.play(FadeIn(title))

        cloud_a = VGroup(*[Dot(point=np.array([np.random.normal(-3, 0.5), np.random.normal(0, 1.5), 0]), color=self.color_a) for _ in range(10)])
        cloud_b = VGroup(*[Dot(point=np.array([np.random.normal(3, 0.5), np.random.normal(0, 1.5), 0]), color=self.color_b) for _ in range(10)])

        self.play(FadeIn(cloud_a), FadeIn(cloud_b))
        self.wait(1)

        lines = VGroup()
        for dot_a in cloud_a:
            for dot_b in cloud_b:
                if np.random.rand() > 0.5: # Connect random with low opacity
                    line = Line(dot_a.get_center(), dot_b.get_center(), color=self.color_flow, stroke_opacity=0.2)
                    lines.add(line)

        self.play(Create(lines, run_time=2))
        self.wait(2)
        
        self.play(FadeOut(VGroup(title, cloud_a, cloud_b, lines)))

    def scene_8_sinkhorn(self):
        title = Text("Sinkhorn Algorithm (Optimal Transport)", font_size=40).to_edge(UP)
        self.play(FadeIn(title))

        # Size represents mass
        dot_a1 = Dot(point=LEFT*3 + UP*1, color=self.color_a, radius=0.2)
        dot_a2 = Dot(point=LEFT*3 + DOWN*1, color=self.color_a, radius=0.1)
        
        dot_b1 = Dot(point=RIGHT*3 + UP*1.5, color=self.color_b, radius=0.15)
        dot_b2 = Dot(point=RIGHT*3, color=self.color_b, radius=0.1)
        dot_b3 = Dot(point=RIGHT*3 + DOWN*1.5, color=self.color_b, radius=0.05)

        self.play(FadeIn(VGroup(dot_a1, dot_a2, dot_b1, dot_b2, dot_b3)))

        # Flow arrows
        arrow1 = Arrow(dot_a1.get_center(), dot_b1.get_center(), color=self.color_flow, stroke_width=4, buff=0.2)
        arrow2 = Arrow(dot_a1.get_center(), dot_b2.get_center(), color=self.color_flow, stroke_width=2, buff=0.2)
        arrow3 = Arrow(dot_a2.get_center(), dot_b3.get_center(), color=self.color_flow, stroke_width=3, buff=0.2)
        
        self.play(GrowArrow(arrow1), GrowArrow(arrow2), GrowArrow(arrow3))
        self.wait(1)

        # Movement / Flow animation
        self.play(
            dot_a1.animate.shift(RIGHT),
            dot_a2.animate.shift(RIGHT),
            dot_b1.animate.shift(LEFT),
            dot_b2.animate.shift(LEFT),
            dot_b3.animate.shift(LEFT),
        )
        self.wait(2)
        
        self.play(FadeOut(VGroup(title, dot_a1, dot_a2, dot_b1, dot_b2, dot_b3, arrow1, arrow2, arrow3)))
