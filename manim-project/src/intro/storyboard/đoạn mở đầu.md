# I. STORYBOARD TỔNG QUAN
### 1. Visual-first, text-last

* Không giải thích bằng chữ → giải thích bằng chuyển động
* Text chỉ dùng để “anchor concept”

### 2. Mapping trừu tượng → hình học

| Concept           | Visual metaphor    |
| ----------------- | ------------------ |
| User              | Dot                |
| Sở thích          | Vector / position  |
| Domain            | Cluster / island   |
| Transfer learning | Motion / alignment |

### 3. Motion = semantics

* Matching → line connect
* Similarity → distance
* Learning → smooth transformation

---

# II. STORYBOARD CHI TIẾT

## SCENE 1 — “Hai thế giới tách biệt”

### Mục tiêu

Thiết lập problem space

### Visual

* 2 cluster (dots) nằm trái/phải
* Màu khác nhau (blue vs green)
* Label nhỏ phía trên

### Animation

1. Fade in từng cluster
2. Zoom nhẹ để thấy separation

### Narration intent

> “Hai hệ thống hoàn toàn tách biệt…”

### Key insight visualized

* Không overlap → domain gap

---

## SCENE 2 — NO3 constraint (căng thẳng chính)

### Visual

* Vẽ 3 dấu X giữa hai cluster:

  * No shared users
  * No shared items
  * No side info

### Animation

* Mỗi constraint xuất hiện sequentially
* Khi xuất hiện → các cluster “giật nhẹ ra xa hơn”

### Narration intent

> “Không có bất kỳ điểm chung nào để bám vào”

### 3Blue1Brown trick

* Distance giữa cluster tăng lên = tăng độ khó bài toán

---

## SCENE 3 — Embedding intuition (cực quan trọng)

### Mục tiêu

Đưa người xem vào không gian latent

### Visual

* Transform từ “island” → “embedding space”
* Dots phân bố thành hình dạng (cluster structure)

### Animation

```text
Island → dissolve → point cloud
```

### Narration intent

> “Mỗi người thực chất là một điểm trong không gian sở thích”

### Insight

* Chuẩn bị cho cả Hungarian và Sinkhorn

---

## SCENE 4 — HNO3 (Hungarian matching)

### Visual

* Hai cluster giữ nguyên shape
* Vẽ line nối từng cặp điểm

### Animation (quan trọng)

1. Highlight 1 điểm bên trái
2. Scan bên phải (glow effect)
3. Snap → connect line
4. Lặp lại cho tất cả

### Motion semantics

* Snap = optimal assignment
* 1-1 mapping

### Narration intent

> “Tìm ‘bản sao gần nhất’ ở thế giới bên kia”

### Enhancement (3B1B-style)

* Hiển thị “cost” bằng độ dài line
* Line ngắn hơn → tốt hơn

---

## SCENE 5 — Limitation của HNO3

### Visual

* Một vài line bị kéo dài bất thường

### Animation

* Stretch line → rung nhẹ (instability)

### Narration intent

> “Nhưng ép buộc 1-1 đôi khi không tự nhiên”

### Insight

* Over-constrained matching

---

## SCENE 6 — SNO3 (Sinkhorn / distribution alignment)

### Visual

* Remove all lines
* Convert cluster → “density cloud” (blur nhẹ)

### Animation

1. Hai cloud tiến lại gần nhau
2. Shape dần align
3. Không có line, chỉ có overlap

### Narration intent

> “Không ghép từng người… mà ghép cả phân bố”

### Motion semantics

* Smooth transport = optimal transport

---

## SCENE 7 — So sánh trực quan

### Visual split screen

| Left | Right |
| ---- | ----- |
| HNO3 | SNO3  |

### Animation

* HNO3: line connections
* SNO3: cloud alignment

### Overlay metric:

* Accuracy (rating) → highlight HNO3
* Ranking → highlight SNO3

### Narration intent

> “Hai cách nhìn — hai loại sức mạnh”

---

## SCENE 8 — Real-world grounding (Amazon)

### Visual

* Icon:

  * CD
  * Digital music

* Map vào 2 cluster

### Animation

* Apply lại HNO3 & SNO3 trên đó

### Narration intent

> “Trên dữ liệu thật, kết quả tách biệt rõ ràng”

---

## SCENE 9 — Big insight (climax)

### Visual

* Hai cluster dần overlap một phần
* Xuất hiện “bridge” mờ giữa chúng

### Animation

* Bridge sáng dần

### Narration intent

> “Bạn không cần biết họ là ai… chỉ cần biết họ giống ai”

---

## SCENE 10 — Outro (signature 3B1B)

### Visual

* Zoom out → toàn bộ embedding space

### Text nhỏ:

* “Matching vs Distribution”

---

# III. NHỊP ĐIỆU (TIMING)

| Scene | Duration |
| ----- | -------- |
| 1–2   | 6–8s     |
| 3     | 6s       |
| 4     | 10s      |
| 5     | 5s       |
| 6     | 10s      |
| 7     | 8s       |
| 8     | 6s       |
| 9–10  | 6s       |

Total ~60s video

---

# IV. PATTERN TỪ BIG TECH

### Google (YouTube RecSys)

* Visualization nội bộ:

  * embedding cloud
  * cluster drift
* Không dùng matching → gần SNO3

### Amazon

* Cross-domain recommendation:

  * matching-based models (early)
  * transport-based models (modern)

### Meta

* Ads ranking:

  * user embedding alignment
  * không cần identity match

---

# V. CHECKLIST TRIỂN KHAI MANIM

### Bắt buộc có:

* `Dot`, `VGroup`, `Line`
* `Transform`, `FadeIn`, `LaggedStart`
* `ValueTracker` (nếu animate smooth transport)

### Optional nâng cấp:

* Gaussian blur (fake density)
* Interpolation morph
