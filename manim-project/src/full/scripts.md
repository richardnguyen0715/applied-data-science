## Kịch bản mẫu

# 1. OVERALL STRUCTURE (≈ 9–10 phút)

| Scene | Nội dung | Duration |
| --- | --- | --- |
| 1 | Problem: Data fragmentation | 1:00 |
| 2 | CDR landscape | 1:00 |
| 3 | NO3 setting | 1:00 |
| 4 | Learning objective | 1:00 |
| 5 | HNO3 | 2:00 |
| 6 | HNO3 limitation | 1:00 |
| 7 | SNO3 | 2:00 |
| 8 | Sinkhorn intuition | 1:30 |
| 9 | Final objective | 0:30 |
| 10 | Insight + case study | 0:30 |

---

# 2. SCENE-BY-SCENE SCRIPT

---

# SCENE 1 — PROBLEM: DATA FRAGMENTATION

## Visual

- 2 circle lớn:
    - Trái: “Music Platform”
    - Phải: “Movie Platform”
- Mỗi circle có 5–6 dots (users)
- Không có connection giữa 2 bên

---

## Animation Flow

1. Fade in title
2. Vẽ 2 platform
3. Spawn users
4. Zoom nhẹ để thấy rõ separation

---

## Voice-over

“Trong thực tế, người dùng tồn tại trên nhiều nền tảng khác nhau.”

“Mỗi hệ thống chỉ quan sát được một phần rất nhỏ hành vi của họ.”

“Điều này dẫn đến một vấn đề lớn: dữ liệu bị phân mảnh.”

---

## Insight

- Đây chính là nguyên nhân của:
    - data sparsity
    - cold-start

---

# SCENE 2 — CDR LANDSCAPE

## Visual

Hiển thị 3 case:

- Case 1: user overlap (có đường nối)
- Case 2: item overlap
- Case 3: no overlap (không có gì)

---

## Animation

- Highlight từng case
- Cuối cùng zoom vào case 3

---

## Voice-over

“Cross-Domain Recommendation cố gắng tận dụng dữ liệu từ nhiều domain.”

“Nhưng hầu hết các phương pháp đều giả định rằng có một dạng overlap.”

“Còn nếu không có overlap thì sao?”

---

## Insight

- Paper này chọn **hardest setting**

---

# SCENE 3 — NO3 SETTING

## Visual

Text lớn:

- No user overlap
- No item overlap
- No side information

Sau đó hiện lại 2 domain hoàn toàn tách biệt

---

## Animation

- Bullet xuất hiện từng dòng
- Fade thành 2 cloud embedding

---

## Voice-over

“Paper này nghiên cứu một setting cực kỳ khó.”

“Không có user chung.”

“Không có item chung.”

“Và cũng không có bất kỳ thông tin bổ sung nào.”

---

## Insight

- Đây là scenario gần với real-world privacy constraints

---

# SCENE 4 — LEARNING OBJECTIVE

## Visual

- Hai domain D1, D2
- Loss mỗi domain:
    - L1
    - L2
- Sau đó merge thành:
    
    L = L1 + L2
    

---

## Animation

- Show từng loss riêng
- Sau đó combine

---

## Voice-over

“Ý tưởng cơ bản là học đồng thời hai domain.”

“Chúng ta tối ưu hóa một objective chung, bao gồm loss của cả hai phía.”

---

## Insight

- Đây là **multi-task learning**

---

# SCENE 5 — HNO3 (HARD MATCHING)

## Visual

- Hai nhóm user embedding (dots)
- Vẽ đường nối 1-1

---

## Animation Flow

1. Hiện embedding
2. Vẽ line từng cặp
3. Merge 2 domain thành 1

---

## Voice-over

“Cách đầu tiên là hard matching.”

“Ta cố gắng ghép từng user bên này với một user bên kia.”

“Sử dụng Hungarian Algorithm để tìm matching tối ưu.”

---

## Insight

- biến no-overlap → overlap

---

# SCENE 6 — LIMITATION

## Visual

- Matching sai → line bị lệch
- Một số line bị “đứt” hoặc sai direction

---

## Voice-over

“Nhưng cách này có nhiều hạn chế.”

“Nó phụ thuộc mạnh vào chất lượng embedding ban đầu.”

“Và không thể học end-to-end.”

---

## Insight

- discrete optimization → không robust

---

# SCENE 7 — SNO3 (SOFT MATCHING)

## Visual

- Hai cloud user
- Fully connected graph (nhiều line mờ)

---

## Animation

- Từng dot bên trái connect tới nhiều dot bên phải
- Opacity thấp → thể hiện soft

---

## Voice-over

“Thay vì ép match 1-1, ta chuyển sang soft matching.”

“Mỗi user có thể liên quan đến nhiều user khác với mức độ khác nhau.”

---

## Insight

- chuyển từ matching → distribution alignment

---

# SCENE 8 — SINKHORN (CORE)

## Visual cực kỳ quan trọng

### Phase 1:

- 2 distribution (dots với size khác nhau)

### Phase 2:

- Arrows thể hiện flow

### Phase 3:

- Dots “dịch chuyển” gần nhau hơn

---

## Animation Flow

1. Show mass (size dot)
2. Draw arrows (flow)
3. Animate movement

---

## Voice-over

“Bài toán này được mô hình hóa bằng Optimal Transport.”

“Ta tìm cách di chuyển ‘khối lượng xác suất’ từ distribution này sang distribution kia với chi phí thấp nhất.”

“Sinkhorn algorithm cho phép giải bài toán này một cách hiệu quả và differentiable.”

---

## Insight

- continuous optimization
- differentiable → train end-to-end

---

# SCENE 9 — FINAL OBJECTIVE

## Visual

Loss cuối:

- Recommendation loss
- 
    - Sinkhorn loss

---

## Animation

- Hai phần merge lại

---

## Voice-over

“Mô hình cuối cùng tối ưu hai thành phần.”

“Một là recommendation accuracy.”

“Hai là sự đồng bộ giữa hai không gian sở thích.”

---

## Insight

- representation learning + alignment

---

# SCENE 10 — KEY INSIGHT

## Visual

3 dòng:

- No identity needed
- Preference is enough
- Soft matching wins

---

## Voice-over

“Điểm quan trọng nhất của paper này là:”

“Chúng ta không cần biết user là ai.”

“Chỉ cần hiểu họ thích gì.”

---

# 3. THIẾT KẾ QUAN TRỌNG CHO MANIM

## Camera

- Zoom khi chuyển từ macro → micro (domain → user)
- Pan khi so sánh 2 domain

---

## Color coding

- Domain A: Blue
- Domain B: Green
- Matching: Yellow
- Flow: Red

---

## Motion semantics

- Line = relationship
- Arrow = direction / transport
- Opacity = probability

---

# TỔNG QUAN CÁC SCENE BỔ SUNG

| Scene | Mục tiêu | Vị trí đề xuất |
| --- | --- | --- |
| 11 | Gradient Descent | sau Scene 4 (learning) |
| 12 | Loss Landscape | sau Scene 11 |
| 13 | 3D Embedding Space | trước HNO3 |
| 14 | Sinkhorn Convergence | sau Scene 8 |

---

# SCENE 11 — GRADIENT DESCENT VISUALIZATION

## Mục tiêu

Giải thích:

- model đang học như thế nào
- embedding được cập nhật ra sao

---

## Visual Design

### Thành phần:

- 1 đường cong loss (2D)
- 1 điểm (model state)
- Vector gradient (arrow)

---

## Animation Flow

### Phase 1: Khởi tạo

- Vẽ curve loss (hình parabolic hoặc non-convex)
- Đặt điểm ở vị trí cao

---

### Phase 2: Gradient step

- Vẽ vector gradient (mũi tên xuống)
- Điểm di chuyển theo từng step

---

### Phase 3: Convergence

- Điểm dao động nhẹ rồi ổn định tại minimum

---

## Voice-over

“Mô hình được huấn luyện bằng gradient descent.”

“Tại mỗi bước, chúng ta tính gradient của loss và cập nhật tham số theo hướng giảm dần.”

“Quá trình này lặp lại cho đến khi hội tụ.”

---

## Insight

- Optimization engine của toàn bộ paper
- HNO3 vs SNO3 khác nhau chủ yếu ở **loss surface**

---

# SCENE 12 — LOSS LANDSCAPE

## Mục tiêu

So sánh:

- Hard matching vs Soft matching
    
    → landscape khác nhau như thế nào
    

---

## Visual Design

### Split screen:

Bên trái: HNO3

Bên phải: SNO3

---

### HNO3

- Landscape gồ ghề (non-smooth)
- Có nhiều local minima
- Step nhảy “giật”

---

### SNO3

- Landscape smooth hơn
- Gradient liên tục

---

## Animation Flow

1. Hiện 2 surface
2. Drop 1 điểm từ trên xuống
3. Quan sát path

---

## Voice-over

“Hard matching tạo ra một hàm loss rời rạc và không trơn.”

“Điều này khiến quá trình tối ưu khó khăn và dễ mắc kẹt ở local minima.”

“Ngược lại, soft matching dựa trên optimal transport tạo ra một hàm loss liên tục và dễ tối ưu hơn.”

---

## Insight

- Đây là lý do cốt lõi khiến SNO3 outperform

---

# SCENE 13 — 3D EMBEDDING SPACE

## Mục tiêu

Trực quan hóa:

- user embedding space
- alignment giữa 2 domain

---

## Visual Design

### 3D coordinate system

- X, Y, Z axes

---

### Hai cluster:

- Domain A (blue cluster)
- Domain B (green cluster)

---

## Animation Flow

### Phase 1: Initial state

- 2 cluster tách biệt

---

### Phase 2: HNO3

- Một số điểm bị kéo “ép” lại gần nhau

---

### Phase 3: SNO3

- Toàn bộ cluster dần align
- distribution overlap tốt hơn

---

## Voice-over

“Mỗi user được biểu diễn dưới dạng một vector trong không gian embedding.”

“Ban đầu, hai domain có phân phối hoàn toàn khác nhau.”

“Hard matching cố gắng ép từng điểm lại với nhau.”

“Trong khi soft matching điều chỉnh toàn bộ phân phối.”

---

## Insight

- Representation learning thực chất là **geometry problem**

---

# SCENE 14 — SINKHORN CONVERGENCE

## Mục tiêu

Giải thích:

- Sinkhorn không chỉ là concept
- mà là **iterative algorithm**

---

## Visual Design

### Matrix heatmap (transport plan)

- trục X: source users
- trục Y: target users

---

## Animation Flow

### Phase 1: Initialization

- matrix random

---

### Phase 2: Iteration

- normalize rows
- normalize columns

(lặp lại)

---

### Phase 3: Convergence

- matrix trở nên ổn định
- pattern rõ ràng

---

### Đồng thời:

- hiển thị arrows giữa 2 domain
- arrows thay đổi theo matrix

---

## Voice-over

“Sinkhorn algorithm hoạt động bằng cách lặp lại hai bước chuẩn hóa.”

“Chuẩn hóa theo hàng, sau đó theo cột.”

“Quá trình này dần dần hội tụ đến một ma trận vận chuyển tối ưu.”

---

## Insight

- biến bài toán OT → differentiable
- cực kỳ quan trọng trong deep learning hiện đại

---

# 5. KẾT NỐI CÁC SCENE

## Flow hoàn chỉnh

1. Problem
2. CDR
3. NO3
4. Objective
5. Gradient descent (NEW)
6. Loss landscape (NEW)
7. 3D embedding (NEW)
8. HNO3
9. Limitation
10. SNO3
11. Sinkhorn intuition
12. Sinkhorn convergence (NEW)
13. Final objective
14. Insight

---

# 6. DESIGN PRINCIPLES (QUAN TRỌNG)

## 1. Consistency

- Dot luôn = user
- Line = relation
- Arrow = flow

---

## 2. Motion = Meaning

- Smooth → continuous optimization
- Jump → discrete matching

---

## 3. Geometry-first thinking

- Embedding = space
- Matching = mapping
- Sinkhorn = mass transport

---

| Scene | Nội dung | Placement |
| --- | --- | --- |
| 15 | Gradient Vector Field | sau 3D embedding |
| 16 | KL vs Sinkhorn | sau Sinkhorn intro |
| 17 | Wasserstein GAN comparison | cuối phần method |

---

# SCENE 15 — GRADIENT VECTOR FIELD TRONG EMBEDDING SPACE

## Mục tiêu

Hiển thị:

- gradient không chỉ là một vector đơn lẻ
- mà là **field trên toàn bộ embedding space**
- và cách nó “đẩy” user embedding

---

## Visual Design

### 3D Embedding Space

- Axes: X, Y, Z
- Cloud A (blue), Cloud B (green)

---

### Gradient Vector Field

- Grid points trong không gian
- Mỗi point có 1 arrow nhỏ (vector)

---

## Animation Flow

### Phase 1 — Static embedding

- Hai cluster tách biệt rõ

Voice-over:

“Không gian embedding ban đầu có hai phân phối hoàn toàn khác nhau.”

---

### Phase 2 — Hiện vector field

- Xuất hiện các arrow nhỏ khắp không gian
- Arrow hướng từ cluster A → B

Voice-over:

“Gradient của loss tạo thành một vector field trong không gian này.”

“Mỗi điểm trong không gian đều có một hướng cập nhật riêng.”

---

### Phase 3 — Apply gradient flow

- Các điểm (user embeddings) bắt đầu di chuyển theo vector field
- Path cong, không thẳng

Voice-over:

“Khi training, mỗi embedding di chuyển theo vector field này.”

“Quá trình này chính là gradient descent trong không gian biểu diễn.”

---

### Phase 4 — Convergence

- Hai cluster tiến lại gần
- overlap tăng

---

## Insight (quan trọng)

- Gradient không chỉ update parameters
    
    → mà **reshape geometry của embedding space**
    

---

# SCENE 16 — KL DIVERGENCE vs SINKHORN DISTANCE

## Mục tiêu

So sánh:

- KL divergence (distribution matching truyền thống)
- Sinkhorn (optimal transport)

---

## Visual Design

### Hai distribution (1D hoặc 2D)

- Source (blue)
- Target (green)

---

## Phase A — KL Divergence

### Animation

- Nếu distributions không overlap:
    - highlight vùng zero density
- KL → ∞

Voice-over:

“KL divergence yêu cầu hai phân phối phải chồng lấp.”

“Nếu không, giá trị sẽ trở nên vô hạn.”

---

### Visual trick

- highlight vùng không overlap bằng màu đỏ
- hiển thị text “undefined / infinite”

---

## Phase B — Sinkhorn

### Animation

- Vẽ arrows từ source → target
- mass “di chuyển”

Voice-over:

“Ngược lại, Sinkhorn distance đo chi phí để biến đổi một phân phối thành phân phối khác.”

“Nó không yêu cầu overlap ban đầu.”

---

## Phase C — Side-by-side

| KL | Sinkhorn |
| --- | --- |
| Diverges | Stable |
| Local | Global transport |
| Density-based | Geometry-based |

---

## Insight

- KL = so sánh density
- Sinkhorn = so sánh **geometry + transport cost**

---

# SCENE 17 — SO SÁNH VỚI WASSERSTEIN GAN

## Mục tiêu

Kết nối paper với:

- Wasserstein distance
- GAN training

---

## Visual Design

### GAN setup

- Generator distribution (blue)
- Real distribution (green)

---

## Phase 1 — KL / JS (GAN truyền thống)

### Animation

- distributions không overlap
- gradient gần như 0

Voice-over:

“Trong GAN truyền thống, khi hai phân phối không overlap…”

“gradient gần như biến mất.”

---

## Phase 2 — Wasserstein distance

### Animation

- vẽ arrows transport
- gradient tồn tại mọi nơi

Voice-over:

“Wasserstein distance cung cấp gradient hữu ích ngay cả khi hai phân phối cách xa nhau.”

---

## Phase 3 — Mapping sang paper

### Visual

- thay GAN bằng:
    - Domain A
    - Domain B

Voice-over:

“Điều này tương tự với bài toán trong paper.”

“Sinkhorn distance là một dạng regularized Wasserstein distance.”

---

## Phase 4 — Insight mapping

| Concept | Paper |
| --- | --- |
| Generator | Domain A |
| Real data | Domain B |
| Wasserstein | Sinkhorn |

---

## Insight sâu

- SNO3 = implicit Wasserstein alignment
- giúp training stable hơn

---

# 4. KẾT NỐI 3 SCENE NÀY (RẤT QUAN TRỌNG)

## Narrative flow

1. Gradient field
    
    → hiểu cách model học
    
2. KL vs Sinkhorn
    
    → hiểu tại sao loss này tốt hơn
    
3. Wasserstein GAN
    
    → hiểu nó nằm trong hệ sinh thái ML lớn hơn
    

---

# 5. DESIGN NGUYÊN TẮC

## 1. Geometry-first

- Không nói “loss” abstract
    
    → luôn biểu diễn bằng space + movement
    

---

## 2. Motion encoding meaning

- KL → không di chuyển được
- Sinkhorn → flow rõ ràng

---

## 3. Local vs Global

- Gradient field → local force
- Sinkhorn → global alignment