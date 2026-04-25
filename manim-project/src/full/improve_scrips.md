Dưới đây là phiên bản mở rộng (≈ gấp đôi độ dài + thêm chiều sâu kỹ thuật + nhịp kể chuyện rõ hơn) cho **SCENE 7 (SNO3 soft matching)** và **SCENE 16 (KL vs Sinkhorn)**. Nội dung được thiết kế theo hướng cinematic + giải thích đúng bản chất trong paper bạn gửi.

---

# SCENE 7 — SNO3 (SOFT MATCHING, EXPANDED)

## Visual — Phase A: Problem setup (recap hard matching failure)

* Hai cụm user (Domain A bên trái, Domain B bên phải)
* Ban đầu:

  * Vẽ vài đường **1–1 hard matching (đậm, rõ)**
  * Một số match nhìn “sai” (cross awkward lines)
* Fade-in text nhỏ:

  * “Hard constraint: 1 user ↔ 1 user”

### Animation

* Các đường nối bị “giật”, lock cứng
* Một vài node rung nhẹ → thể hiện mismatch

### Voice-over

“Trong cách tiếp cận cũ, mỗi user bị ép phải ghép với đúng một user ở domain còn lại.”

“Điều này giả định rằng tồn tại một ánh xạ hoàn hảo… nhưng thực tế thì hiếm khi đúng.”

“User có thể có nhiều khía cạnh sở thích — không thể nén vào một match duy nhất.”

---

## Visual — Phase B: Transition to soft matching

* Hard edges dần mờ đi
* Graph chuyển sang **fully connected bipartite**
* Tất cả nodes bên trái connect tới tất cả nodes bên phải
* Edge opacity khác nhau

### Animation

* Hard edges “tan ra” thành nhiều soft edges
* Xuất hiện heatmap-like intensity trên edges

### Voice-over

“Thay vì ép buộc một ánh xạ rời rạc, ta nới lỏng bài toán.”

“Ta cho phép mỗi user liên kết với nhiều user khác — với các mức độ khác nhau.”

---

## Visual — Phase C: Probability interpretation

* Mỗi node bên trái:

  * Các edge outgoing có tổng weight = 1
* Hiển thị số nhỏ trên edges: 0.1, 0.3, 0.6...

### Animation

* Normalize weights → sum = 1
* Một node “phân phối” sang nhiều node

### Voice-over

“Giờ đây, mỗi user không còn là một điểm cố định.”

“Mà là một phân phối — trải đều ảnh hưởng sang nhiều user khác.”

“Đây không còn là matching… mà là alignment giữa hai phân phối.”

---

## Visual — Phase D: Mass transport intuition (bridge to Sinkhorn)

* Convert node → “mass point”
* Edge → flow
* Dòng chảy từ trái sang phải

### Animation

* Mass (particles) chảy dọc theo edges
* Flow mạnh ở edge đậm, yếu ở edge mờ

### Voice-over

“Ta có thể hiểu quá trình này như việc ‘di chuyển khối lượng’.”

“Mỗi user mang theo một phần ‘preference mass’, và phân phối nó sang domain bên kia.”

---

## Visual — Phase E: Optimization view (link to loss function)

* Overlay:

  * Recommendation loss (ℓ)
  * Transport loss (ℓ_S)

### Animation

* Hai lực:

  * Một kéo theo prediction accuracy
  * Một kéo theo alignment

### Voice-over

“Quá trình học giờ đây trở thành một bài toán cân bằng.”

“Vừa tối ưu recommendation… vừa tối thiểu hóa chi phí vận chuyển giữa hai tập user.”

---

## Insight (hiển thị cuối scene)

* Hard matching → discrete, brittle
* Soft matching → continuous, differentiable
* Matching → Alignment of distributions

---

# SCENE 16 — KL DIVERGENCE vs SINKHORN DISTANCE (EXPANDED)

---

## Visual — Setup (shared)

* 2 distributions:

  * Source (blue)
  * Target (green)
* Có thể dùng:

  * 1D Gaussian hoặc 2D blobs

---

# PHASE A — KL Divergence (failure mode deep dive)

## Visual A1 — Overlap case (baseline)

* Hai Gaussian overlap một phần

### Voice-over

“Khi hai phân phối có vùng chồng lấp, KL divergence hoạt động bình thường.”

“Nó đo sự khác biệt về mật độ xác suất.”

---

## Visual A2 — Non-overlap case

* Dịch distribution xanh sang phải → không overlap

### Animation

* Highlight vùng:

  * Source có density
  * Target = 0

* Vùng đó chuyển màu đỏ

* Text:

  * “log(0)”
  * “∞”

### Voice-over

“Nhưng khi không có overlap…”

“KL divergence yêu cầu so sánh mật độ tại những nơi target bằng 0.”

“Điều này khiến log-probability trở nên không xác định — và giá trị tiến tới vô hạn.”

---

## Visual A3 — Instability

* KL value counter tăng nhanh → ∞
* Screen rung nhẹ

### Voice-over

“Kết quả là loss không còn ổn định — và không thể tối ưu.”

---

## Insight A

* KL = density matching
* Requires support overlap
* Sensitive to zero-density regions

---

# PHASE B — Sinkhorn Distance (mechanism)

## Visual B1 — Same non-overlap setup

* Giữ nguyên distributions không overlap

### Voice-over

“Sinkhorn distance tiếp cận vấn đề theo cách hoàn toàn khác.”

---

## Visual B2 — Transport plan

* Vẽ arrows từ source → target

### Animation

* Mass di chuyển
* Có nhiều path, không chỉ 1–1

### Voice-over

“Thay vì so sánh mật độ tại từng điểm…”

“Nó hỏi: cần tốn bao nhiêu chi phí để biến phân phối này thành phân phối kia?”

---

## Visual B3 — Cost encoding

* Distance càng xa → arrow càng dài / màu đỏ hơn

### Voice-over

“Chi phí vận chuyển phụ thuộc vào khoảng cách hình học giữa các điểm.”

---

## Visual B4 — Entropic smoothing (Sinkhorn specific)

* Arrows không quá sắc nét → hơi “blur”

### Voice-over

“Sinkhorn thêm một regularization entropy…”

“Giúp bài toán trở nên mượt và khả vi — phù hợp với gradient descent.”

---

## Insight B

* No need for overlap
* Uses geometry
* Produces smooth transport plan

---

# PHASE C — Side-by-side deep comparison

## Visual Table (animated)

| Aspect       | KL             | Sinkhorn          |
| ------------ | -------------- | ----------------- |
| Requirement  | Overlap needed | No overlap needed |
| Signal type  | Density        | Geometry + cost   |
| Behavior     | Diverges       | Stable            |
| Matching     | Pointwise      | Global transport  |
| Optimization | Unstable       | Differentiable    |

---

## Animation

* KL side:

  * explode / glitch
* Sinkhorn side:

  * smooth flow animation

---

## Voice-over (final synthesis)

“KL divergence so sánh hai phân phối bằng cách nhìn vào mật độ tại từng điểm.”

“Trong khi đó, Sinkhorn nhìn toàn cục — xem cách ‘biến đổi’ một phân phối thành phân phối khác.”

“Đây chính là lý do nó phù hợp cho bài toán soft matching trong không gian embedding.”

---

# FINAL INSIGHT (very important for your narrative)

Hiển thị text lớn:

“KL = Compare probabilities
Sinkhorn = Move probability mass”