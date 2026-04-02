# Hướng dẫn sử dụng Manim cho kịch bản CDR

File `src.py` chứa mã nguồn Manim để render các scene quan trọng trong kịch bản Cross-Domain Recommendation (CDR). Dưới đây là hướng dẫn cách cài đặt và chạy mã.

## Cài đặt

Cần đảm bảo bạn đã cài đặt Python (từ bản 3.8 trở lên) và pip.

1. **Cài đặt thư viện Manim:**
   Mở terminal và chạy lệnh:
   ```bash
   pip install manim
   ```
2. **Cài đặt các thư viện hệ thống cần thiết (nếu có):**
   Manim phụ thuộc vào FFmpeg, LaTeX, thư viện vẽ pango, v.v.
   - **MacOS:**
     ```bash
     brew install py3cairo ffmpeg pango pkg-config
     brew install mactex # Nếu bạn muốn render text dạng toán học nâng cao bằng LaTex
     ```
   - **Windows:** Nên dùng bộ cài Manim trên Chocolatey hoặc tham khảo [Document của Manim](https://docs.manim.community/en/stable/installation.html).

## Chạy mã để tạo video

Đến thư mục chứa file `src.py`:
```bash
cd /Users/tgng_mac/Coding/applied-data-science/manim-project/src/full/
```

Tham số lệnh chạy:
```bash
manim -p -ql src.py CDRPresentation
```

### Giải thích các tham số:
- `-p` (hoặc `--preview`): Tự động mở video kết quả sau khi render xong.
- `-q` (hoặc `--quality`): Chỉ định chất lượng video.
  - `-ql`: Quality Low (480p, 15fps) - render rất nhanh, lý tưởng trong lúc dev/test.
  - `-qm`: Quality Medium (720p, 30fps).
  - `-qh`: Quality High (1080p, 60fps) - dùng để render bản final.
  - `-qk`: Quality 4K (2160p, 60fps).
- `src.py`: Tên file chứa script.
- `CDRPresentation`: Tên class Scene bạn muốn render (kế thừa từ `Scene`).

## Kết quả xuất ra

Video render xong sẽ được lưu trữ tự động trong thư mục con `media/` của thư mục hiện tại.
Đường dẫn thông thường:
`media/videos/src/480p15/CDRPresentation.mp4`

## Mở rộng chức năng

- Có thể tạo thêm các helper function (ví dụ: `def scene_12_loss_landscape(self):`) trong class `CDRPresentation`.
- Để thay đổi voice-over hay flow animation, thay đổi các thông số về thời gian `run_time`, đối số `wait()` hoặc ghép thêm tệp âm thanh trực tiếp trong manim (add_sound).
