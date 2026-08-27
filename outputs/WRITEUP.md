# Lab 25 — GPU FinOps: Bài viết ngắn

**Tác giả:** Phạm Đức Hải Triều · **Vai trò:** FinOps Engineer @ NimbusAI
**Ngày:** 2026-08-27 · Số liệu: snapshot tháng 6/2026 (seed=25)

---

## 1. Baseline vs. Optimized

| Chỉ số | Baseline | Optimized | Thay đổi |
|---|---|---|---|
| Tổng chi phí GPU / tháng | **$27,133** | **$14,626** | **−46%** (−$12,507) |
| Inference `$/1M-token` | **$6.488** | **$1.126** | **−82.6%** |
| Purchasing / tháng (workloads) | $25,667 | $15,627 | −39.1% |

Bốn đòn bẩy trong báo cáo tổng hợp (`outputs/report.md`):

| Đòn bẩy | Tiết kiệm / tháng |
|---|---|
| Purchasing (spot/reserved) | $10,040 |
| Inference (cascade/cache/batch) | $1,212 |
| Right-size util-lies | $655 |
| Kill idle GPUs | $600 |

> Lưu ý về đơn vị: đòn bẩy inference nhỏ về **tổng $** (traffic hiện tại thấp) nhưng
> khổng lồ về **$/1M-token** (−82.6%) — đây là con số dùng để so sánh giữa các đội và
> để dự báo khi traffic tăng 10×.

## 2. Phân tích từng đòn bẩy — cái nào đáng làm trước?

Thứ tự **theo ROI**, không theo độ lớn tuyệt đối:

1. **Cascade routing** — 80% request đủ đơn giản cho model nhỏ (rẻ 15×). Không capex,
   đảo ngược được, hiệu quả $/công-sức cao nhất. Đây là nguồn chính của mức
   `$6.488 → $1.126 /1M-token`.
2. **Prompt caching (chat/RAG)** — system prompt tĩnh được đọc lại ~600 lần/prefix
   (Ext 3), vượt xa điểm hòa vốn (0.28 lần đọc với giá kiểu Anthropic). Chỉ là thay
   đổi cấu hình.
3. **Batch API cho eval** — traffic eval không cần real-time → chiết khấu 50%.
   `discount_stack(batch + 100% cache) = 0.05` (giảm 95%).
4. **Kill idle + right-size util-lies** — vệ sinh vận hành: $600 + $655/tháng, không
   ảnh hưởng người dùng.
5. **Commitment purchasing (spot/reserved)** — **lớn nhất về $ ($10,040/tháng)**
   nhưng khóa chi tiêu. Làm **sau cùng**, sau khi workload mix ổn định, và ưu tiên
   1yr cho tới khi job chứng minh sống quá 3 năm (Ext 1).

**Đòn bẩy đóng góp nhiều $ nhất: Purchasing.** **Đòn bẩy đáng làm đầu tiên: Cascade.**

## 3. GPU-Util Lie

`nvidia-smi` GPU-Util chỉ báo *có luồng nào trú trên SM trong cửa sổ lấy mẫu vừa rồi
hay không* — nó là **đồng hồ duty-cycle**, không phải thước đo hiệu quả.

- `gpu-h100-4`: **98% GPU-Util nhưng MFU ≈ 0.19** → ~80% FLOPs tensor-core đã thuê
  không tạo ra gì.
- `gpu-a10g-1`: **97% GPU-Util, MFU ≈ 0.27** — cùng bệnh.

**Cơ chế:** memory stall. Kernel chờ HBM (batch nhỏ, attention không fused, KV-cache
thrash) nên SM "bận" nhưng chỉ đang spin trên load. Bạn vẫn trả trọn giá GPU-giờ:
một H100 giá $2.50/giờ chỉ giao khối lượng công việc cỡ A100.

**Tác động tài chính:** hạ cấp 2 GPU "lie" xuống tier phù hợp ≈ **$655/tháng**
(theo `$/GPU-hr`) hoặc **$1,591/tháng** nếu right-size theo **bandwidth thực đạt**
(`$/(TB/s)·giờ`, Ext 2) — vì decode bị memory-bound, không bao giờ chạm tới phần
FLOPs dư mà bạn đang trả tiền.

## 4. Phần mở rộng đã làm (5/5)

Chi tiết + bảng số trong `outputs/extensions.md`. Test tự viết: `tests/test_extensions.py`
(9 test, pass). Tóm tắt:

| # | Nội dung | Kết quả đo được | Insight |
|---|---|---|---|
| **Ext 1** | `recommend_tier_v2`: thêm interruption-rate theo GPU + so sánh 1yr/3yr | v1 = 39.1% saved → **v2 = 30.0% saved** | Savings *giảm* là đúng: mọi job < 365 ngày nên 3yr lock-in vô lý (v2 chọn 1yr −20%); job spot trên L4/A10G churn 12–15%/giờ nên v2 đẩy khỏi spot. v2 đổi headline lấy hồ sơ cam kết chịu được thực tế. |
| **Ext 2** | Right-size theo MBU: `$/(TB/s)·giờ` thay vì `$/GPU-hr` | 3 GPU memory-bound → hạ cấp, **$1,591/tháng** | L4 có `$/GPU-hr` rẻ nhất nhưng `$/(TB/s)` = 2.667 — **đắt nhất** cho decode. Chọn card rẻ nhất mà *bandwidth* vẫn phủ nhu cầu thực + 20% headroom. |
| **Ext 3** | `cache_is_worth_it()` + `cache_break_even_reads()` | Hòa vốn: **0.28 lần đọc** (Anthropic) / **2.2 lần** (Gemini storage-billed). Dataset: **~600 đọc/prefix** | Caching thắng áp đảo ở đây. Hàm guard chỉ quan trọng với prefix hiếm dùng lại (tài liệu one-off dài). |
| **Ext 4** | Tách ngân sách reasoning ($ và Wh) | Reasoning = **8.4% traffic** nhưng **16.5% chi phí** và **94.0% năng lượng**; **172× Wh/request** | Reasoning rẻ về $ (dùng model nhỏ) nhưng là *quả bom carbon*. Gate theo độ phức tạp, giả định 60% bị over-trigger → tiết kiệm **~17,650 Wh/ngày** (~6.7 kg CO2e/ngày @ us-east-1). |
| **Ext 5** | Carbon-aware scheduling job interruptible | Chuyển us-east-1 → europe-north1: **−626 kg CO2e/tháng**, **−$53.67/tháng điện**, cắt 92% carbon | `us-east-wa` rẻ nhất; `europe-north1` sạch nhất; `europe-central2` tệ cả hai. Trade-off: vùng sạch xa user → chỉ áp cho training/eval interruptible, **không** cho đường chat inference. |

## 5. Ba khuyến nghị đầu tiên cho NimbusAI

1. **Bật cascade routing + prompt caching tuần này.** Không capex, đảo ngược được,
   kéo `$/1M-token` từ $6.49 → $1.13. Gắn dashboard `$/1M-token` theo team để giữ
   kỷ luật khi traffic tăng.
2. **Vệ sinh fleet: tắt GPU idle, right-size 2 GPU "util-lie" theo bandwidth.**
   ~$1,200–2,200/tháng, không ai để ý. Đồng thời sửa root cause (batch size,
   fused attention) để MFU của H100 lên ~0.4.
3. **Chỉ commit reserved sau khi có 4–6 tuần dữ liệu duty-cycle ổn định, và mặc
   định 1yr.** Dùng spot + checkpoint cho mọi job interruptible trên H100/A100
   (churn < 5%/giờ). Song song, dời job training interruptible sang `europe-north1`
   để cắt carbon 92% gần như miễn phí.

---

### Cách tái tạo

```bash
pip install -r requirements.txt
python data/generate.py
python missions/run_all.py        # M1–M5, ghi outputs/report.md + savings.png + focus_export.csv
python missions/extensions.py     # 5 extension, ghi outputs/extensions.md
python verify.py                  # 11/11
pytest -q                         # 24 passed (15 gốc + 9 test extension tự viết)
```
