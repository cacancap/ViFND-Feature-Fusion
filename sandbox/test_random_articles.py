import sys
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path

PROJECT_ROOT = Path(r"D:\SourceCodes\Python_SourceCodes\ViFND-Feature-Fusion")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.fusion_classifier import predict_with_shap

test_articles = [
    {
        "id": "MẪU 1",
        "type": "TIN THẬT (Chính sách & Giáo dục)",
        "expected": "TIN THẬT / XÁC THỰC",
        "title": "Bộ Giáo dục và Đào tạo công bố phương án thi tốt nghiệp THPT từ năm 2025",
        "content": "Theo phương án được phê duyệt, kỳ thi tốt nghiệp trung học phổ thông từ năm 2025 gồm hai môn bắt buộc là Toán và Ngữ văn, cùng hai môn lựa chọn trong số các môn còn lại đã học ở lớp 12. Phương án này nhằm giảm áp lực thi cử và phù hợp với chương trình giáo dục phổ thông mới.",
        "url": "https://vnexpress.net/phuong-an-thi-tot-nghiep-thpt-2025.html"
    },
    {
        "id": "MẪU 2",
        "type": "TIN THẬT (Khoa học & Môi trường)",
        "expected": "TIN THẬT / XÁC THỰC",
        "title": "Việt Nam đẩy mạnh ứng dụng công nghệ viễn thám và trí tuệ nhân tạo dự báo bão lũ",
        "content": "Viện Hàn lâm Khoa học và Công nghệ Việt Nam cho biết hệ thống giám sát vệ tinh mới kết hợp mô hình học máy sẽ cung cấp hình ảnh độ phân giải cao phục vụ công tác giám sát tài nguyên rừng, theo dõi biến đổi khí hậu và cảnh báo sớm ngập úng đô thị.",
        "url": "https://tuoitre.vn/viet-nam-day-manh-cong-nghe-du-bao-bao-lu.htm"
    },
    {
        "id": "MẪU 3",
        "type": "TIN GIẢ / GIẬT GÂN (Lừa đảo y tế, phong cách la hét)",
        "expected": "TIN GIẢ / GIẬT GÂN",
        "title": "CHẤN ĐỘNG CẢ NƯỚC: Uống nước đậu đen luộc với tỏi khỏi sạch ung thư sau 5 ngày!!!",
        "content": "Bà con ơi chia sẻ ngay kẻo bài viết bị xoá mất! Bác sĩ bệnh viện lớn không muốn bạn biết bí mật này. Đừng để tiền mất tật mang đi xạ trị hóa trị tốn bạc tỷ, chỉ cần uống nước đậu đen với tỏi mỗi sáng là tiêu tan mọi tế bào ác tính. Ai không tin chia sẻ cho người nhà thử ngay!",
        "url": "https://mangxahoi.vn/chia-se-bai-thuoc-dan-gian-chua-ung-thu"
    },
    {
        "id": "MẪU 4",
        "type": "TIN GIẢ TINH VI (Bịa đặt sự kiện nhưng dùng văn phong đàng hoàng)",
        "expected": "TIN GIẢ / GIẬT GÂN",
        "title": "Chính phủ ban hành lệnh giới nghiêm và tạm dừng toàn bộ giao thông công cộng từ 22h đêm",
        "content": "Theo văn bản khẩn được lan truyền trên mạng, nhằm triển khai diễn tập an ninh diện rộng, các tuyến xe buýt, metro tại Hà Nội và TP.HCM sẽ tạm dừng vận hành vào khung giờ đêm trong 1 tháng. Người dân được yêu cầu hạn chế tối đa ra đường.",
        "url": "https://tin-tuc-mang-xa-hoi.com/tin-don-gioi-nghiem"
    },
    {
        "id": "MẪU 5",
        "type": "TIN GIẢ LỪA ĐẢO TÀI CHÍNH (Phishing)",
        "expected": "TIN GIẢ / GIẬT GÂN",
        "title": "CẢNH BÁO KHẨN: Vietcombank tặng 1.000.000 VNĐ tri ân khách hàng kỷ niệm thành lập, bấm link nhận ngay!",
        "content": "Tin vui cho toàn bộ chủ tài khoản! Nhân dịp kỷ niệm ngày thành lập, ngân hàng tặng mỗi khách hàng 1 triệu đồng tiền mặt chuyển thẳng vào tài khoản ATM. Quý khách vui lòng bấm link vietcombank-tri-an-2026.online nhập mã OTP xác thực trước 24h đêm nay kẻo lỡ!",
        "url": "https://vietcombank-tri-an-2026.online/nhan-thuong"
    }
]

print("=" * 85)
print("🧪 KIỂM THỬ THỰC TẾ 5 BÀI BÁO NGẪU NHIÊN VỚI HỆ THỐNG ViFND (TRI-STREAM FUSION)")
print("=" * 85 + "\n")

correct_count = 0
for item in test_articles:
    res = predict_with_shap(item["title"], item["content"], item["url"])
    pred_label = res["prediction"]
    is_correct = (pred_label == item["expected"])
    if is_correct:
        correct_count += 1
        status_tag = "✅ CHÍNH XÁC"
    else:
        status_tag = "❌ SAI LỆCH"

    prob_pct = res["probability_fake"] * 100
    conf_pct = res["confidence_score"]
    
    print(f"📌 [{item['id']}] Phân loại thực tế: {item['type']}")
    print(f"   Tiêu đề    : \"{item['title']}\"")
    print(f"   Dự đoán    : {pred_label} (Xác suất rủi ro: {prob_pct:.1f}% | Cấp độ: {res['risk_tier']}) -> {status_tag}")
    
    veri = res.get("cross_verification_summary", {})
    max_sim = veri.get("max_corroboration_sim", 0.0) * 100
    src_cnt = int(round(veri.get("verified_source_count", 0.0) * 5))
    penalty = veri.get("unverified_anomaly_penalty", 0.0)
    print(f"   Đối soát   : Khớp báo lớn: {max_sim:.1f}% | Nguồn uy tín cùng đưa tin: {src_cnt} nguồn | Điểm phạt bất thường: {penalty:.2f}")
    
    # Top 3 SHAP factors
    shap_strs = []
    for f in res["top_shap_factors"][:3]:
        shap_strs.append(f"{f['feature']} ({f['contribution']:+.3f})")
    print(f"   SHAP XAI   : {' | '.join(shap_strs)}")
    
    ev_list = res.get("evidence_articles", [])
    if ev_list:
        top_ev = ev_list[0]
        ev_title = top_ev.get("title", "")
        ev_src = top_ev.get("source", "Báo chính thống")
        ev_sim = top_ev.get("similarity", 0.0) * 100
        print(f"   Bằng chứng : [{ev_src}] {ev_title} (Độ khớp: {ev_sim:.1f}%)")
    else:
        print("   Bằng chứng : ⚠️ Không có cơ quan báo chí chính thống nào xác nhận sự kiện.")
    print("-" * 85)

print(f"\n🎯 TỔNG KẾT ĐỘ CHÍNH XÁC TRÊN 5 BÀI BÁO NGẪU NHIÊN: {correct_count}/{len(test_articles)} ({correct_count/len(test_articles)*100:.1f}%)")
