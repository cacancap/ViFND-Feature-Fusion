"""
ViFND - Vietnamese Fake News Detection
Module: app.py
Vai trò: Thành viên 5 (TV5) - Giao diện Web Trực quan Tinh gọn (Gradio 4.x/5.x)

Tuân thủ nguyên tắc KISS (Keep It Simple, Stupid) với các khối chức năng cốt lõi:
- KHỐI 1: Module Đầu vào (Dán URL tự cào hoặc Nhập văn bản tùy ý, Chọn nhanh bài mẫu)
- KHỐI 2: Module Dự đoán (Nhãn phân loại, Đồng hồ đo rủi ro, Điểm tin cậy, Chỉ số đối soát đa nguồn)
- KHỐI 3: Module Minh giải XAI (Biểu đồ thanh ngang SHAP Waterfall & Bằng chứng đối chứng từ Báo chí Chính thống)
"""

import sys
from pathlib import Path

# Đảm bảo project root trong sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

import time
import requests
from bs4 import BeautifulSoup
import gradio as gr
import plotly.graph_objects as go

from src.models.fusion_classifier import predict_with_shap

# Các mẫu ví dụ chuẩn bị sẵn cho buổi bảo vệ thuyết trình 15 phút
PRESET_SAMPLES = {
    "⚠️ Mẫu 1: Tin giả y tế giật gân (Thuốc thần kỳ)": {
        "title": "CHẤN ĐỘNG: Phát hiện bài thuốc thần kỳ chữa khỏi 100% ung thư trong 7 ngày!!!",
        "content": "Bà con ơi chia sẻ ngay kẻo bài viết bị gỡ! Em thề bài thuốc gia truyền 3 đời này chữa dứt điểm mọi loại u hạch. Bác sĩ bệnh viện trả về uống vào là khỏi tiệt nọc. Đừng dại dột đi xạ trị tốn tiền, hãy cứu lấy người thân ngay hôm nay!",
        "url": "https://mangxahoi.vn/chia-se/bai-thuoc-than-ky-chua-ung-thu"
    },
    "📰 Mẫu 2: Báo chính thống (Kinh tế & Ngân hàng)": {
        "title": "Ngân hàng Nhà nước duy trì mặt bằng lãi suất điều hành ổn định",
        "content": "Theo báo cáo mới nhất từ Ngân hàng Nhà nước, cơ quan này tiếp tục điều hành chính sách tiền tệ chủ động, linh hoạt nhằm kiểm soát lạm phát và ổn định kinh tế vĩ mô. Thanh khoản hệ thống tổ chức tín dụng được đảm bảo thông suốt, đáp ứng kịp thời nhu cầu vốn phục vụ sản xuất kinh doanh của doanh nghiệp.",
        "url": "https://vnexpress.net/ngan-hang-nha-nuoc-duy-tri-lai-suat-on-dinh.html"
    },
    "🚨 Mẫu 3: Lừa đảo đầu tư tài chính sinh lời 300%": {
        "title": "BẬT MÍ: Cơ hội làm giàu không khó, đầu tư sinh lời 300% sau một đêm!",
        "content": "Các bác ạ, em nói thật ko làm thì đói, tham gia ngay nhóm kín để nhận tín hiệu VIP từ chuyên gia quốc tế. Cam kết bao lỗ 100%, bảo hiểm vốn trọn đời, ngồi nhà lướt điện thoại tiền tự về tài khoản ATM. Nhanh tay đăng ký duy nhất hôm nay kẻo lỡ!",
        "url": "https://dau-tu-sinh-loi.online/co-hoi-vang"
    }
}


def crawl_url_article(url: str):
    """
    Tự động cào và trích xuất tiêu đề & thân bài từ liên kết báo chí.
    Thời gian thực thi < 0.8 giây.
    """
    url = url.strip()
    if not url:
        return "", "", "❌ Vui lòng dán một đường dẫn URL hợp lệ."

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    try:
        start_time = time.time()
        resp = requests.get(url, headers=headers, timeout=5)
        resp.raise_for_status()
        
        soup = BeautifulSoup(resp.content, "html.parser")
        
        # 1. Trích xuất tiêu đề
        title = ""
        h1 = soup.find("h1")
        if h1 and h1.get_text(strip=True):
            title = h1.get_text(strip=True)
        elif soup.title and soup.title.get_text(strip=True):
            title = soup.title.get_text(strip=True)

        # 2. Bóc tách nội dung thân bài theo thẻ <p>
        paragraphs = []
        for p in soup.find_all("p"):
            p_text = p.get_text(strip=True)
            # Lọc bỏ các dòng ngắn hoặc quảng cáo
            if len(p_text) > 40 and not any(kw in p_text.lower() for kw in ["quảng cáo", "liên hệ", "hotline", "bản quyền", "xem thêm"]):
                paragraphs.append(p_text)
                
        content = " ".join(paragraphs[:15]) # Lấy tối đa 15 đoạn văn đầu
        elapsed = time.time() - start_time

        if not content and not title:
            return "", "", f"⚠️ Không bóc tách được nội dung tự động ({elapsed:.2f}s). Vui lòng sao chép văn bản thủ công."

        return title, content, f"✅ Đã cào thành công trong {elapsed:.2f}s ({len(paragraphs)} đoạn văn)."
    except Exception as e:
        return "", "", f"❌ Lỗi khi cào URL: {str(e)}"


def create_shap_plot(top_factors):
    """
    Tạo biểu đồ thanh ngang SHAP Waterfall bằng Plotly.
    Thanh màu đỏ: Đẩy về Tin giả / Giật gân (+)
    Thanh màu xanh: Đẩy về Tin thật / Xác thực (-)
    """
    if not top_factors:
        return go.Figure()

    # Sắp xếp để hiển thị từ trên xuống dưới
    factors_sorted = list(reversed(top_factors))
    
    labels = [f["feature"] for f in factors_sorted]
    values = [f["contribution"] for f in factors_sorted]
    colors = ["#E63946" if v > 0 else "#2A9D8F" for v in values]
    hover_texts = [f"{f['description_vi']}<br>Đóng góp SHAP: {v:+.3f}" for f, v in zip(factors_sorted, values)]

    fig = go.Figure(go.Bar(
        x=values,
        y=labels,
        orientation="h",
        marker_color=colors,
        text=[f"{v:+.3f}" for v in values],
        textposition="outside",
        hovertext=hover_texts,
        hoverinfo="text"
    ))

    fig.update_layout(
        title=dict(
            text="<b>Phân Rã Đóng Góp Đặc Trưng SHAP (TreeExplainer)</b>",
            font=dict(size=15, color="#1D3557")
        ),
        xaxis=dict(
            title="Giá trị SHAP (Dương: Tăng rủi ro Tin giả | Âm: Tăng độ tin cậy Tin thật)",
            zeroline=True,
            zerolinecolor="#333",
            zerolinewidth=2,
            gridcolor="#E0E0E0"
        ),
        yaxis=dict(title=""),
        margin=dict(l=180, r=40, t=50, b=40),
        height=340,
        plot_bgcolor="#F8F9FA",
        paper_bgcolor="#FFFFFF"
    )
    return fig


def analyze_news(title: str, content: str, url: str = ""):
    """
    Quy trình phân tích chính:
    Nhận bài báo -> Trích xuất Stylometrics & PhoBERT & Kiểm chứng đa nguồn -> Late Fusion -> SHAP Explanation.
    """
    title = title.strip()
    content = content.strip()

    if not title and not content:
        empty_fig = go.Figure()
        return (
            "<div style='color:red; font-size:16px;'>⚠️ Vui lòng nhập tiêu đề hoặc nội dung bài báo!</div>",
            empty_fig,
            ""
        )

    start_time = time.time()
    result = predict_with_shap(title, content, url)
    elapsed = time.time() - start_time

    # Xây dựng huy hiệu dự đoán (Prediction Badge)
    is_fake = result["prediction"] == "TIN GIẢ / GIẬT GÂN"
    badge_color = "#E63946" if is_fake else "#2A9D8F"
    badge_icon = "🚨" if is_fake else "✅"
    tier_color = {
        "CRITICAL": "#D90429",
        "MODERATE": "#FB8500",
        "LOW": "#2A9D8F"
    }.get(result["risk_tier"], "#333")

    prob_fake_pct = result["probability_fake"] * 100
    confidence_score = result["confidence_score"]

    veri_summary = result.get("cross_verification_summary", {})
    max_sim = veri_summary.get("max_corroboration_sim", 0.0)
    source_count = int(round(veri_summary.get("verified_source_count", 0.0) * 5.0))
    evidence_list = result.get("evidence_articles", [])

    result_html = f"""
    <div style="background: white; border-radius: 12px; padding: 20px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); border-left: 6px solid {badge_color};">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
            <div style="font-size: 22px; font-weight: bold; color: {badge_color};">
                {badge_icon} {result['prediction']}
            </div>
            <div style="background: {tier_color}; color: white; padding: 4px 12px; border-radius: 20px; font-weight: bold; font-size: 13px;">
                MỨC ĐỘ RỦI RO: {result['risk_tier']}
            </div>
        </div>
        
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px; margin-bottom: 15px; text-align: center;">
            <div style="background: #F8F9FA; padding: 10px; border-radius: 8px; border: 1px solid #E2E8F0;">
                <div style="font-size: 12px; color: #64748B !important; font-weight: 600;">XÁC SUẤT TIN GIẢ</div>
                <div style="font-size: 20px; font-weight: bold; color: {badge_color} !important;">{prob_fake_pct:.1f}%</div>
            </div>
            <div style="background: #F8F9FA; padding: 10px; border-radius: 8px; border: 1px solid #E2E8F0;">
                <div style="font-size: 12px; color: #64748B !important; font-weight: 600;">ĐỘ TIN CẬY MÔ HÌNH</div>
                <div style="font-size: 20px; font-weight: bold; color: #1D3557 !important;">{confidence_score:.1f}%</div>
            </div>
            <div style="background: #F8F9FA; padding: 10px; border-radius: 8px; border: 1px solid #E2E8F0;">
                <div style="font-size: 12px; color: #64748B !important; font-weight: 600;">THỜI GIAN XỬ LÝ</div>
                <div style="font-size: 20px; font-weight: bold; color: #0D9488 !important;">{elapsed:.2f}s</div>
            </div>
        </div>
        
        <div style="font-size: 13px; color: #334155 !important; border-top: 1px solid #E2E8F0; padding-top: 10px; line-height: 1.8;">
            <b>Chỉ số văn phong:</b> Dấu chấm than: <span style="font-family: monospace; background: #F1F5F9; color: #0F172A !important; padding: 2px 6px; border-radius: 4px; font-weight: 600; border: 1px solid #CBD5E1;">{result['stylometric_summary'].get('exclamation_density', 0)}</span> | Từ giật gân: <span style="font-family: monospace; background: #F1F5F9; color: #0F172A !important; padding: 2px 6px; border-radius: 4px; font-weight: 600; border: 1px solid #CBD5E1;">{result['stylometric_summary'].get('clickbait_keyword_hits', 0)} từ</span> | Tỷ lệ viết hoa: <span style="font-family: monospace; background: #F1F5F9; color: #0F172A !important; padding: 2px 6px; border-radius: 4px; font-weight: 600; border: 1px solid #CBD5E1;">{result['stylometric_summary'].get('upper_ratio', 0) * 100:.1f}%</span><br>
            <b>Kiểm chứng đa nguồn:</b> Độ tương đồng báo lớn: <span style="font-family: monospace; background: #F1F5F9; color: #0F172A !important; padding: 2px 6px; border-radius: 4px; font-weight: 600; border: 1px solid #CBD5E1;">{max_sim * 100:.1f}%</span> | Nguồn uy tín cùng xác nhận: <span style="font-family: monospace; background: #F1F5F9; color: #0F172A !important; padding: 2px 6px; border-radius: 4px; font-weight: 600; border: 1px solid #CBD5E1;">{source_count} nguồn</span>
        </div>
    </div>
    """

    # Tạo biểu đồ SHAP
    shap_fig = create_shap_plot(result["top_shap_factors"])

    # Tạo tóm tắt giải thích bằng tiếng Việt tự nhiên với card tương phản cao (High Contrast)
    shap_cards = []
    for f in result["top_shap_factors"]:
        is_risk = f["contribution"] > 0
        tag_bg = "#FEE2E2" if is_risk else "#DCFCE7"
        tag_color = "#991B1B" if is_risk else "#166534"
        tag_icon = "🔺" if is_risk else "🟢"
        tag_label = "Tăng nguy cơ tin giả" if is_risk else "Giảm nguy cơ tin giả"

        shap_cards.append(
            f"<div style='margin-bottom: 8px; padding: 10px 12px; background: #FFFFFF; border-radius: 6px; border: 1px solid #E2E8F0; line-height: 1.5;'>"
            f"  <div style='display: flex; align-items: center; gap: 8px; margin-bottom: 5px; flex-wrap: wrap;'>"
            f"    <span style='background: {tag_bg} !important; color: {tag_color} !important; font-size: 11px; padding: 3px 8px; border-radius: 12px; font-weight: bold; display: inline-flex; align-items: center; gap: 4px;'>"
            f"      <span>{tag_icon}</span> {tag_label}"
            f"    </span>"
            f"    <span style='font-family: Consolas, Monaco, monospace; background: #F1F5F9 !important; color: #0F172A !important; font-size: 12px; padding: 2px 7px; border-radius: 4px; font-weight: 600; border: 1px solid #CBD5E1;'>"
            f"      {f['feature']} ({f['contribution']:+.3f})"
            f"    </span>"
            f"  </div>"
            f"  <div style='color: #1E293B !important; font-size: 13px; font-weight: 500; padding-left: 2px;'>"
            f"    {f['description_vi']}"
            f"  </div>"
            f"</div>"
        )

    # Khối bằng chứng đối chứng
    evidence_html_items = []
    if evidence_list:
        for ev in evidence_list[:3]:
            title_text = ev.get("title", "")
            source_text = ev.get("source", "Báo chính thống")
            sim_val = ev.get("similarity", 0.0) * 100
            url_link = ev.get("url", "#")
            evidence_html_items.append(
                f"<div style='margin-bottom: 8px; padding: 10px; background: white; border-radius: 6px; border: 1px solid #E2E8F0;'>"
                f"  <span style='background:#E2E8F0 !important; color:#334155 !important; font-size:11px; padding:3px 8px; border-radius:4px; font-weight:bold;'>{source_text}</span> "
                f"  <a href='{url_link}' target='_blank' style='color:#1D4ED8 !important; font-weight:600; text-decoration:none;'>{title_text}</a> "
                f"  <span style='color:#64748B !important; font-size:12px;'>(Độ khớp: {sim_val:.1f}%)</span>"
                f"</div>"
            )
        evidence_content = "".join(evidence_html_items)
    else:
        evidence_content = "<div style='color:#DC2626 !important; font-size:13px; font-weight:500; background:white; padding:10px; border-radius:6px; border:1px solid #FECACA;'>⚠️ Không tìm thấy bài báo chính thống nào đưa tin về sự kiện này.</div>"

    explanation_html = f"""
    <div style="background: #F8F9FA; border-radius: 8px; padding: 16px; margin-top: 10px; border: 1px solid #DEE2E6;">
        <h4 style="margin-top: 0; color: #1D3557 !important; font-size: 15px; font-weight: bold;">💡 Tóm Tắt Minh Giải Khoa Học (Explainable AI - SHAP):</h4>
        <div style="margin-bottom: 14px;">
            {''.join(shap_cards)}
        </div>

        <h4 style="margin-top: 14px; margin-bottom: 8px; color: #1D3557 !important; font-size: 15px; font-weight: bold;">📰 Bằng Chứng Đối Chiếu Từ Báo Chí Chính Thống (Verified Cross-Source Evidence):</h4>
        {evidence_content}
    </div>
    """

    return result_html, shap_fig, explanation_html


def load_preset_sample(preset_name: str):
    """Nạp bài báo mẫu được cấu hình sẵn."""
    sample = PRESET_SAMPLES.get(preset_name, {})
    return (
        sample.get("url", ""),
        sample.get("title", ""),
        sample.get("content", "")
    )


# ==========================================
# KHỞI TẠO GIAO DIỆN GRADIO (THEME & LAYOUT)
# ==========================================
custom_theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="slate"
)

with gr.Blocks(title="ViFND - Hệ Thống Phát Hiện Tin Giả Tiếng Việt") as demo:
    gr.HTML("""
    <div style="text-align: center; padding: 15px 0 25px 0; border-bottom: 2px solid #E9ECEF; margin-bottom: 20px;">
        <h1 style="color: #1D3557; margin-bottom: 5px; font-size: 28px;">
            🇻🇳 HỆ THỐNG PHÁT HIỆN TIN GIẢ TIẾNG VIỆT (ViFND)
        </h1>
        <p style="color: #457B9D; font-size: 15px; margin: 0;">
            Hợp Nhất Ba Luồng: <b>15 Chỉ Số Hình Thức (Stylometrics)</b> ⊕ <b>768 Chiều Ngữ Nghĩa (PhoBERT)</b> ⊕ <b>Kiểm Chứng Chéo Đa Nguồn (Hybrid)</b> ⊕ <b>Minh Giải XAI (SHAP)</b>
        </p>
    </div>
    """)

    with gr.Row():
        # CỘT TRÁI: KHỐI 1 - NHẬP DỮ LIỆU
        with gr.Column(scale=5):
            gr.Markdown("### 📥 KHỐI 1: ĐẦU VÀO BÀI BÁO (INPUT MODULE)")
            
            with gr.Row():
                preset_selector = gr.Dropdown(
                    label="⚡ Chọn nhanh bài báo mẫu thử nghiệm:",
                    choices=list(PRESET_SAMPLES.keys()),
                    value=list(PRESET_SAMPLES.keys())[0],
                    interactive=True
                )
                btn_load_preset = gr.Button("Nạp bài mẫu", variant="secondary")

            with gr.Tabs():
                with gr.TabItem("🔗 Dán Đường Dẫn URL (Tự Động Cào)"):
                    url_input = gr.Textbox(
                        label="Đường dẫn bài báo (VnExpress, Tuổi Trẻ, blog mạng...):",
                        placeholder="https://vnexpress.net/...",
                        value=PRESET_SAMPLES[list(PRESET_SAMPLES.keys())[0]]["url"]
                    )
                    btn_crawl = gr.Button("🕷️ Trích xuất tự động từ URL", variant="secondary")
                    crawl_status = gr.Markdown("")

                with gr.TabItem("✍️ Nhập Văn Bản Trực Tiếp"):
                    gr.Markdown("Nhập trực tiếp tiêu đề và nội dung bài báo cần phân tích.")

            title_input = gr.Textbox(
                label="Tiêu đề bài báo (Title):",
                placeholder="Nhập tiêu đề tại đây...",
                lines=2,
                value=PRESET_SAMPLES[list(PRESET_SAMPLES.keys())[0]]["title"]
            )
            content_input = gr.Textbox(
                label="Nội dung bài báo (Content):",
                placeholder="Dán toàn bộ thân bài hoặc các đoạn trích quan trọng...",
                lines=6,
                value=PRESET_SAMPLES[list(PRESET_SAMPLES.keys())[0]]["content"]
            )

            btn_analyze = gr.Button("🔍 KIỂM TRA TIN TỨC & MINH GIẢI XAI", variant="primary", size="lg")

        # CỘT PHẢI: KHỐI 2 & KHỐI 3 - KẾT QUẢ & XAI
        with gr.Column(scale=6):
            gr.Markdown("### 📊 KHỐI 2: KẾT QUẢ DỰ ĐOÁN & ĐỒNG HỒ ĐO RỦI RO")
            prediction_output = gr.HTML("""
            <div style="background: #F8F9FA; border-radius: 12px; padding: 30px; text-align: center; border: 2px dashed #CED4DA;">
                <div style="font-size: 16px; color: #6C757D;">Vui lòng bấm nút <b>'KIỂM TRA TIN TỨC'</b> để hệ thống tiến hành trích xuất đặc trưng và đưa ra phân tích rủi ro.</div>
            </div>
            """)

            gr.Markdown("### 🔍 KHỐI 3: MINH GIẢI QUYẾT ĐỊNH XAI (SHAP WATERFALL) & ĐỐI CHIẾU")
            shap_plot_output = gr.Plot(label="Biểu đồ phân rã SHAP")
            explanation_output = gr.HTML("")

    # ==========================================
    # ĐIỀU KHIỂN SỰ KIỆN (EVENT WIRING)
    # ==========================================
    btn_load_preset.click(
        fn=load_preset_sample,
        inputs=[preset_selector],
        outputs=[url_input, title_input, content_input]
    )

    btn_crawl.click(
        fn=crawl_url_article,
        inputs=[url_input],
        outputs=[title_input, content_input, crawl_status]
    )

    btn_analyze.click(
        fn=analyze_news,
        inputs=[title_input, content_input, url_input],
        outputs=[prediction_output, shap_plot_output, explanation_output]
    )


if __name__ == "__main__":
    print("\n" + "=" * 65)
    print("🚀 Đang khởi động máy chủ Web UI ViFND tại http://localhost:7860 ...")
    print("=" * 65 + "\n")
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
