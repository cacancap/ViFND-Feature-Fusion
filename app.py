"""
ViFND - Vietnamese Fake News Detection
Module: app.py
Vai trò: Thành viên 5 (TV5) - Giao diện Web Trực quan Tinh gọn (Gradio 4.x/5.x)

Tuân thủ nguyên tắc KISS (Keep It Simple, Stupid) với 3 khối chức năng cốt lõi:
- KHỐI 1: Module Đầu vào (Dán URL tự cào hoặc Nhập văn bản tùy ý)
- KHỐI 2: Module Dự đoán (Nhãn phân loại, Đồng hồ đo rủi ro, Điểm tin cậy)
- KHỐI 3: Module Minh giải XAI (Biểu đồ thanh ngang SHAP Waterfall & Tóm tắt tiếng Việt)
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
            # Cắt bỏ tên báo ở đuôi (ví dụ: "... - Báo VnExpress")
            title = title.split(" - ")[0].split(" | ")[0]

        # 2. Trích xuất thân bài
        paragraphs = []
        # Các class chứa bài viết phổ biến
        article_elem = soup.find("article") or soup.find(class_=lambda c: c and any(k in str(c).lower() for k in ["detail", "content", "fck"]))
        if article_elem:
            p_tags = article_elem.find_all("p")
        else:
            p_tags = soup.find_all("p")

        for p in p_tags:
            text = p.get_text(strip=True)
            # Bỏ các đoạn ngắn vô nghĩa hoặc bản quyền
            if len(text) > 40 and not any(k in text.lower() for k in ["bản quyền", "copyright", "hotline", "liên hệ quảng cáo"]):
                paragraphs.append(text)

        content = "\n\n".join(paragraphs[:8])
        elapsed = time.time() - start_time

        if not title and not content:
            return "", "", "⚠️ Không thể trích xuất tự động từ trang này (có thể do trang chặn bot). Vui lòng nhập bằng tay."

        return title, content, f"✅ Đã trích xuất thành công từ nguồn trong {elapsed:.2f} giây."

    except Exception as e:
        return "", "", f"❌ Lỗi khi tải URL: {str(e)}"


def create_shap_plot(top_factors: list) -> go.Figure:
    """
    Tạo biểu đồ thanh ngang SHAP Waterfall minh giải quyết định bằng Plotly.
    Thanh đỏ: Tăng rủi ro tin giả (Positive SHAP).
    Thanh xanh: Giảm rủi ro tin giả (Negative SHAP).
    """
    if not top_factors:
        fig = go.Figure()
        fig.update_layout(title="Chưa có dữ liệu giải thích SHAP")
        return fig

    # Đảo ngược thứ tự để yếu tố quan trọng nhất nằm trên cùng
    factors = list(reversed(top_factors))
    features = [f["feature"] for f in factors]
    contributions = [f["contribution"] for f in factors]
    
    colors = [
        "#E63946" if c > 0 else "#2A9D8F"
        for c in contributions
    ]
    
    text_labels = [f"{c:+.3f}" for c in contributions]

    fig = go.Figure(go.Bar(
        x=contributions,
        y=features,
        orientation='h',
        marker=dict(color=colors, line=dict(color="#1D3557", width=1.5)),
        text=text_labels,
        textposition="outside",
    ))

    fig.update_layout(
        title=dict(
            text="<b>MỨC ĐỘ ĐÓNG GÓP CỦA CÁC ĐẶC TRƯNG VÀO QUYẾT ĐỊNH (SHAP VALUES)</b>",
            font=dict(size=14, color="#1D3557")
        ),
        xaxis=dict(
            title="Giá trị SHAP (Dương: Đẩy về Tin giả | Âm: Đẩy về Tin thật)",
            zeroline=True,
            zerolinecolor="#333",
            zerolinewidth=2,
            gridcolor="#E0E0E0"
        ),
        yaxis=dict(title=""),
        margin=dict(l=140, r=40, t=50, b=40),
        height=320,
        plot_bgcolor="#F8F9FA",
        paper_bgcolor="#FFFFFF"
    )
    return fig


def analyze_news(title: str, content: str, url: str = ""):
    """
    Quy trình phân tích chính:
    Nhận bài báo -> Trích xuất Stylometrics & PhoBERT -> Late Fusion -> SHAP Explanation.
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
            <div style="background: #F8F9FA; padding: 10px; border-radius: 8px;">
                <div style="font-size: 12px; color: #6C757D;">XÁC SUẤT TIN GIẢ</div>
                <div style="font-size: 20px; font-weight: bold; color: {badge_color};">{prob_fake_pct:.1f}%</div>
            </div>
            <div style="background: #F8F9FA; padding: 10px; border-radius: 8px;">
                <div style="font-size: 12px; color: #6C757D;">ĐỘ TIN CẬY MÔ HÌNH</div>
                <div style="font-size: 20px; font-weight: bold; color: #1D3557;">{confidence_score:.1f}%</div>
            </div>
            <div style="background: #F8F9FA; padding: 10px; border-radius: 8px;">
                <div style="font-size: 12px; color: #6C757D;">THỜI GIAN XỬ LÝ</div>
                <div style="font-size: 20px; font-weight: bold; color: #2A9D8F;">{elapsed:.2f}s</div>
            </div>
        </div>
        
        <div style="font-size: 13px; color: #495057; border-top: 1px solid #E9ECEF; pt: 10px; padding-top: 10px;">
            <b>Chỉ số hình thức:</b> 
            Dấu chấm than: <code>{result['stylometric_summary'].get('exclamation_density', 0)}</code> | 
            Từ giật gân: <code>{result['stylometric_summary'].get('clickbait_keyword_hits', 0)} từ</code> | 
            Tỷ lệ viết hoa: <code>{result['stylometric_summary'].get('upper_ratio', 0) * 100:.1f}%</code>
        </div>
    </div>
    """

    # Tạo biểu đồ SHAP
    shap_fig = create_shap_plot(result["top_shap_factors"])

    # Tạo tóm tắt giải thích bằng tiếng Việt tự nhiên
    bullet_points = []
    for f in result["top_shap_factors"]:
        sign_str = "🔺 <b>Tăng nguy cơ tin giả</b>" if f["contribution"] > 0 else "🟢 <b>Giảm nguy cơ tin giả</b>"
        bullet_points.append(
            f"<li>{sign_str} (<code>{f['feature']}</code> = {f['contribution']:+.3f}): {f['description_vi']}</li>"
        )

    explanation_html = f"""
    <div style="background: #F8F9FA; border-radius: 8px; padding: 16px; margin-top: 10px; border: 1px solid #DEE2E6;">
        <h4 style="margin-top: 0; color: #1D3557;">💡 Tóm Tắt Minh Giải Khoa Học (Explainable AI):</h4>
        <ul style="margin-bottom: 0; padding-left: 20px; line-height: 1.6; font-size: 14px; color: #212529;">
            {''.join(bullet_points)}
        </ul>
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
            🇻🇳 HỆ THỐNG PHÁT HIỆN TIN GIẢ TIẾNG VIỆT
        </h1>
        <p style="color: #457B9D; font-size: 15px; margin: 0;">
            Hợp Nhất Đặc Trưng Muộn: <b>15 Chỉ Số Hình Thức (Stylometrics)</b> ⊕ <b>768 Chiều Ngữ Nghĩa (PhoBERT)</b> ⊕ <b>Minh Giải XAI (SHAP)</b>
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

            gr.Markdown("### 🔍 KHỐI 3: MINH GIẢI QUYẾT ĐỊNH XAI (SHAP WATERFALL)")
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
