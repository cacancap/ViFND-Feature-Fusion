#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script: build_pitch_deck.py
Dự án: ViFND-Feature-Fusion (Phát Hiện Tin Giả Tiếng Việt Qua Hợp Nhất Đặc Trưng & XAI)
Mục đích: Tự động khởi tạo file thuyết trình PowerPoint đề xuất đề tài (.pptx)
          chuẩn học thuật, tỉ lệ 16:9, phong cách hiện đại cho môn học IT3190 (HUST SOICT).
          Tích hợp sơ đồ khối vector trực quan, bảng biểu định dạng chuẩn,
          và chuẩn hóa chính xác 15 chỉ số phong cách học Stylometrics.
"""

import os
import sys
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from pptx.dml.color import RGBColor

# ==============================================================================
# BẢNG MÀU CHUẨN HỌC THUẬT (ACADEMIC DESIGN SYSTEM)
# ==============================================================================
COLOR_NAVY = RGBColor(27, 54, 93)          # #1B365D - Tiêu đề chính, thanh phân cách, khung chính
COLOR_TEAL = RGBColor(0, 128, 128)         # #008080 - Điểm nhấn phụ, nhãn nổi bật, khung phụ
COLOR_CHARCOAL = RGBColor(34, 34, 34)      # #222222 - Văn bản nội dung chính
COLOR_LIGHT_GRAY = RGBColor(244, 246, 249) # #F4F6F9 - Nền card, khối nội dung
COLOR_BORDER = RGBColor(218, 224, 233)     # #DAE0E9 - Viền card mềm mại
COLOR_WHITE = RGBColor(255, 255, 255)      # #FFFFFF - Trắng tinh khiết
COLOR_MUTED = RGBColor(100, 110, 125)      # #646E7D - Chú thích mờ, số trang
COLOR_ALERT = RGBColor(192, 57, 43)        # #C0392B - Cảnh báo, tin giả, điểm mù
COLOR_SUCCESS = RGBColor(39, 174, 96)      # #27AE60 - Tin thật, tối ưu, thành công
COLOR_CARD_DARK = RGBColor(15, 32, 57)     # Navy siêu đậm cho card nổi bật
COLOR_ACCENT_BG = RGBColor(232, 244, 244)  # Teal siêu nhạt cho khối nổi bật

FONT_HEADING = "Calibri"
FONT_BODY = "Calibri"

def add_run(p, text="", name=FONT_BODY, size=Pt(12), bold=False, italic=False, color=COLOR_CHARCOAL):
    """Hàm bổ trợ thêm đoạn văn bản có định dạng chuẩn vào paragraph."""
    r = p.add_run()
    r.text = text
    r.font.name = name
    r.font.size = size
    r.font.bold = bold
    r.font.italic = italic
    r.font.color.rgb = color
    return r

def add_header(slide, title_text: str, category_text: str, slide_num: int):
    """Tạo thanh tiêu đề chuẩn học thuật phía trên cùng của mỗi slide."""
    tag_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.35), Inches(9.0), Inches(0.35))
    tf_tag = tag_box.text_frame
    tf_tag.word_wrap = True
    tf_tag.margin_left = tf_tag.margin_top = tf_tag.margin_right = tf_tag.margin_bottom = 0
    p_tag = tf_tag.paragraphs[0]
    add_run(p_tag, f"ĐỒ ÁN MÔN HỌC IT3190 — HỌC MÁY & KHAI PHÁ DỮ LIỆU  |  {category_text.upper()}",
            name=FONT_HEADING, size=Pt(10), bold=True, color=COLOR_TEAL)

    badge_box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(11.233), Inches(0.35), Inches(1.3), Inches(0.35))
    badge_box.fill.solid()
    badge_box.fill.fore_color.rgb = COLOR_LIGHT_GRAY
    badge_box.line.color.rgb = COLOR_BORDER
    badge_box.line.width = Pt(1)
    tf_badge = badge_box.text_frame
    tf_badge.vertical_anchor = MSO_ANCHOR.MIDDLE
    p_b = tf_badge.paragraphs[0]
    p_b.alignment = PP_ALIGN.CENTER
    add_run(p_b, f"SLIDE {slide_num:02d} / 08", name=FONT_HEADING, size=Pt(10), bold=True, color=COLOR_NAVY)

    title_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.70), Inches(11.733), Inches(0.55))
    tf_title = title_box.text_frame
    tf_title.word_wrap = True
    tf_title.margin_left = tf_title.margin_top = tf_title.margin_right = tf_title.margin_bottom = 0
    p_title = tf_title.paragraphs[0]
    add_run(p_title, title_text, name=FONT_HEADING, size=Pt(20), bold=True, color=COLOR_NAVY)

    divider = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(1.30), Inches(11.733), Inches(0.02))
    divider.fill.solid()
    divider.fill.fore_color.rgb = COLOR_TEAL
    divider.line.fill.background()

def add_card(slide, left, top, width, height, bg_color=COLOR_LIGHT_GRAY, border_color=COLOR_BORDER, border_width=Pt(1)):
    """Tạo hộp card bo góc bo tròn với nền và viền tùy chỉnh."""
    card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    card.fill.solid()
    card.fill.fore_color.rgb = bg_color
    if border_color:
        card.line.color.rgb = border_color
        card.line.width = border_width
    else:
        card.line.fill.background()
    return card

def add_vector_block(slide, left, top, width, height, title, subtitle="", bg_color=COLOR_NAVY, border_color=None, title_color=COLOR_WHITE, sub_color=COLOR_LIGHT_GRAY):
    """Tạo khối quy trình vector với định dạng văn bản căn giữa trực quan."""
    box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    box.fill.solid()
    box.fill.fore_color.rgb = bg_color
    if border_color:
        box.line.color.rgb = border_color
        box.line.width = Pt(1.2)
    else:
        box.line.fill.background()
    tf = box.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.margin_left = tf.margin_right = Inches(0.12)
    tf.margin_top = tf.margin_bottom = Inches(0.08)
    
    p0 = tf.paragraphs[0]
    p0.alignment = PP_ALIGN.CENTER
    add_run(p0, title, name=FONT_HEADING, size=Pt(10), bold=True, color=title_color)
    
    if subtitle:
        p1 = tf.add_paragraph()
        p1.alignment = PP_ALIGN.CENTER
        p1.space_before = Pt(2)
        add_run(p1, subtitle, name=FONT_BODY, size=Pt(8.5), color=sub_color)
    return box

def add_arrow_right(slide, left, top, width=Inches(0.24), height=Inches(0.18), color=COLOR_TEAL):
    """Tạo mũi tên vector chỉ hướng sang phải."""
    arrow = slide.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, left, top, width, height)
    arrow.fill.solid()
    arrow.fill.fore_color.rgb = color
    arrow.line.fill.background()
    return arrow

def create_deck(output_path: str):
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    # ==========================================================================
    # SLIDE 1: TRANG BÌA & ĐỊNH VỊ ĐỀ TÀI (TITLE & IDENTITY)
    # ==========================================================================
    slide1 = prs.slides.add_slide(blank_layout)

    # Background Top Header Bar
    top_bar = slide1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(0.35))
    top_bar.fill.solid()
    top_bar.fill.fore_color.rgb = COLOR_NAVY
    top_bar.line.fill.background()

    # University & Department Tag
    org_box = slide1.shapes.add_textbox(Inches(0.8), Inches(0.48), Inches(11.733), Inches(0.32))
    tf_org = org_box.text_frame
    tf_org.margin_left = tf_org.margin_top = tf_org.margin_right = tf_org.margin_bottom = 0
    p_org = tf_org.paragraphs[0]
    add_run(p_org, "TRƯỜNG CÔNG NGHỆ THÔNG TIN VÀ TRUYỀN THÔNG (SOICT) — ĐẠI HỌC BÁCH KHOA HÀ NỘI",
            name=FONT_HEADING, size=Pt(11), bold=True, color=COLOR_TEAL)

    course_box = slide1.shapes.add_textbox(Inches(0.8), Inches(0.78), Inches(11.733), Inches(0.3))
    tf_c = course_box.text_frame
    tf_c.margin_left = tf_c.margin_top = tf_c.margin_right = tf_c.margin_bottom = 0
    p_c = tf_c.paragraphs[0]
    add_run(p_c, "HỌC PHẦN: NHẬP MÔN HỌC MÁY VÀ KHAI PHÁ DỮ LIỆU (IT3190) — BÁO CÁO ĐỀ XUẤT ĐỀ TÀI ĐỒ ÁN",
            name=FONT_BODY, size=Pt(10), bold=False, color=COLOR_MUTED)

    # Hero Banner Box for Main Title
    hero_card = add_card(slide1, Inches(0.8), Inches(1.18), Inches(11.733), Inches(3.05), bg_color=COLOR_LIGHT_GRAY, border_color=COLOR_BORDER)
    hero_tf = hero_card.text_frame
    hero_tf.word_wrap = True
    hero_tf.margin_left = Inches(0.4)
    hero_tf.margin_top = Inches(0.25)
    hero_tf.margin_right = Inches(0.4)

    # Topic badge
    p_badge = hero_tf.paragraphs[0]
    add_run(p_badge, "ĐỀ TÀI 01: PHÁT HIỆN TIN GIẢ & TIN GIẬT GÂN TIẾNG VIỆT (ViFND)\n",
            name=FONT_HEADING, size=Pt(12), bold=True, color=COLOR_TEAL)

    # Main Project Title
    p_main = hero_tf.add_paragraph()
    p_main.space_before = Pt(3)
    add_run(p_main, "Phát Hiện Tin Giả Tiếng Việt Đa Chiều: Kết Hợp Biểu Diễn Ngữ Nghĩa Sâu (PhoBERT) & Đặc Trưng Phong Cách Viết (Stylometrics) Cùng AI Giải Thích Được (SHAP XAI)\n",
            name=FONT_HEADING, size=Pt(20.5), bold=True, color=COLOR_NAVY)

    # Subtitle matching exact prompt
    p_sub = hero_tf.add_paragraph()
    p_sub.space_before = Pt(4)
    add_run(p_sub, "Đề xuất Bài tập lớn môn Học máy & Khai phá Dữ liệu | Phương pháp tiếp cận Contract-First & Late Feature Fusion\n",
            name=FONT_HEADING, size=Pt(12), bold=True, color=COLOR_TEAL)

    # Tagline
    p_tagline = hero_tf.add_paragraph()
    p_tagline.space_before = Pt(2)
    add_run(p_tagline, "Vượt qua giới hạn 'hộp đen' và rào cản trùng lặp từ vựng bằng kỹ thuật Hợp nhất Đặc trưng Muộn (Late Feature Fusion) & Kiến trúc Vận hành Contract-First Zero-Blocking",
            name=FONT_BODY, size=Pt(11), italic=True, color=COLOR_CHARCOAL)

    # 3 Info Cards at Bottom
    c_w = Inches(3.68)
    c_h = Inches(2.72)
    gap = Inches(0.34)

    # Card 1: Team & Roles
    card1 = add_card(slide1, Inches(0.8), Inches(4.45), c_w, c_h, bg_color=COLOR_WHITE, border_color=COLOR_BORDER)
    tf1 = card1.text_frame
    tf1.word_wrap = True
    tf1.margin_left = tf1.margin_right = Inches(0.25)
    tf1.margin_top = Inches(0.2)
    p = tf1.paragraphs[0]
    add_run(p, "ĐỘI HÌNH THỰC HIỆN (NHÓM 3)\n", name=FONT_HEADING, size=Pt(11), bold=True, color=COLOR_NAVY)
    
    roles = [
        ("TV1", "Kỹ sư Dữ liệu: Cào RSS đa nguồn & Khử trùng MinHash LSH"),
        ("TV2", "Kỹ sư Ngôn ngữ: Tách từ PyVi & Trích xuất 15 Stylometrics"),
        ("TV3", "Kỹ sư Học sâu: PhoBERT Head-Tail 256 & Masked Pooling"),
        ("TV4", "Trưởng nhóm Mô hình: Late Fusion (XGB/NN) & SHAP XAI"),
        ("TV5", "Kỹ sư MLOps: Giao diện Web Gradio, REST API & Docker")
    ]
    for tv, role in roles:
        p_r = tf1.add_paragraph()
        p_r.space_before = Pt(4)
        add_run(p_r, f"• {tv}: ", size=Pt(9.5), bold=True, color=COLOR_TEAL)
        add_run(p_r, role, size=Pt(9), color=COLOR_CHARCOAL)

    # Card 2: Principles & Architecture
    card2 = add_card(slide1, Inches(0.8) + c_w + gap, Inches(4.45), c_w, c_h, bg_color=COLOR_WHITE, border_color=COLOR_BORDER)
    tf2 = card2.text_frame
    tf2.word_wrap = True
    tf2.margin_left = tf2.margin_right = Inches(0.25)
    tf2.margin_top = Inches(0.2)
    p = tf2.paragraphs[0]
    add_run(p, "PHƯƠNG PHÁP & NGUYÊN TẮC\n", name=FONT_HEADING, size=Pt(11), bold=True, color=COLOR_NAVY)
    
    pts2 = [
        ("Mô hình vận hành:", "Contract-First & Mock-Driven Architecture (lập trình song song Ngày 1)."),
        ("Kiến trúc cốt lõi:", "Late Feature Fusion 783D (768D Ngữ nghĩa sâu + 15D Phong cách hình thức)."),
        ("Khử trùng lặp:", "MinHash LSH Jaccard > 0.85 triệt tiêu bài sao chép, chống rò rỉ dữ liệu."),
        ("Minh giải quyết định:", "SHAP TreeExplainer bóc tách % rủi ro từng đặc trưng cho mỗi bài báo.")
    ]
    for k, v in pts2:
        p_pt = tf2.add_paragraph()
        p_pt.space_before = Pt(4)
        add_run(p_pt, f"✔ {k} ", size=Pt(9.5), bold=True, color=COLOR_NAVY)
        add_run(p_pt, v, size=Pt(9), color=COLOR_CHARCOAL)

    # Card 3: Defense Scope & Deliverables
    card3 = add_card(slide1, Inches(0.8) + (c_w + gap) * 2, Inches(4.45), c_w, c_h, bg_color=COLOR_WHITE, border_color=COLOR_BORDER)
    tf3 = card3.text_frame
    tf3.word_wrap = True
    tf3.margin_left = tf3.margin_right = Inches(0.25)
    tf3.margin_top = Inches(0.2)
    p = tf3.paragraphs[0]
    add_run(p, "MỤC TIÊU BẢO VỆ ĐỀ CƯƠNG\n", name=FONT_HEADING, size=Pt(11), bold=True, color=COLOR_NAVY)

    pts3 = [
        ("Phạm vi báo cáo:", "Đề xuất Ý tưởng & Thiết kế Hệ thống (không báo cáo kết quả hoàn chỉnh sớm)."),
        ("Thời lượng chuẩn mực:", "12 - 15 phút theo đúng quy chế môn học IT3190 (Viện CNTT&TT)."),
        ("Mã nguồn chuẩn:", "Repo GitHub ViFND-Feature-Fusion tích hợp sẵn bộ test tự động."),
        ("Sản phẩm cam kết:", "Corpus sạch 4,000+ bài, Checkpoints mô hình, Web Gradio < 1.5s.")
    ]
    for k, v in pts3:
        p_pt = tf3.add_paragraph()
        p_pt.space_before = Pt(4)
        add_run(p_pt, f"★ {k} ", size=Pt(9.5), bold=True, color=COLOR_TEAL)
        add_run(p_pt, v, size=Pt(9), color=COLOR_CHARCOAL)

    # ==========================================================================
    # SLIDE 2: BỐI CẢNH & 3 ĐIỂM MÙ TRUYỀN THỐNG (PROBLEM STATEMENT)
    # ==========================================================================
    slide2 = prs.slides.add_slide(blank_layout)
    add_header(slide2, "Bối Cảnh Bài Toán & 3 'Điểm Mù' Cốt Tử Của Mô Hình Truyền Thống", "1. Tính cấp thiết & Điểm nghẽn học thuật", 2)

    col_w = Inches(3.68)
    col_h = Inches(5.8)
    gap_col = Inches(0.34)

    # Column 1: Content Syndication Leakage
    card_col1 = add_card(slide2, Inches(0.8), Inches(1.45), col_w, col_h, bg_color=COLOR_LIGHT_GRAY, border_color=COLOR_ALERT, border_width=Pt(1.5))
    tf_c1 = card_col1.text_frame
    tf_c1.word_wrap = True
    tf_c1.margin_left = tf_c1.margin_right = Inches(0.25)
    tf_c1.margin_top = Inches(0.2)
    p = tf_c1.paragraphs[0]
    add_run(p, "ĐIỂM MÙ 1: RÒ RỈ DO SAO CHÉP\n(Content Syndication Leakage)", name=FONT_HEADING, size=Pt(12), bold=True, color=COLOR_ALERT)

    items_c1 = [
        ("Thực trạng không gian mạng", "Các trang lá cải, diễn đàn liên tục sao chép, xào xáo lại bài viết từ báo chính thống (thay đổi vài câu từ, giật lại tiêu đề)."),
        ("Lỗi kỹ thuật phổ biến", "Chia tập ngẫu nhiên (Random Train-Test Split) mà KHÔNG khử trùng lặp cấp độ ngữ nghĩa gần (Near-Duplicate)."),
        ("Hậu quả nghiêm trọng", "Cùng một bài báo xào lại xuất hiện ở cả Train và Test -> Điểm F1 ảo vọt lên 98-99%, nhưng mô hình chỉ 'học vẹt' bản ghi cũ thay vì học cách tổng quát hóa."),
        ("Giải pháp đề xuất", "Ứng dụng MinHash (128 hàm băm) kết hợp LSH Indexing (Jaccard > 0.85) để triệt tiêu toàn bộ bài báo xào nấu trước khi chia tập dữ liệu.")
    ]
    for title, desc in items_c1:
        p_t = tf_c1.add_paragraph()
        p_t.space_before = Pt(6)
        add_run(p_t, f"▶ {title}:\n", size=Pt(9.5), bold=True, color=COLOR_NAVY)
        add_run(p_t, desc, size=Pt(9), color=COLOR_CHARCOAL)

    # Column 2: Stylistic Blindness
    card_col2 = add_card(slide2, Inches(0.8) + col_w + gap_col, Inches(1.45), col_w, col_h, bg_color=COLOR_LIGHT_GRAY, border_color=COLOR_TEAL, border_width=Pt(1.5))
    tf_c2 = card_col2.text_frame
    tf_c2.word_wrap = True
    tf_c2.margin_left = tf_c2.margin_right = Inches(0.25)
    tf_c2.margin_top = Inches(0.2)
    p = tf_c2.paragraphs[0]
    add_run(p, "ĐIỂM MÙ 2: 'MÙ' HÌNH THỨC\n(Stylistic Blindness)", name=FONT_HEADING, size=Pt(12), bold=True, color=COLOR_TEAL)

    items_c2 = [
        ("Bẫy trùng lặp từ vựng", "Bài báo chính thống điều tra vụ án lừa đảo và bài viết bịa đặt giật gân có chung 85% từ vựng ('công an', 'bắt giữ', 'lừa đảo', 'tiền ảo')."),
        ("Lỗi kỹ thuật phổ biến", "Mô hình Pre-trained Transformer (BERT/PhoBERT) chỉ chú trọng ngữ nghĩa từ vựng mà hoàn toàn 'mù' trước hình thức trình bày bề mặt."),
        ("Hậu quả nghiêm trọng", "Mô hình bị đánh lừa bởi từ vựng giống nhau; bỏ qua mật độ dấu !, ?, viết IN HOA toàn bộ, và tỷ lệ tiêu đề dài bất thường so với thân bài."),
        ("Giải pháp đề xuất", "Trích xuất riêng nhánh 15 đặc trưng Stylometrics (hình thức, cấu trúc, dấu câu, cảm thán) bổ trợ trực giao cho vector ngữ nghĩa PhoBERT.")
    ]
    for title, desc in items_c2:
        p_t = tf_c2.add_paragraph()
        p_t.space_before = Pt(6)
        add_run(p_t, f"▶ {title}:\n", size=Pt(9.5), bold=True, color=COLOR_NAVY)
        add_run(p_t, desc, size=Pt(9), color=COLOR_CHARCOAL)

    # Column 3: Black-Box Trap
    card_col3 = add_card(slide2, Inches(0.8) + (col_w + gap_col) * 2, Inches(1.45), col_w, col_h, bg_color=COLOR_LIGHT_GRAY, border_color=COLOR_NAVY, border_width=Pt(1.5))
    tf_c3 = card_col3.text_frame
    tf_c3.word_wrap = True
    tf_c3.margin_left = tf_c3.margin_right = Inches(0.25)
    tf_c3.margin_top = Inches(0.2)
    p = tf_c3.paragraphs[0]
    add_run(p, "ĐIỂM MÙ 3: BẪY 'HỘP ĐEN'\n(Black-Box Predictions)", name=FONT_HEADING, size=Pt(12), bold=True, color=COLOR_NAVY)

    items_c3 = [
        ("Yêu cầu thực tiễn", "Gắn cờ một bài viết là 'Tin giả' tác động trực tiếp đến uy tín pháp lý của tác giả, tòa soạn báo và trải nghiệm người dùng mạng."),
        ("Lỗi kỹ thuật phổ biến", "Các mô hình nơ-ron sâu thông thường chỉ trả về con số nhãn khô khốc (0 hoặc 1) kèm xác suất mà không thể giải thích nguyên nhân."),
        ("Hậu quả nghiêm trọng", "Hội đồng kiểm duyệt và người dùng không tin cậy mô hình 'hộp đen'; không thể ứng dụng vào các hệ thống kiểm chứng tin tức thực tế."),
        ("Giải pháp đề xuất", "Tích hợp SHAP TreeExplainer, bóc tách chính xác % đóng góp của từng đặc trưng (dấu câu, từ khóa kích động hay ngữ cảnh) trong < 1.5s.")
    ]
    for title, desc in items_c3:
        p_t = tf_c3.add_paragraph()
        p_t.space_before = Pt(6)
        add_run(p_t, f"▶ {title}:\n", size=Pt(9.5), bold=True, color=COLOR_NAVY)
        add_run(p_t, desc, size=Pt(9), color=COLOR_CHARCOAL)

    # ==========================================================================
    # SLIDE 3: GIẢI PHÁP ĐỀ XUẤT – SƠ ĐỒ KHỐI VECTOR TRỰC QUAN (PROPOSED SOLUTION)
    # ==========================================================================
    slide3 = prs.slides.add_slide(blank_layout)
    add_header(slide3, "Giải Pháp Đề Xuất: Kiến Trúc Hợp Nhất Đặc Trưng Muộn (Late Feature Fusion)", "2. Ý tưởng cốt lõi & Kiến trúc hệ thống", 3)

    # TOP HALF: Sơ đồ khối vector hoàn mỹ (Graphical Vector Pipeline Diagram)
    flow_bg = add_card(slide3, Inches(0.8), Inches(1.45), Inches(11.733), Inches(2.25), bg_color=COLOR_LIGHT_GRAY, border_color=COLOR_BORDER)
    
    # Title inside flow container
    lbl_box = slide3.shapes.add_textbox(Inches(1.0), Inches(1.50), Inches(11.3), Inches(0.3))
    tf_lbl = lbl_box.text_frame
    tf_lbl.margin_left = tf_lbl.margin_top = tf_lbl.margin_right = tf_lbl.margin_bottom = 0
    p_lbl = tf_lbl.paragraphs[0]
    add_run(p_lbl, "SƠ ĐỒ LUỒNG KIẾN TRÚC HỢP NHẤT ĐẶC TRƯNG MUỘN (END-TO-END FUSION PIPELINE)",
            name=FONT_HEADING, size=Pt(10.5), bold=True, color=COLOR_NAVY)

    # Block 1: Input Article
    add_vector_block(slide3, Inches(1.0), Inches(1.85), Inches(2.0), Inches(1.65),
                     "BÀI BÁO ĐẦU VÀO", "Tiêu đề + Thân bài\n(Đã tẩy bias nguồn & NFC)",
                     bg_color=COLOR_NAVY, border_color=None, title_color=COLOR_WHITE, sub_color=COLOR_LIGHT_GRAY)

    # Arrow 1 to branches
    add_arrow_right(slide3, Inches(3.08), Inches(2.15), width=Inches(0.24), height=Inches(0.18), color=COLOR_TEAL)
    add_arrow_right(slide3, Inches(3.08), Inches(2.95), width=Inches(0.24), height=Inches(0.18), color=COLOR_NAVY)

    # Block 2A: Stream 1 Stylometrics
    add_vector_block(slide3, Inches(3.38), Inches(1.85), Inches(2.65), Inches(0.78),
                     "LUỒNG 1: STYLOMETRICS (15D)", "Quét 100% toàn văn: Dấu câu, IN HOA, Lexicon",
                     bg_color=COLOR_TEAL, border_color=None, title_color=COLOR_WHITE, sub_color=COLOR_WHITE)

    # Block 2B: Stream 2 PhoBERT
    add_vector_block(slide3, Inches(3.38), Inches(2.72), Inches(2.65), Inches(0.78),
                     "LUỒNG 2: PHOBERT BASE v2 (768D)", "PyVi tách từ + Head-Tail 256 + Mean Pooling",
                     bg_color=COLOR_CARD_DARK, border_color=None, title_color=COLOR_WHITE, sub_color=COLOR_LIGHT_GRAY)

    # Arrow 2 to Fusion
    add_arrow_right(slide3, Inches(6.10), Inches(2.15), width=Inches(0.24), height=Inches(0.18), color=COLOR_TEAL)
    add_arrow_right(slide3, Inches(6.10), Inches(2.95), width=Inches(0.24), height=Inches(0.18), color=COLOR_NAVY)

    # Block 3: Late Fusion Layer (783D)
    add_vector_block(slide3, Inches(6.40), Inches(1.85), Inches(2.35), Inches(1.65),
                     "HỢP NHẤT ĐẶC TRƯNG (783D)", "Ghép nối ma trận:\n[ 768D Ngữ nghĩa sâu ||\n15D Phong cách hình thức ]\n(Chuẩn hóa L2 & Z-score)",
                     bg_color=COLOR_WHITE, border_color=COLOR_TEAL, title_color=COLOR_TEAL, sub_color=COLOR_CHARCOAL)

    # Arrow 3 to Classifier
    add_arrow_right(slide3, Inches(8.82), Inches(2.55), width=Inches(0.24), height=Inches(0.18), color=COLOR_TEAL)

    # Block 4: Classifier
    add_vector_block(slide3, Inches(9.12), Inches(1.85), Inches(1.85), Inches(1.65),
                     "BỘ PHÂN LOẠI", "XGBoost Classifier\n(Optuna 50 trials)\n----\nPyTorch Dual-Stream\nNeural Network",
                     bg_color=COLOR_LIGHT_GRAY, border_color=COLOR_NAVY, title_color=COLOR_NAVY, sub_color=COLOR_CHARCOAL)

    # Arrow 4 to Output
    add_arrow_right(slide3, Inches(11.04), Inches(2.55), width=Inches(0.24), height=Inches(0.18), color=COLOR_TEAL)

    # Block 5: Prediction & XAI
    add_vector_block(slide3, Inches(11.34), Inches(1.85), Inches(1.05), Inches(1.65),
                     "KẾT QUẢ\n& XAI", "Nhãn 0/1\n(Fake %)\n----\nSHAP\nWaterfall",
                     bg_color=COLOR_NAVY, border_color=None, title_color=COLOR_WHITE, sub_color=COLOR_LIGHT_GRAY)

    # BOTTOM HALF: 2 Deep Dive Columns
    w_bot = Inches(5.7)
    h_bot = Inches(3.35)

    # Left Box: Stream 1 Details
    box_s1 = add_card(slide3, Inches(0.8), Inches(3.85), w_bot, h_bot, bg_color=COLOR_LIGHT_GRAY, border_color=COLOR_TEAL, border_width=Pt(1.2))
    tf_s1 = box_s1.text_frame
    tf_s1.word_wrap = True
    tf_s1.margin_left = tf_s1.margin_right = Inches(0.3)
    tf_s1.margin_top = Inches(0.18)
    p = tf_s1.paragraphs[0]
    add_run(p, "LUỒNG 1: 'THÁM TỬ HÌNH THỨC' (STYLOMETRICS - 15D)\n", name=FONT_HEADING, size=Pt(11.5), bold=True, color=COLOR_TEAL)

    pts_s1 = [
        "Phạm vi quét: 100% toàn văn bài báo từ đầu đến cuối (không bị giới hạn độ dài token).",
        "Trọng tâm phân tích: Bắt trọn hành vi viết giật gân, khẩn cấp và câu view bề mặt.",
        "15 Chỉ số toán học: Mật độ dấu câu (!, ?, \", : -), tỷ lệ chữ viết IN HOA toàn bộ, độ lệch tiêu đề/thân bài, từ điển kích động cảm xúc ('chấn động', '100%').",
        "Chi phí tính toán: Cực nhẹ (< 2ms / bài viết trên CPU máy tính cá nhân).",
        "Giá trị thặng dư: Vô hiệu hóa 'Bẫy trùng lặp từ vựng' mà mô hình ngôn ngữ lớn mắc phải."
    ]
    for pt in pts_s1:
        p_pt = tf_s1.add_paragraph()
        p_pt.space_before = Pt(3)
        add_run(p_pt, f"• {pt}", size=Pt(9), color=COLOR_CHARCOAL)

    # Right Box: Stream 2 Details
    box_s2 = add_card(slide3, Inches(6.833), Inches(3.85), w_bot, h_bot, bg_color=COLOR_LIGHT_GRAY, border_color=COLOR_NAVY, border_width=Pt(1.2))
    tf_s2 = box_s2.text_frame
    tf_s2.word_wrap = True
    tf_s2.margin_left = tf_s2.margin_right = Inches(0.3)
    tf_s2.margin_top = Inches(0.18)
    p = tf_s2.paragraphs[0]
    add_run(p, "LUỒNG 2: 'CHUYÊN GIA NGỮ NGHĨA' (PHOBERT - 768D)\n", name=FONT_HEADING, size=Pt(11.5), bold=True, color=COLOR_NAVY)

    pts_s2 = [
        "Mô hình nền tảng: vinai/phobert-base-v2 (mô hình Transformer tối tân cho tiếng Việt).",
        "Tiền xử lý từ ghép: Tách từ PyVi bắt buộc ('học_sinh', 'chấn_động') trước khi nạp tokenizer.",
        "Chiến lược Head-Tail Truncation: Lấy 128 tokens đầu (Tiêu đề + Lede 5W1H) ghép 128 tokens cuối (Kết luận, cơ quan ngôn luận) để vừa trọn giới hạn 256 tokens mà không mất ý.",
        "Pooling & Chuẩn hóa: Attention-Masked Mean Pooling + Chuẩn hóa vector đơn vị L2.",
        "Tối ưu tài nguyên: Đóng băng 10 tầng đầu, chỉ tinh chỉnh nhẹ Layer 10 & 11 trên Colab T4."
    ]
    for pt in pts_s2:
        p_pt = tf_s2.add_paragraph()
        p_pt.space_before = Pt(3)
        add_run(p_pt, f"• {pt}", size=Pt(9), color=COLOR_CHARCOAL)

    # ==========================================================================
    # SLIDE 4: QUY TRÌNH KHAI PHÁ DỮ LIỆU NÂNG CAO (DATA MINING PIPELINE)
    # ==========================================================================
    slide4 = prs.slides.add_slide(blank_layout)
    add_header(slide4, "Quy Trình Khai Phá Dữ Liệu Nâng Cao & Khử Trùng Lặp MinHash LSH", "3. Khai phá dữ liệu & Tiền xử lý kỹ thuật cao", 4)

    h_step = Inches(1.8)
    w_step = Inches(11.733)
    top_s = Inches(1.45)
    gap_s = Inches(0.18)

    # Step 1: Multi-Source Harvesting
    card_s1 = add_card(slide4, Inches(0.8), top_s, w_step, h_step, bg_color=COLOR_LIGHT_GRAY, border_color=COLOR_TEAL)
    # Left step badge
    add_vector_block(slide4, Inches(0.95), top_s + Inches(0.2), Inches(1.6), Inches(1.4),
                     "BƯỚC 01", "THU THẬP\nĐA NGUỒN\n(4,000+ BÀI)",
                     bg_color=COLOR_TEAL, border_color=None, title_color=COLOR_WHITE, sub_color=COLOR_WHITE)
    
    tf1 = card_s1.text_frame
    tf1.word_wrap = True
    tf1.margin_left = Inches(1.9)
    tf1.margin_right = Inches(0.25)
    tf1.margin_top = Inches(0.15)
    p = tf1.paragraphs[0]
    add_run(p, "BƯỚC 1: THU THẬP ĐA NGUỒN CÂN BẰNG (MULTI-SOURCE HARVESTING — MỤC TIÊU 4,000+ BÀI)",
            name=FONT_HEADING, size=Pt(11), bold=True, color=COLOR_NAVY)

    p1 = tf1.add_paragraph()
    p1.space_before = Pt(3)
    add_run(p1, "• Nhãn Thật (Class 0 - 50%): ", size=Pt(9.5), bold=True, color=COLOR_SUCCESS)
    add_run(p1, "Cào tự động qua kênh RSS chuẩn của các báo chính thống uy tín (VnExpress, Tuổi Trẻ, VietnamNet) thuộc đa dạng chủ đề (Thời sự, Pháp luật, Kinh tế, Sức khỏe) kết hợp bộ chuẩn học thuật VLSP 2020 ReINTEL.\n",
            size=Pt(9), color=COLOR_CHARCOAL)
    add_run(p1, "• Nhãn Giả / Giật gân (Class 1 - 50%): ", size=Pt(9.5), bold=True, color=COLOR_ALERT)
    add_run(p1, "Tập hợp từ bộ dữ liệu mở kiểm duyệt (VFND, ViFactCheck) + Cào có chọn lọc từ các blog tin lá cải, bài thuốc chữa bệnh phản khoa học + Khôi phục nguồn tin giả từ Cổng Thông tin VAFC (tingia.gov.vn) và AFP Kiểm chứng.",
            size=Pt(9), color=COLOR_CHARCOAL)

    # Step 2: Source De-biasing & Normalization
    card_s2 = add_card(slide4, Inches(0.8), top_s + h_step + gap_s, w_step, h_step, bg_color=COLOR_LIGHT_GRAY, border_color=COLOR_TEAL)
    # Left step badge
    add_vector_block(slide4, Inches(0.95), top_s + h_step + gap_s + Inches(0.2), Inches(1.6), Inches(1.4),
                     "BƯỚC 02", "TẨY BIAS NGUỒN\n& CHUẨN HÓA\nUNICODE NFC",
                     bg_color=COLOR_NAVY, border_color=None, title_color=COLOR_WHITE, sub_color=COLOR_LIGHT_GRAY)

    tf2 = card_s2.text_frame
    tf2.word_wrap = True
    tf2.margin_left = Inches(1.9)
    tf2.margin_right = Inches(0.25)
    tf2.margin_top = Inches(0.15)
    p = tf2.paragraphs[0]
    add_run(p, "BƯỚC 2: TẨY DẤU VẾT NGUỒN & CHUẨN HÓA VĂN BẢN (SOURCE DE-BIASING & TEXT NORMALIZATION)",
            name=FONT_HEADING, size=Pt(11), bold=True, color=COLOR_NAVY)

    p2 = tf2.add_paragraph()
    p2.space_before = Pt(3)
    add_run(p2, "• Chống hiện tượng 'Học vẹt' (Shortcut Learning): ", size=Pt(9.5), bold=True, color=COLOR_TEAL)
    add_run(p2, "Dùng Regex quét sạch tên tòa soạn ở phần mở đầu bài (ví dụ: '(Dân trí) -', 'Theo VnExpress', 'VTV News -') và chữ ký tác giả/nguồn tin ở cuối bài. Ép mô hình phải phân loại dựa trên ngữ nghĩa và phong cách viết thay vì nhớ mặt tên báo.\n",
            size=Pt(9), color=COLOR_CHARCOAL)
    add_run(p2, "• Chuẩn hóa Bảng mã & Khử rác HTML: ", size=Pt(9.5), bold=True, color=COLOR_TEAL)
    add_run(p2, "Đồng bộ 100% bảng mã về Unicode NFC chuẩn (unicodedata.normalize) để triệt tiêu xung đột giữa ký tự dựng sẵn và tổ hợp trong tiếng Việt; bóc tách HTML thô bằng Trafilatura và BeautifulSoup4, loại bỏ quảng cáo, caption ảnh rác.",
            size=Pt(9), color=COLOR_CHARCOAL)

    # Step 3: MinHash & LSH Deduplication
    card_s3 = add_card(slide4, Inches(0.8), top_s + (h_step + gap_s) * 2, w_step, h_step, bg_color=COLOR_LIGHT_GRAY, border_color=COLOR_NAVY)
    # Left step badge
    add_vector_block(slide4, Inches(0.95), top_s + (h_step + gap_s) * 2 + Inches(0.2), Inches(1.6), Inches(1.4),
                     "BƯỚC 03", "KHỬ TRÙNG LẶP\nMINHASH & LSH\n(JACCARD > 0.85)",
                     bg_color=COLOR_CARD_DARK, border_color=None, title_color=COLOR_WHITE, sub_color=COLOR_LIGHT_GRAY)

    tf3 = card_s3.text_frame
    tf3.word_wrap = True
    tf3.margin_left = Inches(1.9)
    tf3.margin_right = Inches(0.25)
    tf3.margin_top = Inches(0.15)
    p = tf3.paragraphs[0]
    add_run(p, "BƯỚC 3: KHỬ TRÙNG LẶP CẬN ÂM BẰNG MINHASH & LSH (NEAR-DUPLICATE DEDUPLICATION — JACCARD > 0.85)",
            name=FONT_HEADING, size=Pt(11), bold=True, color=COLOR_NAVY)

    p3 = tf3.add_paragraph()
    p3.space_before = Pt(3)
    add_run(p3, "• Thuật toán MinHash & LSH Indexing: ", size=Pt(9.5), bold=True, color=COLOR_NAVY)
    add_run(p3, "Cắt văn bản thành tập hợp 3-word shingles, sinh vector chữ ký số MinHash với K = 128 hàm băm xác suất độc lập. Phân chia LSH thành b = 32 băng (bands), mỗi băng r = 4 hàng (32 x 4 = 128), thiết lập ngưỡng tương đồng Jaccard theta = 0.85.\n",
            size=Pt(9), color=COLOR_CHARCOAL)
    add_run(p3, "• Cơ chế giữ lại & Xuất file chuẩn: ", size=Pt(9.5), bold=True, color=COLOR_NAVY)
    add_run(p3, "Trong mỗi cụm bài báo trùng lặp Jaccard >= 0.85, chỉ giữ lại bản ghi có thời gian xuất bản sớm nhất (bài gốc), loại bỏ toàn bộ bài sao chép; đóng gói kết quả sạch sang data/processed/corpus_dedup.parquet (6 cột chuẩn).",
            size=Pt(9), color=COLOR_CHARCOAL)

    # ==========================================================================
    # SLIDE 5: KỸ THUẬT TRÍCH XUẤT ĐẶC TRƯNG KÉP (DUAL-STREAM ENGINEERING)
    # ==========================================================================
    slide5 = prs.slides.add_slide(blank_layout)
    add_header(slide5, "Kỹ Thuật Trích Xuất Kép: 15 Chỉ Số Phong Cách Học vs Biểu Diễn Ngữ Nghĩa PhoBERT", "4. Kỹ thuật đặc trưng chuyên sâu", 5)

    w_half = Inches(5.7)
    h_half = Inches(5.8)

    # Left Column: TV2 - 15 Stylometrics
    card_st = add_card(slide5, Inches(0.8), Inches(1.45), w_half, h_half, bg_color=COLOR_LIGHT_GRAY, border_color=COLOR_TEAL)
    tf_st = card_st.text_frame
    tf_st.word_wrap = True
    tf_st.margin_left = tf_st.margin_right = Inches(0.25)
    tf_st.margin_top = Inches(0.2)
    p = tf_st.paragraphs[0]
    add_run(p, "NHÁNH 1: 15 ĐẶC TRƯNG PHONG CÁCH HỌC (TV2 - 15D)\n", name=FONT_HEADING, size=Pt(11.5), bold=True, color=COLOR_TEAL)

    st_groups = [
        ("Nhóm Dấu câu & Cảm thán (4 chỉ số)", 
         "• exclamation_density: Mật độ dấu chấm than (!) / tổng số từ\n"
         "• question_density: Mật độ dấu hỏi chấm (?) kích thích tò mò\n"
         "• quote_density: Mật độ dấu ngoặc kép trích dẫn phát ngôn\n"
         "• colon_dash_density & punct_density: Dấu ngắt nhịp (: -) & tổng mật độ dấu câu"),
        ("Nhóm Hình thức & Cấu trúc (5 chỉ số)", 
         "• upper_ratio: Tỷ lệ ký tự viết IN HOA (hành vi la hét, khẩn cấp)\n"
         "• title_body_ratio: Tỷ lệ độ dài Tiêu đề / Thân bài (Clickbait có tít rất dài)\n"
         "• digit_in_title: Đếm số lượng chữ số ở tiêu đề (100%, 500k, 7 ngày...)\n"
         "• avg_sentence_len: Độ dài trung bình của câu (từ / câu)\n"
         "• teencode_ratio & type_token_ratio: Tỷ lệ viết tắt / độ phong phú từ vựng (TTR)"),
        ("Nhóm Ngữ dụng & Cảm xúc (6 chỉ số)", 
         "• clickbait_keyword_hits: Số lần xuất hiện từ giật tít ('chấn động', 'ngã ngửa')\n"
         "• urgency_density: Cụm từ thúc ép chia sẻ ('chia sẻ gấp', 'xem ngay kẻo xóa')\n"
         "• pronoun_density: Đại từ xưng hô phi chuẩn báo chí ('bà con ơi', 'em thề')\n"
         "• title_body_jaccard: Độ tương đồng từ vựng tiêu đề vs thân bài (bẫy lừa)\n"
         "• sentiment_extremity: Mức độ phân cực cảm xúc tuyệt đối |P_pos - P_neg|\n"
         "• domain_trust_score: Điểm tín nhiệm tên miền & nguồn phát tán tin tức")
    ]
    for grp_name, grp_content in st_groups:
        p_g = tf_st.add_paragraph()
        p_g.space_before = Pt(3)
        add_run(p_g, f"▶ {grp_name}:\n", size=Pt(9.5), bold=True, color=COLOR_NAVY)
        add_run(p_g, grp_content, size=Pt(8.5), color=COLOR_CHARCOAL)

    p_norm = tf_st.add_paragraph()
    p_norm.space_before = Pt(4)
    add_run(p_norm, "★ Chuẩn hóa: StandardScaler (mu=0, sigma=1) -> Xuất X_stylometric.npy (N x 15) đạt tốc độ trích xuất > 500 bài / giây trên CPU.",
            size=Pt(8.5), italic=True, bold=True, color=COLOR_TEAL)

    # Right Column: TV3 - PhoBERT Embeddings
    card_ph = add_card(slide5, Inches(6.833), Inches(1.45), w_half, h_half, bg_color=COLOR_LIGHT_GRAY, border_color=COLOR_NAVY)
    tf_ph = card_ph.text_frame
    tf_ph.word_wrap = True
    tf_ph.margin_left = tf_ph.margin_right = Inches(0.25)
    tf_ph.margin_top = Inches(0.2)
    p = tf_ph.paragraphs[0]
    add_run(p, "NHÁNH 2: BIỂU DIỄN NGỮ NGHĨA SÂU PHOBERT (TV3 - 768D)\n", name=FONT_HEADING, size=Pt(11.5), bold=True, color=COLOR_NAVY)

    ph_steps = [
        ("Tách từ ghép tiếng Việt (PyVi Word Segmentation)",
         "• Tiếng Việt là ngôn ngữ đơn lập, ranh giới từ không tương ứng dấu cách.\n"
         "• Dùng pyvi.ViTokenizer gom âm tiết thành từ ghép: 'học_sinh', 'chấn_động'.\n"
         "• Bắt buộc tiền xử lý trước khi nạp vào PhoBERT Tokenizer."),
        ("Chiến lược Head-Tail Truncation (256 Tokens)",
         "• Bài báo dài 800 - 1,500 từ nhưng PhoBERT chỉ nhận tối đa 256 subwords.\n"
         "• Thay vì cắt ngọn (tokens[:256] làm mất hoàn toàn thân & kết bài), nhóm lấy:\n"
         "  + 128 tokens đầu: Chứa Tiêu đề + Mở bài (Lede 5W1H của bài báo)\n"
         "  + 128 tokens cuối: Chứa Kết luận, lời đính chính, cơ quan phát ngôn\n"
         "• Bảo toàn > 90% tín hiệu ngữ nghĩa cốt lõi mà không bị tràn VRAM GPU."),
        ("Cơ chế Attention-Masked Mean Pooling & L2 Norm",
         "• Thay vì chỉ dùng token [CLS], áp dụng tính trung bình có mặt nạ chú ý:\n"
         "  z_sem = sum(m_i * h_i) / sum(m_i) (với m_i là mask 0/1 của token thật)\n"
         "• Chuẩn hóa vector đơn vị L2: ||z||_2 = 1.0 (khoảng cách Euclid tương đương trực tiếp với Cosine Similarity)."),
        ("Tối ưu GPU Colab T4 (Chi phí 0 VNĐ)",
         "• Đóng băng 10 tầng đầu, chỉ tinh chỉnh nhẹ 2 tầng cuối (Layer 10 & 11).\n"
         "• Batch size = 32, FP16: Mã hóa 4,000 bài báo chỉ mất < 3 phút trên Colab GPU; suy luận đơn lẻ trên CPU mất < 200ms.")
    ]
    for step_title, step_desc in ph_steps:
        p_s = tf_ph.add_paragraph()
        p_s.space_before = Pt(3)
        add_run(p_s, f"▶ {step_title}:\n", size=Pt(9.5), bold=True, color=COLOR_NAVY)
        add_run(p_s, step_desc, size=Pt(8.5), color=COLOR_CHARCOAL)

    p_sem_out = tf_ph.add_paragraph()
    p_sem_out.space_before = Pt(4)
    add_run(p_sem_out, "★ Xuất kho: X_semantic.npy (N x 768) kiểu float32 chuẩn hóa L2.",
            size=Pt(8.5), italic=True, bold=True, color=COLOR_NAVY)

    # ==========================================================================
    # SLIDE 6: MÔ HÌNH HÓA, ĐỐI SÁNH & XAI (MODELING, ABLATION & SHAP)
    # ==========================================================================
    slide6 = prs.slides.add_slide(blank_layout)
    add_header(slide6, "Thiết Kế Thí Nghiệm Đối Sánh 3 Cấp Độ (Ablation Study) & Minh Giải SHAP XAI", "5. Thiết kế thực nghiệm & Trí tuệ nhân tạo giải thích được", 6)

    # Section 1: 3-Tier Ablation Study Table / Flow (Top Half)
    card_ab = add_card(slide6, Inches(0.8), Inches(1.45), Inches(11.733), Inches(2.6), bg_color=COLOR_LIGHT_GRAY, border_color=COLOR_NAVY)
    tf_ab = card_ab.text_frame
    tf_ab.word_wrap = True
    tf_ab.margin_left = tf_ab.margin_right = Inches(0.25)
    tf_ab.margin_top = Inches(0.15)

    p = tf_ab.paragraphs[0]
    add_run(p, "THIẾT KẾ ĐỐI SÁNH 3 CẤP ĐỘ KHOA HỌC (3-TIER ABLATION STUDY DESIGN)",
            name=FONT_HEADING, size=Pt(11.5), bold=True, color=COLOR_NAVY)

    ab_items = [
        ("Baseline 1 (Mô hình Cổ điển):", "TF-IDF (1-2 gram, max_features=500) + 15 Stylometrics -> Logistic Regression / XGBoost Classifier. Mục tiêu: Đo lường ngưỡng chặn tối thiểu của các phương pháp học máy truyền thống."),
        ("Baseline 2 (Mô hình Đơn luồng sâu):", "Chỉ sử dụng 768 chiều biểu diễn ngữ nghĩa của PhoBERT -> Linear Classifier / XGBoost. Mục tiêu: Đánh giá trần hiệu năng của mô hình ngôn ngữ lớn khi không có thông tin phong cách."),
        ("Mô hình Đề xuất A (Late Fusion XGBoost):", "Ghép nối ma trận 783 chiều [768D Semantics || 15D Stylometrics] đưa vào XGBoost Classifier, tối ưu siêu tham số bằng Optuna (50 trials) và xử lý lệch lớp bằng scale_pos_weight."),
        ("Mô hình Đề xuất B (PyTorch Dual-Stream NN):", "Mạng nơ-ron sâu đa đầu vào: Nhánh ngữ nghĩa nén 768D -> 128D; Nhánh phong cách mở rộng 15D -> 32D; Hợp nhất tại tầng 160 chiều -> FC 64 -> Sigmoid (Giải quyết triệt để sự lệch pha số chiều).")
    ]
    for ab_name, ab_desc in ab_items:
        p_ab = tf_ab.add_paragraph()
        p_ab.space_before = Pt(3)
        add_run(p_ab, f"• {ab_name} ", size=Pt(9.5), bold=True, color=COLOR_TEAL)
        add_run(p_ab, ab_desc, size=Pt(9), color=COLOR_CHARCOAL)

    # Section 2: Metrics (Bottom Left)
    w_bot_l = Inches(5.7)
    card_met = add_card(slide6, Inches(0.8), Inches(4.2), w_bot_l, Inches(3.0), bg_color=COLOR_WHITE, border_color=COLOR_BORDER)
    tf_met = card_met.text_frame
    tf_met.word_wrap = True
    tf_met.margin_left = tf_met.margin_right = Inches(0.25)
    tf_met.margin_top = Inches(0.18)

    p = tf_met.paragraphs[0]
    add_run(p, "QUY TRÌNH ĐÁNH GIÁ & HỆ THỐNG CHỈ SỐ\n", name=FONT_HEADING, size=Pt(11), bold=True, color=COLOR_NAVY)

    met_pts = [
        ("Stratified 5-Fold Cross Validation:", "Đảm bảo phân bố tỷ lệ nhãn thật/giả đồng đều ở từng fold, triệt tiêu thiên lệch đánh giá."),
        ("Macro-F1 Score & Binary-F1:", "Đánh giá công bằng hiệu năng trên cả 2 lớp, đặc biệt quan trọng khi dữ liệu có thể chênh lệch nhẹ."),
        ("Precision & Recall (Độ chính xác & Thu hồi):", "Đo lường khả năng bắt trúng tin giả mà không bắt nhầm bài báo thật uy tín."),
        ("PR-AUC (Precision-Recall Area Under Curve):", "Thước đo đáng tin cậy hơn ROC-AUC trong các bài toán phân loại thông tin bất đối xứng.")
    ]
    for m_name, m_desc in met_pts:
        p_m = tf_met.add_paragraph()
        p_m.space_before = Pt(3)
        add_run(p_m, f"✔ {m_name} ", size=Pt(9), bold=True, color=COLOR_TEAL)
        add_run(p_m, m_desc, size=Pt(8.5), color=COLOR_CHARCOAL)

    # Section 3: Explainable AI SHAP (Bottom Right)
    w_bot_r = Inches(5.7)
    card_xai = add_card(slide6, Inches(6.833), Inches(4.2), w_bot_r, Inches(3.0), bg_color=COLOR_WHITE, border_color=COLOR_BORDER)
    tf_xai = card_xai.text_frame
    tf_xai.word_wrap = True
    tf_xai.margin_left = tf_xai.margin_right = Inches(0.25)
    tf_xai.margin_top = Inches(0.18)

    p = tf_xai.paragraphs[0]
    add_run(p, "MINH GIẢI QUYẾT ĐỊNH VỚI SHAP XAI (TV4)\n", name=FONT_HEADING, size=Pt(11), bold=True, color=COLOR_NAVY)

    xai_pts = [
        ("SHAP TreeExplainer:", "Tính toán giá trị Shapley từ lý thuyết trò chơi cho mô hình XGBoost tốt nhất trên không gian 783 chiều."),
        ("Biểu đồ toàn cục (Global Beeswarm Plot):", "Xếp hạng và chứng minh 15 đặc trưng Stylometrics đóng góp tỷ trọng quan trọng như thế nào so với ngữ nghĩa."),
        ("Biểu đồ cục bộ (Local Waterfall Plot):", "Bóc tách từng bài báo cụ thể ra các yếu tố tăng/giảm nguy cơ (ví dụ: +0.48 do dấu !, +0.35 do từ giật tít, -0.12 do ngữ nghĩa bài báo)."),
        ("Tích hợp thời gian thực vào Web Gradio:", "Xuất kết quả giải thích thân thiện bằng tiếng Việt tới người dùng cuối trong < 1.5 giây.")
    ]
    for x_name, x_desc in xai_pts:
        p_x = tf_xai.add_paragraph()
        p_x.space_before = Pt(3)
        add_run(p_x, f"★ {x_name} ", size=Pt(9), bold=True, color=COLOR_NAVY)
        add_run(p_x, x_desc, size=Pt(8.5), color=COLOR_CHARCOAL)

    # ==========================================================================
    # SLIDE 7: HỆ SINH THÁI CÔNG CỤ & HẠ TẦNG (TECH STACK & COMPUTING)
    # ==========================================================================
    slide7 = prs.slides.add_slide(blank_layout)
    add_header(slide7, "Hệ Sinh Thái Công Cụ Khai Phá & Tính Khả Thi Hạ Tầng (0 Đồng Chi Phí)", "6. Công nghệ & Tính khả thi kỹ thuật", 7)

    # Table of 4 Technology Layers with explicit compact row heights & vertical centering
    rows, cols = 5, 3
    left_t, top_t, width_t, height_t = Inches(0.8), Inches(1.45), Inches(7.5), Inches(5.6)
    table_shape = slide7.shapes.add_table(rows, cols, left_t, top_t, width_t, height_t)
    table = table_shape.table
    table.columns[0].width = Inches(1.8)
    table.columns[1].width = Inches(2.3)
    table.columns[2].width = Inches(3.4)

    # Explicit row heights
    table.rows[0].height = Inches(0.42)
    for r_i in range(1, 5):
        table.rows[r_i].height = Inches(1.18)

    headers = ["Phân Tầng Công Nghệ", "Thư Viện & Framework", "Vai Trò & Chức Năng Cốt Lõi"]
    for col_idx, h_text in enumerate(headers):
        cell = table.cell(0, col_idx)
        cell.fill.solid()
        cell.fill.fore_color.rgb = COLOR_NAVY
        cell.vertical_anchor = MSO_ANCHOR.MIDDLE
        tf = cell.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_right = Inches(0.12)
        tf.margin_top = tf.margin_bottom = Inches(0.06)
        p = tf.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        add_run(p, h_text, name=FONT_HEADING, size=Pt(10), bold=True, color=COLOR_WHITE)

    data_rows = [
        ("Tầng 1: Thu thập & Khai phá dữ liệu", 
         "feedparser, trafilatura, BeautifulSoup4, datasketch, pyarrow",
         "Cào RSS tự động, bóc tách toàn văn sạch không dính quảng cáo, băm MinHash LSH khử trùng lặp Jaccard > 0.85, lưu Parquet nén Snappy."),
        ("Tầng 2: Xử lý Ngôn ngữ & Đặc trưng",
         "pyvi (ViTokenizer), transformers (HuggingFace), scikit-learn",
         "Tách từ ghép tiếng Việt, tải mô hình vinai/phobert-base-v2, tính 15 công thức Stylometrics, chuẩn hóa StandardScaler."),
        ("Tầng 3: Huấn luyện Mô hình & XAI",
         "PyTorch (Mixed Precision FP16), XGBoost, Optuna, SHAP",
         "Xây dựng mạng nơ-ron Dual-Stream Classifier, huấn luyện XGBoost Late Fusion, tối ưu siêu tham số 50 trials, giải thích Shapley values."),
        ("Tầng 4: Đóng gói & Triển khai Trực quan",
         "Gradio Blocks, FastAPI, Docker, Docker Compose",
         "Giao diện tương tác dán URL phân tích < 1.5s, cung cấp REST API chuẩn /predict, đóng gói môi trường độc lập chạy mọi máy tính.")
    ]

    for row_idx, row_data in enumerate(data_rows, start=1):
        for col_idx, cell_text in enumerate(row_data):
            cell = table.cell(row_idx, col_idx)
            cell.fill.solid()
            cell.fill.fore_color.rgb = COLOR_LIGHT_GRAY if row_idx % 2 == 1 else COLOR_WHITE
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            tf = cell.text_frame
            tf.word_wrap = True
            tf.margin_left = tf.margin_right = Inches(0.12)
            tf.margin_top = tf.margin_bottom = Inches(0.06)
            p = tf.paragraphs[0]
            is_bold = (col_idx == 0)
            c_color = COLOR_NAVY if col_idx == 0 else (COLOR_TEAL if col_idx == 1 else COLOR_CHARCOAL)
            add_run(p, cell_text, name=FONT_BODY, size=Pt(9), bold=is_bold, color=c_color)

    # Right Card: Hardware & Feasibility Callout (0 VND Cost)
    card_fe = add_card(slide7, Inches(8.55), Inches(1.45), Inches(3.98), Inches(5.6), bg_color=COLOR_LIGHT_GRAY, border_color=COLOR_TEAL, border_width=Pt(1.5))
    tf_fe = card_fe.text_frame
    tf_fe.word_wrap = True
    tf_fe.margin_left = tf_fe.margin_right = Inches(0.25)
    tf_fe.margin_top = Inches(0.2)

    p = tf_fe.paragraphs[0]
    add_run(p, "KIỂM CHỨNG TÍNH KHẢ THI TÍNH TOÁN\n(HẠ TẦNG 0 ĐỒNG CHI PHÍ)\n",
            name=FONT_HEADING, size=Pt(11.5), bold=True, color=COLOR_NAVY)

    fe_items = [
        ("Google Colab Free GPU (NVIDIA T4 15GB VRAM):",
         "• Chỉ tiêu tốn ~1.8GB VRAM khi chạy batch size 32 với PhoBERT v2.\n"
         "• Thời gian mã hóa toàn bộ 4,000 bài báo mất dưới 3 phút.\n"
         "• Bật Mixed Precision FP16 giúp tăng tốc gấp 2.4 lần."),
        ("CPU Máy Cá Nhân (Local Environment):",
         "• Chạy trơn tru toàn bộ pipeline cào tin, tách từ PyVi và MinHash LSH.\n"
         "• Tốc độ trích xuất 15 đặc trưng Stylometrics: > 500 bài / giây.\n"
         "• Thời gian huấn luyện XGBoost Late Fusion: < 45 giây."),
        ("Cam kết Kỹ thuật Học viên:",
         "• 100% mã nguồn chạy ổn định trên Conda thông thường.\n"
         "• Không phụ thuộc vào GPU thương mại hay API trả phí.")
    ]
    for fe_t, fe_d in fe_items:
        p_item = tf_fe.add_paragraph()
        p_item.space_before = Pt(4)
        add_run(p_item, f"★ {fe_t}\n", size=Pt(9.5), bold=True, color=COLOR_TEAL)
        add_run(p_item, fe_d, size=Pt(8.5), color=COLOR_CHARCOAL)

    # ==========================================================================
    # SLIDE 8: PHƯƠNG PHÁP CONTRACT-FIRST & PHÂN CÔNG 5 THÀNH VIÊN
    # ==========================================================================
    slide8 = prs.slides.add_slide(blank_layout)
    add_header(slide8, "Phương Pháp Luận Contract-First & Phân Công Nhiệm Vụ 5 Thành Viên", "7. Tổ chức đội ngũ & Kế hoạch triển khai", 8)

    # Top Half: Roles Table (5 Members) with compact header & vertical centering
    rows_r, cols_r = 6, 4
    left_r, top_r, width_r, height_r = Inches(0.8), Inches(1.45), Inches(11.733), Inches(2.8)
    table_shape_r = slide8.shapes.add_table(rows_r, cols_r, left_r, top_r, width_r, height_r)
    table_r = table_shape_r.table
    table_r.columns[0].width = Inches(1.2)
    table_r.columns[1].width = Inches(2.6)
    table_r.columns[2].width = Inches(4.5)
    table_r.columns[3].width = Inches(3.433)

    table_r.rows[0].height = Inches(0.40)
    for r_i in range(1, 6):
        table_r.rows[r_i].height = Inches(0.46)

    r_headers = ["Thành Viên", "Vai Trò Chuyên Môn", "Nhiệm Vụ Kỹ Thuật Chính", "Tập Tin Bàn Giao Thật (Deliverable)"]
    for col_idx, h_text in enumerate(r_headers):
        cell = table_r.cell(0, col_idx)
        cell.fill.solid()
        cell.fill.fore_color.rgb = COLOR_NAVY
        cell.vertical_anchor = MSO_ANCHOR.MIDDLE
        tf = cell.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_right = Inches(0.12)
        tf.margin_top = tf.margin_bottom = Inches(0.04)
        p = tf.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        add_run(p, h_text, name=FONT_HEADING, size=Pt(9.5), bold=True, color=COLOR_WHITE)

    members_data = [
        ("TV1", "Kỹ sư Dữ liệu & Web Crawling", "Cào RSS đa nguồn, bóc tách toàn văn sạch, khử trùng MinHash LSH (Jaccard > 0.85)", "corpus_dedup.parquet (4,000+ bài, 6 cột)"),
        ("TV2", "Kỹ sư Ngôn ngữ & Stylometrics", "Tách từ ghép PyVi, xây từ điển giật gân, lượng hóa 15 công thức phong cách học", "X_stylometric.npy (N x 15, StandardScaler)"),
        ("TV3", "Kỹ sư Học sâu PhoBERT", "Cắt tỉa Head-Tail 256 tokens, GPU Mean Pooling, chuẩn hóa vector đơn vị L2", "X_semantic.npy (N x 768, L2 Norm)"),
        ("TV4", "Trưởng nhóm Mô hình & XAI", "Huấn luyện Late Fusion (XGBoost + Dual-Stream NN), Stratified 5-Fold CV, SHAP Explainer", "best_fusion_xgboost.json & predict_with_shap()"),
        ("TV5", "Kỹ sư MLOps & Web UI Demo", "Xây dựng Web Gradio (Dán URL/Text), REST API FastAPI, đóng gói Docker, điều phối slide", "app.py, Dockerfile, Oral Defense Slides")
    ]

    for row_idx, r_data in enumerate(members_data, start=1):
        for col_idx, cell_text in enumerate(r_data):
            cell = table_r.cell(row_idx, col_idx)
            cell.fill.solid()
            cell.fill.fore_color.rgb = COLOR_LIGHT_GRAY if row_idx % 2 == 1 else COLOR_WHITE
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            tf = cell.text_frame
            tf.word_wrap = True
            tf.margin_left = tf.margin_right = Inches(0.12)
            tf.margin_top = tf.margin_bottom = Inches(0.04)
            p = tf.paragraphs[0]
            is_bold = (col_idx <= 1)
            c_color = COLOR_NAVY if col_idx == 0 else (COLOR_TEAL if col_idx == 1 else COLOR_CHARCOAL)
            add_run(p, cell_text, name=FONT_BODY, size=Pt(8.5), bold=is_bold, color=c_color)

    # Bottom Half: 3 Modular Cards for Contract-First Architecture
    c_w3 = Inches(3.68)
    c_h3 = Inches(2.70)
    gap3 = Inches(0.34)
    top_c3 = Inches(4.45)

    # Mini Card 1: Waterfall Pitfalls
    mc1 = add_card(slide8, Inches(0.8), top_c3, c_w3, c_h3, bg_color=COLOR_WHITE, border_color=COLOR_BORDER)
    tf_mc1 = mc1.text_frame
    tf_mc1.word_wrap = True
    tf_mc1.margin_left = tf_mc1.margin_right = Inches(0.2)
    tf_mc1.margin_top = Inches(0.15)
    p = tf_mc1.paragraphs[0]
    add_run(p, "BẪY NGHẼN THÁC NƯỚC (WATERFALL)\n", name=FONT_HEADING, size=Pt(10.5), bold=True, color=COLOR_ALERT)
    
    pts_mc1 = [
        "Phụ thuộc tuần tự: TV2 & TV3 phải chờ TV1 cào dữ liệu (mất 2-3 tuần đầu dự án).",
        "Áp lực mô hình: TV4 phải chờ feature matrix xong mới có thể dựng và tinh chỉnh mô hình.",
        "Rủi ro MLOps: TV5 chỉ có thể làm giao diện ở tuần cuối, dễ vỡ tiến độ khi tích hợp."
    ]
    for pt in pts_mc1:
        p_pt = tf_mc1.add_paragraph()
        p_pt.space_before = Pt(3)
        add_run(p_pt, f"⚠ {pt}", size=Pt(8.5), color=COLOR_CHARCOAL)

    # Mini Card 2: Contract-First & Mock Data
    mc2 = add_card(slide8, Inches(0.8) + c_w3 + gap3, top_c3, c_w3, c_h3, bg_color=COLOR_WHITE, border_color=COLOR_TEAL)
    tf_mc2 = mc2.text_frame
    tf_mc2.word_wrap = True
    tf_mc2.margin_left = tf_mc2.margin_right = Inches(0.2)
    tf_mc2.margin_top = Inches(0.15)
    p = tf_mc2.paragraphs[0]
    add_run(p, "GIẢI PHÁP CONTRACT-FIRST & MOCK\n", name=FONT_HEADING, size=Pt(10.5), bold=True, color=COLOR_TEAL)

    pts_mc2 = [
        "Đóng băng 4 Hợp đồng Dữ liệu: Khóa chặt schema Parquet, Tensor [N,15], Embedding [N,768] và JSON đầu ra.",
        "Bộ sinh Mock Fixtures: Chạy mock_data_generator.py sinh sẵn sample data trong data/fixtures/ ngay Ngày 1.",
        "Lập trình độc lập 100%: 5 thành viên lập trình song song từ Ngày 1 không chờ đợi bất kỳ ai."
    ]
    for pt in pts_mc2:
        p_pt = tf_mc2.add_paragraph()
        p_pt.space_before = Pt(3)
        add_run(p_pt, f"✔ {pt}", size=Pt(8.5), color=COLOR_CHARCOAL)

    # Mini Card 3: Continuous Verification
    mc3 = add_card(slide8, Inches(0.8) + (c_w3 + gap3) * 2, top_c3, c_w3, c_h3, bg_color=COLOR_WHITE, border_color=COLOR_NAVY)
    tf_mc3 = mc3.text_frame
    tf_mc3.word_wrap = True
    tf_mc3.margin_left = tf_mc3.margin_right = Inches(0.2)
    tf_mc3.margin_top = Inches(0.15)
    p = tf_mc3.paragraphs[0]
    add_run(p, "LỢI ÍCH VẬN HÀNH & KIỂM THỬ CI/CD\n", name=FONT_HEADING, size=Pt(10.5), bold=True, color=COLOR_NAVY)

    pts_mc3 = [
        "Kiểm thử tự động: Chỉ 1 lệnh `pytest tests/test_data_contracts.py` để verify tương thích 100%.",
        "Tích hợp không đau đớn: Khi dữ liệu thật về, thay thế mock fixtures mà không sửa 1 dòng code logic nào.",
        "Đúng tiến độ môn học: Đảm bảo toàn nhóm về đích trước hạn bảo vệ đề tài với chất lượng cao nhất."
    ]
    for pt in pts_mc3:
        p_pt = tf_mc3.add_paragraph()
        p_pt.space_before = Pt(3)
        add_run(p_pt, f"★ {pt}", size=Pt(8.5), color=COLOR_CHARCOAL)

    # Save presentation
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    prs.save(output_path)
    print(f"[SUCCESS] Successfully created pitch deck at: {output_path}")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_pptx = os.path.join(current_dir, "Team3_FakeNews_Proposal_PitchDeck.pptx")
    create_deck(output_pptx)
