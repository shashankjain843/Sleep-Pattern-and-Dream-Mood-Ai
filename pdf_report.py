"""
pdf_report.py — Generate a clean, styled PDF report using ReportLab.
Falls back gracefully (returns None) if ReportLab is not installed.
"""
import logging
from io import BytesIO
from typing import Optional

logger = logging.getLogger(__name__)


def generate_report_pdf(report: dict) -> Optional[bytes]:
    """
    Accept the same dict that /generate_report returns and produce a PDF.
    Returns raw bytes on success, None if ReportLab is unavailable.
    """
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, HRFlowable,
            Table, TableStyle, ListFlowable, ListItem,
        )
    except ImportError:
        logger.warning("ReportLab not installed — PDF export unavailable")
        return None

    buf = BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        rightMargin=2 * cm,
        leftMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
    )

    # ── Colour palette (matches neon theme) ──────────────────────────────────
    NEON_TEAL  = colors.HexColor("#00ffcc")
    DARK_BG    = colors.HexColor("#111827")
    LIGHT_TEXT = colors.HexColor("#e0e0e0")
    ACCENT     = colors.HexColor("#0099cc")
    WARNING_C  = colors.HexColor("#f59e0b")

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        "Title",
        parent=styles["Title"],
        fontSize=22,
        textColor=NEON_TEAL,
        spaceAfter=6,
        fontName="Helvetica-Bold",
    )
    subtitle_style = ParagraphStyle(
        "Subtitle",
        parent=styles["Normal"],
        fontSize=11,
        textColor=colors.HexColor("#aaaaaa"),
        spaceAfter=14,
    )
    section_style = ParagraphStyle(
        "Section",
        parent=styles["Heading2"],
        fontSize=13,
        textColor=ACCENT,
        spaceBefore=14,
        spaceAfter=6,
        fontName="Helvetica-Bold",
    )
    body_style = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontSize=10,
        textColor=colors.HexColor("#333333"),
        spaceAfter=8,
        leading=14,
    )
    suggestion_style = ParagraphStyle(
        "Suggestion",
        parent=styles["Normal"],
        fontSize=10,
        textColor=colors.HexColor("#222222"),
        leftIndent=12,
        spaceAfter=4,
        leading=13,
    )
    warning_style = ParagraphStyle(
        "Warning",
        parent=styles["Normal"],
        fontSize=10,
        textColor=WARNING_C,
        leftIndent=12,
        spaceAfter=4,
        leading=13,
    )

    story = []

    # ── Header ───────────────────────────────────────────────────────────────
    story.append(Paragraph("🌌 Sleep Pattern & Dream Mood AI", title_style))
    story.append(Paragraph(report.get("title", "Daily Health & Wellness Report"), subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1, color=NEON_TEAL))
    story.append(Spacer(1, 10))

    # Summary
    if report.get("summary"):
        story.append(Paragraph(report["summary"], body_style))

    # ── Sleep summary ─────────────────────────────────────────────────────────
    story.append(Paragraph("💤 Sleep Analysis", section_style))
    sleep_text = report.get("sleep_summary", "").replace("**", "")
    story.append(Paragraph(sleep_text, body_style))

    # ── Mood summary ──────────────────────────────────────────────────────────
    story.append(Paragraph("💭 Dream Mood Analysis", section_style))
    mood_text = report.get("mood_summary", "").replace("**", "")
    story.append(Paragraph(mood_text, body_style))

    # ── Combined insights ─────────────────────────────────────────────────────
    story.append(Paragraph("🔗 Combined Insights", section_style))
    story.append(Paragraph(report.get("combined_insights", ""), body_style))

    # ── Narrative (Claude-generated if present) ───────────────────────────────
    if report.get("narrative"):
        story.append(Paragraph("🤖 AI Narrative", section_style))
        story.append(Paragraph(report["narrative"], body_style))

    # ── Suggestions ───────────────────────────────────────────────────────────
    story.append(Paragraph("💡 Personalised Recommendations", section_style))
    for s in report.get("suggestions_list", []):
        clean = s.replace("**", "")
        story.append(Paragraph(f"• {clean}", suggestion_style))

    # ── Warnings ─────────────────────────────────────────────────────────────
    warnings = report.get("warnings", [])
    if warnings:
        story.append(Spacer(1, 8))
        story.append(Paragraph("⚠️ Health Alerts", section_style))
        for w in warnings:
            story.append(Paragraph(f"▲ {w.replace('⚠️', '').strip()}", warning_style))

    # ── Footer ────────────────────────────────────────────────────────────────
    story.append(Spacer(1, 20))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#cccccc")))
    story.append(
        Paragraph(
            "Generated by Sleep Pattern & Dream Mood AI  |  For informational purposes only",
            ParagraphStyle("Footer", parent=styles["Normal"], fontSize=8,
                           textColor=colors.grey, alignment=1),
        )
    )

    doc.build(story)
    return buf.getvalue()
