# ============================================================
# pdf_export.py — PDF Report Generator
# Save this file in the SAME folder as app.py
#
# HOW IT WORKS:
#   1. Takes the DataFrame + col_types + AI summary as input
#   2. Builds KPI table using reportlab Table
#   3. Renders charts as PNG images using matplotlib
#      (matplotlib replicates what Plotly shows, without kaleido)
#   4. Embeds everything into a professional A4 PDF
#   5. Returns PDF as bytes → Streamlit download button
#
# WHY MATPLOTLIB INSTEAD OF PLOTLY FOR CHARTS IN PDF?
#   Plotly needs kaleido to export images — kaleido doesn't install
#   easily on all platforms. Matplotlib is pre-installed everywhere
#   and produces clean static charts for PDF embedding.
#   In an interview: "I chose matplotlib for PDF charts because it's
#   a universal dependency, while Plotly is used for the interactive
#   web charts — right tool for each job."
# ============================================================

import io
import pandas as pd
import matplotlib
matplotlib.use('Agg')          # Non-interactive backend — no display window
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime

from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table,
    TableStyle, Image, HRFlowable, PageBreak
)
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT


# ── COLOUR PALETTE ──────────────────────────────────────────
DARK_BLUE   = colors.HexColor('#1e3a5f')
MID_BLUE    = colors.HexColor('#2563eb')
LIGHT_BLUE  = colors.HexColor('#dbeafe')
GREEN       = colors.HexColor('#166534')
LIGHT_GREEN = colors.HexColor('#dcfce7')
GREY_LIGHT  = colors.HexColor('#f9fafb')
GREY_MID    = colors.HexColor('#e5e7eb')
GREY_DARK   = colors.HexColor('#6b7280')
WHITE       = colors.white
BLACK       = colors.HexColor('#111827')


def format_number(val):
    """Format large numbers into readable Indian format."""
    try:
        val = float(val)
        if abs(val) >= 1e7:
            return f"{val/1e7:.2f} Cr"
        elif abs(val) >= 1e5:
            return f"{val/1e5:.1f} L"
        elif abs(val) >= 1e3:
            return f"{val/1e3:.1f} K"
        else:
            return f"{val:,.2f}"
    except:
        return str(val)


def make_styles():
    """Build all paragraph styles used in the report."""
    base = getSampleStyleSheet()

    styles = {
        'title': ParagraphStyle(
            'ReportTitle',
            fontSize=22,
            fontName='Helvetica-Bold',
            textColor=DARK_BLUE,
            spaceAfter=6,
            alignment=TA_CENTER,
        ),
        'subtitle': ParagraphStyle(
            'Subtitle',
            fontSize=11,
            fontName='Helvetica',
            textColor=GREY_DARK,
            spaceAfter=4,
            alignment=TA_CENTER,
        ),
        'section': ParagraphStyle(
            'Section',
            fontSize=13,
            fontName='Helvetica-Bold',
            textColor=DARK_BLUE,
            spaceBefore=14,
            spaceAfter=6,
            borderPad=4,
        ),
        'body': ParagraphStyle(
            'Body',
            fontSize=10,
            fontName='Helvetica',
            textColor=BLACK,
            spaceAfter=4,
            leading=16,
        ),
        'insight': ParagraphStyle(
            'Insight',
            fontSize=10,
            fontName='Helvetica',
            textColor=GREEN,
            spaceAfter=4,
            leading=16,
            leftIndent=10,
        ),
        'footer': ParagraphStyle(
            'Footer',
            fontSize=8,
            fontName='Helvetica',
            textColor=GREY_DARK,
            alignment=TA_CENTER,
        ),
        'kpi_label': ParagraphStyle(
            'KPILabel',
            fontSize=9,
            fontName='Helvetica',
            textColor=GREY_DARK,
            alignment=TA_CENTER,
        ),
        'kpi_value': ParagraphStyle(
            'KPIValue',
            fontSize=14,
            fontName='Helvetica-Bold',
            textColor=DARK_BLUE,
            alignment=TA_CENTER,
        ),
    }
    return styles


# ── CHART GENERATORS ────────────────────────────────────────
# Each function creates a matplotlib chart and returns it
# as an in-memory PNG bytes object for embedding in the PDF

def chart_bar(df, x_col, y_col, title, color='#2563eb'):
    """Bar chart — grouped comparison."""
    agg = (
        df.groupby(x_col)[y_col]
        .sum()
        .reset_index()
        .sort_values(y_col, ascending=False)
        .head(10)
    )
    fig, ax = plt.subplots(figsize=(7, 3.2))
    bars = ax.bar(
        agg[x_col].astype(str),
        agg[y_col],
        color=color,
        edgecolor='white',
        linewidth=0.5
    )
    # Value labels on top of bars
    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h * 1.01,
            format_number(h),
            ha='center', va='bottom',
            fontsize=7, color='#374151'
        )
    ax.set_title(title, fontsize=11, fontweight='bold', color='#1e3a5f', pad=8)
    ax.set_xlabel(x_col, fontsize=9, color='#6b7280')
    ax.set_ylabel(y_col, fontsize=9, color='#6b7280')
    ax.tick_params(axis='x', rotation=30, labelsize=8)
    ax.tick_params(axis='y', labelsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_facecolor('#f9fafb')
    fig.patch.set_facecolor('white')
    plt.tight_layout()
    return fig_to_bytes(fig)


def chart_line(df, date_col, y_col, title):
    """Line chart — trend over time."""
    df_copy = df.copy()
    df_copy['_period'] = df_copy[date_col].dt.to_period('M').astype(str)
    trend = df_copy.groupby('_period')[y_col].sum().reset_index()

    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.plot(
        trend['_period'], trend[y_col],
        color='#6366f1', linewidth=2,
        marker='o', markersize=4, markerfacecolor='white',
        markeredgecolor='#6366f1', markeredgewidth=1.5
    )
    ax.fill_between(
        trend['_period'], trend[y_col],
        alpha=0.08, color='#6366f1'
    )
    ax.set_title(title, fontsize=11, fontweight='bold', color='#1e3a5f', pad=8)
    ax.set_xlabel('Period', fontsize=9, color='#6b7280')
    ax.set_ylabel(y_col, fontsize=9, color='#6b7280')
    ax.tick_params(axis='x', rotation=45, labelsize=7)
    ax.tick_params(axis='y', labelsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_facecolor('#f9fafb')
    fig.patch.set_facecolor('white')

    # Show every 3rd label to avoid overlap
    labels = trend['_period'].tolist()
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(
        [l if i % 3 == 0 else '' for i, l in enumerate(labels)]
    )
    plt.tight_layout()
    return fig_to_bytes(fig)


def chart_donut(df, cat_col, num_col, title):
    """Donut chart — proportional share."""
    data = (
        df.groupby(cat_col)[num_col]
        .sum()
        .reset_index()
        .sort_values(num_col, ascending=False)
        .head(8)
    )
    colors_list = [
        '#2563eb','#16a34a','#ea580c','#9333ea',
        '#0891b2','#db2777','#84cc16','#f59e0b'
    ]
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    wedges, texts, autotexts = ax.pie(
        data[num_col],
        labels=None,
        autopct='%1.1f%%',
        colors=colors_list[:len(data)],
        pctdistance=0.75,
        startangle=90,
        wedgeprops={'linewidth': 2, 'edgecolor': 'white'}
    )
    for at in autotexts:
        at.set_fontsize(7)
        at.set_color('white')
        at.set_fontweight('bold')

    # Donut hole
    centre = plt.Circle((0, 0), 0.5, fc='white')
    ax.add_patch(centre)

    # Legend on right
    ax.legend(
        wedges,
        data[cat_col].astype(str).tolist(),
        loc='center left',
        bbox_to_anchor=(1, 0, 0.4, 1),
        fontsize=7,
        frameon=False
    )
    ax.set_title(title, fontsize=11, fontweight='bold', color='#1e3a5f', pad=8)
    fig.patch.set_facecolor('white')
    plt.tight_layout()
    return fig_to_bytes(fig)


def chart_scatter(df, cat_col, x_col, y_col, title):
    """Bubble scatter — relationship between two metrics."""
    data = df.groupby(cat_col).agg(
        x=(x_col, 'sum'),
        y=(y_col, 'sum')
    ).reset_index()

    fig, ax = plt.subplots(figsize=(6, 3.2))
    scatter = ax.scatter(
        data['x'], data['y'],
        s=80, alpha=0.7,
        c=data['y'], cmap='RdYlGn',
        edgecolors='white', linewidth=0.5
    )
    # Label top 5 points
    for _, row in data.nlargest(5, 'x').iterrows():
        ax.annotate(
            str(row[cat_col])[:10],
            (row['x'], row['y']),
            fontsize=6.5, color='#374151',
            xytext=(4, 4), textcoords='offset points'
        )
    ax.set_title(title, fontsize=11, fontweight='bold', color='#1e3a5f', pad=8)
    ax.set_xlabel(x_col, fontsize=9, color='#6b7280')
    ax.set_ylabel(y_col, fontsize=9, color='#6b7280')
    ax.tick_params(labelsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_facecolor('#f9fafb')
    plt.colorbar(scatter, ax=ax, shrink=0.8, label=y_col)
    fig.patch.set_facecolor('white')
    plt.tight_layout()
    return fig_to_bytes(fig)


def fig_to_bytes(fig):
    """Converts a matplotlib figure to PNG bytes and closes the figure."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)   # IMPORTANT: always close to free memory
    return buf


# ── MAIN PDF BUILDER ────────────────────────────────────────
def generate_pdf_report(df, col_types, ai_summary, filename="analysis_report"):
    """
    Main function called from app.py.
    Returns PDF as bytes — plug into st.download_button().

    Parameters:
        df          : filtered DataFrame (what user sees on screen)
        col_types   : dict from detect_columns()
        ai_summary  : string — the AI executive summary text
        filename    : base name for the downloaded file

    Returns:
        bytes — the complete PDF file
    """
    buf = io.BytesIO()

    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        rightMargin=1.8*cm,
        leftMargin=1.8*cm,
        topMargin=2*cm,
        bottomMargin=2*cm,
    )

    styles  = make_styles()
    story   = []         # list of reportlab elements to render
    W       = A4[0] - 3.6*cm   # usable page width

    num_cols = col_types.get('numeric', [])
    cat_cols = col_types.get('categorical', [])
    date_cols= col_types.get('datetime', [])

    # Identify primary columns (same logic as app.py)
    pure_numeric = [c for c in num_cols if c not in cat_cols]
    chart_numeric = pure_numeric if pure_numeric else num_cols
    primary_num = (
        max(chart_numeric, key=lambda c: df[c].nunique())
        if chart_numeric else None
    )
    second_num = (
        [c for c in chart_numeric if c != primary_num][0]
        if len(chart_numeric) > 1 else None
    )
    text_cats  = [c for c in cat_cols if df[c].dtype == object]
    num_cats   = [c for c in cat_cols if c not in text_cats]
    all_cats   = text_cats + num_cats
    primary_cat = all_cats[0] if all_cats else None
    second_cat  = all_cats[1] if len(all_cats) > 1 else None
    date_col    = date_cols[0] if date_cols else None

    generated_at = datetime.now().strftime("%d %B %Y, %I:%M %p")

    # ── PAGE 1: HEADER ──────────────────────────────────────
    story.append(Spacer(1, 0.3*cm))

    # Title block with background
    title_data = [[
        Paragraph("AI DATA ANALYST REPORT", styles['title'])
    ]]
    title_table = Table(title_data, colWidths=[W])
    title_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), LIGHT_BLUE),
        ('ROUNDEDCORNERS', [8]),
        ('TOPPADDING', (0,0), (-1,-1), 14),
        ('BOTTOMPADDING', (0,0), (-1,-1), 14),
        ('LEFTPADDING', (0,0), (-1,-1), 12),
        ('RIGHTPADDING', (0,0), (-1,-1), 12),
    ]))
    story.append(title_table)
    story.append(Spacer(1, 0.2*cm))
    story.append(Paragraph(
        f"Generated on {generated_at} | "
        f"{len(df):,} rows × {len(df.columns)} columns",
        styles['subtitle']
    ))
    story.append(HRFlowable(
        width=W, thickness=1.5,
        color=MID_BLUE, spaceAfter=10
    ))

    # ── DATASET OVERVIEW TABLE ───────────────────────────────
    story.append(Paragraph("Dataset Overview", styles['section']))

    overview_data = [
        ['Metric', 'Value'],
        ['Total Rows', f"{len(df):,}"],
        ['Total Columns', str(len(df.columns))],
        ['Numeric Columns', str(len(num_cols))],
        ['Category Columns', str(len(cat_cols))],
        ['Date Columns', str(len(date_cols))],
        ['Missing Values', f"{df.isnull().sum().sum():,}"],
    ]
    if date_col:
        try:
            overview_data.append([
                'Date Range',
                f"{df[date_col].min().date()} to {df[date_col].max().date()}"
            ])
        except:
            pass

    ov_table = Table(overview_data, colWidths=[W*0.45, W*0.55])
    ov_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), DARK_BLUE),
        ('TEXTCOLOR', (0,0), (-1,0), WHITE),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 9),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [GREY_LIGHT, WHITE]),
        ('GRID', (0,0), (-1,-1), 0.3, GREY_MID),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(ov_table)
    story.append(Spacer(1, 0.3*cm))

    # ── KPI CARDS TABLE ─────────────────────────────────────
    if num_cols:
        story.append(Paragraph("Key Performance Indicators", styles['section']))

        show_cols = num_cols[:4]
        kpi_cells = []
        for col in show_cols:
            total = df[col].sum()
            avg   = df[col].mean()
            kpi_cells.append([
                Paragraph(col, styles['kpi_label']),
                Paragraph(format_number(total), styles['kpi_value']),
                Paragraph(f"Avg: {format_number(avg)}", styles['kpi_label']),
            ])

        # Arrange KPIs in a 2-column grid
        kpi_rows = []
        for i in range(0, len(kpi_cells), 2):
            row_left  = kpi_cells[i]
            row_right = kpi_cells[i+1] if i+1 < len(kpi_cells) else ['','','']

            # Build 2-cell row with a divider column in between
            kpi_rows.append([
                Table([row_left],  colWidths=[W*0.45]),
                Spacer(0.1*cm, 0),
                Table([row_right], colWidths=[W*0.45]),
            ])

        for kpi_row in kpi_rows:
            row_table = Table(
                [kpi_row],
                colWidths=[W*0.47, W*0.06, W*0.47]
            )
            row_table.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (0,0), LIGHT_BLUE),
                ('BACKGROUND', (2,0), (2,0), LIGHT_BLUE),
                ('ROUNDEDCORNERS', [6]),
                ('TOPPADDING', (0,0), (-1,-1), 8),
                ('BOTTOMPADDING', (0,0), (-1,-1), 8),
                ('LEFTPADDING', (0,0), (-1,-1), 6),
                ('RIGHTPADDING', (0,0), (-1,-1), 6),
            ]))
            story.append(row_table)
            story.append(Spacer(1, 0.2*cm))

    # ── AI EXECUTIVE SUMMARY ────────────────────────────────
    story.append(Paragraph("AI Executive Summary", styles['section']))

    # Clean the AI text — remove markdown symbols if any
    clean_summary = (ai_summary or "No AI summary available.")
    clean_summary = clean_summary.replace('**', '').replace('*', '').replace('#','')

    summary_data = [[Paragraph(clean_summary, styles['insight'])]]
    summary_table = Table(summary_data, colWidths=[W])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), LIGHT_GREEN),
        ('LEFTPADDING', (0,0), (-1,-1), 14),
        ('RIGHTPADDING', (0,0), (-1,-1), 14),
        ('TOPPADDING', (0,0), (-1,-1), 10),
        ('BOTTOMPADDING', (0,0), (-1,-1), 10),
        ('LINEAFTER', (0,0), (0,-1), 3, colors.HexColor('#22c55e')),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 0.3*cm))

    # ── NUMERIC STATS TABLE ─────────────────────────────────
    if num_cols:
        story.append(Paragraph("Numeric Column Statistics", styles['section']))

        stat_header = ['Column', 'Total', 'Average', 'Min', 'Max', 'Nulls']
        stat_rows   = [stat_header]
        for col in num_cols[:8]:   # max 8 rows
            stat_rows.append([
                col,
                format_number(df[col].sum()),
                format_number(df[col].mean()),
                format_number(df[col].min()),
                format_number(df[col].max()),
                str(df[col].isnull().sum()),
            ])

        col_w = W / len(stat_header)
        stat_table = Table(stat_rows, colWidths=[col_w]*len(stat_header))
        stat_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), DARK_BLUE),
            ('TEXTCOLOR', (0,0), (-1,0), WHITE),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE', (0,0), (-1,-1), 8),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [GREY_LIGHT, WHITE]),
            ('GRID', (0,0), (-1,-1), 0.3, GREY_MID),
            ('ALIGN', (1,0), (-1,-1), 'RIGHT'),
            ('TOPPADDING', (0,0), (-1,-1), 4),
            ('BOTTOMPADDING', (0,0), (-1,-1), 4),
            ('LEFTPADDING', (0,0), (-1,-1), 6),
        ]))
        story.append(stat_table)

    # ── PAGE 2: CHARTS ───────────────────────────────────────
    story.append(PageBreak())
    story.append(Paragraph("Visual Analysis", styles['section']))
    story.append(HRFlowable(
        width=W, thickness=1,
        color=GREY_MID, spaceAfter=8
    ))

    charts_added = 0

    # Chart 1 — Bar: primary numeric by primary category
    if primary_cat and primary_num:
        try:
            img_bytes = chart_bar(
                df, primary_cat, primary_num,
                f"{primary_num} by {primary_cat}"
            )
            story.append(Image(img_bytes, width=W, height=8*cm))
            story.append(Spacer(1, 0.3*cm))
            charts_added += 1
        except Exception as e:
            story.append(Paragraph(f"Chart unavailable: {e}", styles['body']))

    # Chart 2 — Line: trend over time (if date exists)
    if date_col and primary_num:
        try:
            img_bytes = chart_line(
                df, date_col, primary_num,
                f"{primary_num} Trend Over Time"
            )
            story.append(Image(img_bytes, width=W, height=8*cm))
            story.append(Spacer(1, 0.3*cm))
            charts_added += 1
        except Exception as e:
            story.append(Paragraph(f"Chart unavailable: {e}", styles['body']))

    # Page break before next 2 charts if we already have 2
    if charts_added >= 2:
        story.append(PageBreak())
        story.append(Paragraph("Visual Analysis (continued)", styles['section']))
        story.append(HRFlowable(
            width=W, thickness=1,
            color=GREY_MID, spaceAfter=8
        ))

    # Chart 3 — Donut: category share
    if primary_cat and primary_num:
        try:
            img_bytes = chart_donut(
                df, primary_cat, primary_num,
                f"{primary_num} Share by {primary_cat}"
            )
            story.append(Image(img_bytes, width=W*0.75, height=8*cm))
            story.append(Spacer(1, 0.3*cm))
            charts_added += 1
        except Exception as e:
            story.append(Paragraph(f"Chart unavailable: {e}", styles['body']))

    # Chart 4 — Scatter: two numeric columns
    if primary_cat and primary_num and second_num:
        try:
            img_bytes = chart_scatter(
                df, primary_cat, primary_num, second_num,
                f"{primary_num} vs {second_num}"
            )
            story.append(Image(img_bytes, width=W, height=8*cm))
            charts_added += 1
        except Exception as e:
            story.append(Paragraph(f"Chart unavailable: {e}", styles['body']))

    # Fallback — if no charts rendered
    if charts_added == 0:
        story.append(Paragraph(
            "No charts could be generated for this dataset. "
            "Add categorical or date columns to enable chart exports.",
            styles['body']
        ))

    # ── PAGE 3: DATA SAMPLE ─────────────────────────────────
    story.append(PageBreak())
    story.append(Paragraph("Data Sample (First 20 Rows)", styles['section']))
    story.append(HRFlowable(
        width=W, thickness=1,
        color=GREY_MID, spaceAfter=8
    ))

    # Show max 7 columns so table fits on page
    sample_df   = df.head(20)
    show_cols_s = list(df.columns[:7])
    sample_df   = sample_df[show_cols_s]

    # Build table data
    table_data  = [show_cols_s]   # header row
    for _, row in sample_df.iterrows():
        table_data.append([str(v)[:18] for v in row.values])

    col_w_s = W / len(show_cols_s)
    data_table = Table(table_data, colWidths=[col_w_s]*len(show_cols_s))
    data_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), DARK_BLUE),
        ('TEXTCOLOR', (0,0), (-1,0), WHITE),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 7),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [GREY_LIGHT, WHITE]),
        ('GRID', (0,0), (-1,-1), 0.3, GREY_MID),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('LEFTPADDING', (0,0), (-1,-1), 4),
        ('WORDWRAP', (0,0), (-1,-1), True),
    ]))
    story.append(data_table)
    story.append(Spacer(1, 0.5*cm))

    # ── FOOTER ──────────────────────────────────────────────
    story.append(HRFlowable(
        width=W, thickness=0.5,
        color=GREY_MID, spaceBefore=10
    ))
    story.append(Paragraph(
        f"Report generated by AI Data Analyst Dashboard | "
        f"Powered by Groq LLaMA 3.3 70B | {generated_at}",
        styles['footer']
    ))

    # ── BUILD PDF ────────────────────────────────────────────
    doc.build(story)
    buf.seek(0)
    return buf.read()   # return raw bytes