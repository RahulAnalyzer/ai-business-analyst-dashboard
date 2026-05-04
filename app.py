# ============================================================
# AI DATA ANALYST DASHBOARD — FULLY GENERIC VERSION
# Works on ANY CSV file — no hardcoded column names
# AI powered by Groq (free, 14,400 req/day, no credit card)
#
# HOW GENERIC MODE WORKS:
#   1. Upload any CSV
#   2. Auto-detect numeric / categorical / date columns
#   3. Build KPIs, charts, filters from whatever columns exist
#   4. AI understands the data context and answers questions
# ============================================================

import streamlit as st
import pandas as pd
import plotly.express as px
from dotenv import load_dotenv
import os
from groq import Groq

# PDF export module — save pdf_export.py in same folder as app.py
try:
    from pdf_export import generate_pdf_report
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

# ── SETUP ──────────────────────────────────────────────────
load_dotenv()
client = Groq(
    api_key=os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", "")
)
MODEL = "llama-3.3-70b-versatile"

st.set_page_config(
    page_title="AI Data Analyst",
    page_icon="📊",
    layout="wide"
)

st.markdown("""
<style>
    .insight-box {
        background: #f0fdf4;
        border-left: 4px solid #22c55e;
        padding: 1rem 1.2rem;
        border-radius: 0 8px 8px 0;
        margin: 0.5rem 0 1rem 0;
        font-size: 0.93rem;
        color: #166534;
        line-height: 1.8;
    }
    .section-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1f2937;
        padding-bottom: 0.4rem;
        border-bottom: 2px solid #e5e7eb;
        margin-bottom: 1rem;
    }
    .chat-user {
        background: #eff6ff;
        border: 1px solid #bfdbfe;
        border-radius: 12px 12px 2px 12px;
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
        font-size: 0.92rem;
        color: #1e3a5f;
        line-height: 1.6;
    }
    .chat-ai {
        background: #f9fafb;
        border: 1px solid #e5e7eb;
        border-radius: 12px 12px 12px 2px;
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
        font-size: 0.92rem;
        color: #1f2937;
        line-height: 1.8;
    }
    .kpi-note {
        font-size: 0.75rem;
        color: #9ca3af;
        margin-top: 2px;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# GROQ AI CALL — with caching to save quota
# @st.cache_data: same prompt → returns cached result instantly
# ttl=600 = cache expires after 10 minutes
# ============================================================
@st.cache_data(ttl=600, show_spinner=False)
def ask_ai(prompt):
    """
    Calls Groq LLaMA model and returns response text.
    Cached so the same prompt never hits the API twice.
    """
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400,
            temperature=0.3   # lower = more consistent, factual responses
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"AI unavailable: {str(e)}"


# ============================================================
# COLUMN TYPE DETECTION — the core of generic mode
#
# We scan every column and classify it into one of 4 types:
#   numeric     → float/int columns  (Sales, Age, Score...)
#   categorical → text with few unique values (Region, Status...)
#   datetime    → date/time columns  (Order_Date, Created_At...)
#   text        → free text (Name, Description...) — skip for charts
#
# KEY FIX from previous version:
#   We try date parsing STRICTLY — only columns where MOST values
#   parse successfully are marked as dates. This prevents columns
#   like "Product Name" from being misclassified as dates.
# ============================================================
def detect_columns(df):
    """
    Smarter column detection that handles any CSV type:
    - Medical data (MIMIC), Sales data, HR data, Finance data etc.
    
    KEY FIXES vs old version:
    1. Numeric columns with FEW unique values → also treated as categorical
       (e.g. age groups 0-10, score 1-5, rating 1-3)
    2. Text columns are checked more generously for categorical
       (relaxed from ratio<0.5 to ratio<0.8)
    3. Numeric columns that look like IDs (patient_id, order_id) are skipped
    """
    cols = {
        'numeric'    : [],
        'categorical': [],
        'datetime'   : [],
        'text'       : [],
    }

    total_rows = len(df)

    for col in df.columns:
        series = df[col].dropna()
        if len(series) == 0:
            continue

        col_lower = col.lower()

        # ── SKIP ID-like columns ──────────────────────────
        # Columns named like "id", "patient_id", "order_id" are identifiers
        # They're not useful for analysis or filters
        if any(id_kw in col_lower for id_kw in
               ['_id', 'id_', ' id', 'index', 'row_num']):
            cols['text'].append(col)
            continue

        # ── DATETIME DETECTION ────────────────────────────
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            cols['datetime'].append(col)
            continue

        if df[col].dtype == object:
            try:
                parsed = pd.to_datetime(
                    series, infer_datetime_format=True, errors='coerce'
                )
                if parsed.notna().mean() >= 0.8:
                    cols['datetime'].append(col)
                    continue
            except:
                pass

        # ── NUMERIC COLUMNS ───────────────────────────────
        if pd.api.types.is_numeric_dtype(df[col]):
            unique_count = series.nunique()

            # Numeric with very few unique values = likely a category
            # Examples: gender (0/1), rating (1-5), grade (1-4)
            # Threshold: if fewer than 15 unique values → categorical too
            if unique_count <= 15:
                cols['categorical'].append(col)
                # Also keep in numeric for KPI calculations
                cols['numeric'].append(col)
            else:
                cols['numeric'].append(col)
            continue

        # ── OBJECT COLUMNS → CATEGORICAL or TEXT ─────────
        if df[col].dtype == object:
            unique_count = series.nunique()
            # ratio = how unique is this column (0=all same, 1=all unique)
            ratio = unique_count / total_rows

            # GENEROUS categorical check:
            # - Fewer than 100 unique values, OR
            # - Less than 80% unique (lots of repeats = category)
            if unique_count <= 100 or ratio < 0.8:
                cols['categorical'].append(col)
            else:
                cols['text'].append(col)

    # Remove duplicates while preserving order
    for key in cols:
        seen = set()
        cols[key] = [x for x in cols[key]
                     if not (x in seen or seen.add(x))]

    return cols


# ============================================================
# LOAD DATA — reads any CSV, auto-converts date columns
# ============================================================
def load_data(uploaded_file):
    df = pd.read_csv(uploaded_file)

    # Auto-convert columns that look like dates
    # We use errors='coerce' so invalid dates become NaT, not errors
    for col in df.columns:
        if df[col].dtype == object:
            try:
                parsed = pd.to_datetime(
                    df[col], infer_datetime_format=True, errors='coerce'
                )
                # Only convert if 80%+ of values successfully parsed
                if parsed.notna().mean() >= 0.8:
                    df[col] = parsed
            except:
                pass

    return df


# ============================================================
# DATA PROFILE — text description passed to AI as context
# AI cannot see the CSV directly — we describe it in words
# ============================================================
def get_data_profile(df, col_types):
    lines = []
    lines.append(f"Rows: {len(df):,} | Columns: {len(df.columns)}")
    lines.append(f"All columns: {', '.join(df.columns.tolist())}")

    if col_types['datetime']:
        dc = col_types['datetime'][0]
        try:
            lines.append(f"Date range ({dc}): {df[dc].min().date()} to {df[dc].max().date()}")
        except:
            pass

    if col_types['numeric']:
        lines.append("\nNUMERIC COLUMNS:")
        for c in col_types['numeric'][:6]:
            lines.append(
                f"  {c}: total={df[c].sum():,.1f}, "
                f"avg={df[c].mean():,.1f}, "
                f"min={df[c].min():,.1f}, max={df[c].max():,.1f}"
            )

    if col_types['categorical']:
        lines.append("\nCATEGORY COLUMNS:")
        for c in col_types['categorical'][:5]:
            top = df[c].value_counts().index[0]
            lines.append(f"  {c}: {df[c].nunique()} unique values, top='{top}'")

    return "\n".join(lines)


# ============================================================
# FORMAT LARGE NUMBERS — makes KPIs readable
# ============================================================
def format_number(val):
    if abs(val) >= 1e7:
        return f"{val/1e7:.2f} Cr"
    elif abs(val) >= 1e5:
        return f"{val/1e5:.1f} L"
    elif abs(val) >= 1e3:
        return f"{val/1e3:.1f} K"
    else:
        return f"{val:,.2f}"


# ============================================================
# KPI CARDS — top numeric columns shown as metric cards
# GENERIC: shows whatever numeric columns exist (up to 4)
# ============================================================
def generate_kpi_cards(df, col_types):
    st.markdown('<p class="section-title">📈 Key Metrics</p>',
                unsafe_allow_html=True)

    numeric_cols = col_types['numeric']
    if not numeric_cols:
        st.info("No numeric columns found in this dataset.")
        return

    # Show up to 4 numeric columns as KPI cards
    show_cols = numeric_cols[:4]
    cols      = st.columns(len(show_cols))

    for i, col in enumerate(show_cols):
        total = df[col].sum()
        avg   = df[col].mean()
        mx    = df[col].max()
        with cols[i]:
            st.metric(
                label=f"Total {col}",
                value=format_number(total),
                help=f"Avg: {avg:,.2f} | Max: {mx:,.2f}"
            )
            st.markdown(
                f'<p class="kpi-note">Avg: {format_number(avg)} | '
                f'Max: {format_number(mx)}</p>',
                unsafe_allow_html=True
            )


# ============================================================
# AI EXECUTIVE SUMMARY
# Pandas computes all numbers → AI writes the narrative
# ============================================================
def generate_ai_summary(df, col_types):
    st.markdown('<p class="section-title">🤖 AI Executive Summary</p>',
                unsafe_allow_html=True)

    with st.spinner("AI is analyzing your data..."):
        profile = get_data_profile(df, col_types)
        sample  = df.head(3).to_string(index=False)

        prompt = f"""You are a senior data analyst. A user uploaded a CSV dataset.

DATASET PROFILE:
{profile}

SAMPLE DATA (first 3 rows):
{sample}

Write a professional executive summary in 4-5 sentences:
1. Identify what type of data this is
2. State the most important top-line finding with actual numbers
3. Highlight one strong performer by name
4. Flag one concern or pattern worth investigating
5. Give one specific actionable recommendation

Rules: Flowing paragraph, NO bullet points. Under 120 words.
Be specific — use actual column names and values from the data."""

        ai_text = ask_ai(prompt)
        # Save to session state so PDF export can use it without extra API call
        st.session_state['last_ai_summary'] = ai_text
        st.markdown(
            f'<div class="insight-box">{ai_text}</div>',
            unsafe_allow_html=True
        )


# ============================================================
# CHARTS — 4 business charts built dynamically
#
# GENERIC LOGIC:
#   Chart 1 → Primary numeric by primary category   (Bar)
#   Chart 2 → Primary numeric over time             (Line)
#             OR primary numeric by second category  (Bar)
#   Chart 3 → Category share of primary numeric     (Donut)
#   Chart 4 → Two numerics relationship             (Scatter)
#             OR second numeric by category          (Bar)
#
# If a chart can't be built (column doesn't exist),
# we show a helpful message instead of crashing.
# ============================================================
def show_charts(df, col_types):
    st.markdown('<p class="section-title">📊 Visual Analysis</p>',
                unsafe_allow_html=True)

    num_cols  = col_types['numeric']
    cat_cols  = col_types['categorical']
    date_cols = col_types['datetime']

    # Need at least one numeric column to show charts
    if not num_cols:
        st.warning("No numeric columns detected — cannot generate charts.")
        return

    # Pick best columns for visualization
    # primary_num = numeric col with most unique values (most interesting to chart)
    # Excludes cols that are ALSO categorical (those are better as filters)
    pure_numeric = [c for c in num_cols if c not in cat_cols]
    chart_numeric = pure_numeric if pure_numeric else num_cols

    primary_num = max(chart_numeric, key=lambda c: df[c].nunique()) \
                  if chart_numeric else None
    second_num  = [c for c in chart_numeric if c != primary_num][0] \
                  if len(chart_numeric) > 1 else None

    # For categories: prefer text-type categoricals over numeric-turned-categorical
    text_cats = [c for c in cat_cols if df[c].dtype == object]
    num_cats  = [c for c in cat_cols if c not in text_cats]

    all_cats   = text_cats + num_cats   # text categories first
    primary_cat = all_cats[0] if all_cats else None
    second_cat  = all_cats[1] if len(all_cats) > 1 else None
    date_col    = date_cols[0] if date_cols else None

    # ── ROW 1 ──────────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        # CHART 1: Primary numeric by primary category — Bar Chart
        # "Which category has the most?"
        if primary_cat and primary_num:
            agg = (
                df.groupby(primary_cat)[primary_num]
                .sum()
                .reset_index()
                .sort_values(primary_num, ascending=False)
                .head(10)
            )
            fig = px.bar(
                agg,
                x=primary_cat,
                y=primary_num,
                color=primary_num,
                color_continuous_scale='Blues',
                title=f'📊 {primary_num} by {primary_cat}',
                text_auto='.2s'
            )
            fig.update_layout(
                height=380,
                showlegend=False,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            fig.update_xaxes(tickangle=30)
            st.plotly_chart(fig, use_container_width=True)

            top_val = agg.loc[agg[primary_num].idxmax(), primary_cat]
            top_num = agg[primary_num].max()
            st.caption(
                f"📌 **{top_val}** leads with {primary_num} = "
                f"{format_number(top_num)}"
            )
        elif primary_num:
            # No categories — show histogram
            fig = px.histogram(
                df, x=primary_num,
                title=f'📊 Distribution of {primary_num}',
                color_discrete_sequence=['#3b82f6'],
                nbins=20
            )
            fig.update_layout(height=380, plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Upload a CSV with numeric columns to see charts.")

    with col2:
        # CHART 2: Trend over time (Line) OR second category (Bar)
        if date_col and primary_num:
            # Line chart — trend over time
            df_t = df.copy()
            df_t['_period'] = df_t[date_col].dt.to_period('M').astype(str)
            trend = (
                df_t.groupby('_period')[primary_num]
                .sum()
                .reset_index()
            )
            fig = px.line(
                trend,
                x='_period',
                y=primary_num,
                title=f'📅 {primary_num} Trend Over Time',
                markers=True,
                color_discrete_sequence=['#6366f1']
            )
            fig.update_layout(
                height=380,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)

            peak = trend.loc[trend[primary_num].idxmax(), '_period']
            st.caption(f"📌 Peak {primary_num} occurred in **{peak}**")

        elif second_cat and primary_num:
            # No dates — use second categorical column
            agg2 = (
                df.groupby(second_cat)[primary_num]
                .sum()
                .reset_index()
                .sort_values(primary_num, ascending=False)
                .head(10)
            )
            fig = px.bar(
                agg2,
                x=second_cat,
                y=primary_num,
                color=primary_num,
                color_continuous_scale='Greens',
                title=f'📊 {primary_num} by {second_cat}',
                text_auto='.2s'
            )
            fig.update_layout(
                height=380,
                showlegend=False,
                plot_bgcolor='rgba(0,0,0,0)'
            )
            fig.update_xaxes(tickangle=30)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(
                "💡 Add a date column to your CSV to see trend charts here."
            )

    # ── ROW 2 ──────────────────────────────────────────────
    col3, col4 = st.columns(2)

    with col3:
        # CHART 3: Category share — Donut
        if primary_cat and primary_num:
            donut = (
                df.groupby(primary_cat)[primary_num]
                .sum()
                .reset_index()
                .sort_values(primary_num, ascending=False)
                .head(8)   # max 8 slices for readability
            )
            fig = px.pie(
                donut,
                values=primary_num,
                names=primary_cat,
                title=f'🍩 {primary_num} Share by {primary_cat}',
                hole=0.45,
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            fig.update_layout(
                height=380,
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)

            top_pct = (
                donut[primary_num].max()
                / donut[primary_num].sum() * 100
            )
            top_cat = donut.loc[
                donut[primary_num].idxmax(), primary_cat
            ]
            st.caption(
                f"📌 **{top_cat}** holds {top_pct:.1f}% of total {primary_num}. "
                f"{'High concentration — consider diversifying.' if top_pct > 45 else 'Relatively balanced.'}"
            )
        else:
            st.info("Add categorical columns to see share breakdown.")

    with col4:
        # CHART 4: Scatter (2 numerics) OR second category bar
        if second_num and primary_cat and primary_num:
            # Scatter — relationship between two metrics
            scatter = df.groupby(primary_cat).agg(
                x=(primary_num, 'sum'),
                y=(second_num, 'sum')
            ).reset_index()

            fig = px.scatter(
                scatter,
                x='x',
                y='y',
                hover_name=primary_cat,
                color='y',
                size='x',
                color_continuous_scale='RdYlGn',
                title=f'🔵 {primary_num} vs {second_num}',
                labels={'x': primary_num, 'y': second_num}
            )
            fig.update_layout(
                height=380,
                showlegend=False,
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                f"📌 Each bubble = one {primary_cat}. "
                f"Bigger = more {primary_num}. "
                f"Greener = more {second_num}."
            )

        elif second_cat and primary_num:
            # No second numeric — bar with second category
            agg3 = (
                df.groupby(second_cat)[primary_num]
                .sum()
                .reset_index()
                .sort_values(primary_num, ascending=False)
                .head(10)
            )
            fig = px.bar(
                agg3,
                x=second_cat,
                y=primary_num,
                color=primary_num,
                color_continuous_scale='Purples',
                title=f'📊 {primary_num} by {second_cat}',
                text_auto='.2s'
            )
            fig.update_layout(
                height=380,
                showlegend=False,
                plot_bgcolor='rgba(0,0,0,0)'
            )
            fig.update_xaxes(tickangle=30)
            st.plotly_chart(fig, use_container_width=True)

        elif second_num:
            # Only numerics, no categories — histogram of second numeric
            fig = px.histogram(
                df, x=second_num,
                title=f'Distribution of {second_num}',
                color_discrete_sequence=['#8b5cf6'],
                nbins=20
            )
            fig.update_layout(height=380, plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(
                "💡 Add more numeric or category columns to "
                "unlock this chart."
            )


# ============================================================
# COLUMN EXPLORER — shows what was auto-detected
# ============================================================
def show_column_explorer(df, col_types):
    with st.expander("🔍 What the app detected in your CSV"):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**📊 Numeric columns**")
            for c in col_types['numeric']:
                st.markdown(f"- `{c}`")
        with c2:
            st.markdown("**🏷️ Category columns**")
            for c in col_types['categorical']:
                st.markdown(f"- `{c}`")
        with c3:
            st.markdown("**📅 Date columns**")
            for c in col_types['datetime']:
                st.markdown(f"- `{c}`")
        if col_types['text']:
            st.markdown(
                f"**Skipped (high-cardinality text):** "
                f"{', '.join([f'`{c}`' for c in col_types['text']])}"
            )
        st.dataframe(df.head(5), use_container_width=True)


# ============================================================
# AI CHAT SECTION
# ============================================================
def ai_chat_section(df, col_types):
    st.markdown('<p class="section-title">💬 Ask the AI Analyst</p>',
                unsafe_allow_html=True)

    num_cols = col_types['numeric']
    cat_cols = col_types['categorical']

    # Auto-generate suggested questions from actual column names
    suggestions = []
    if num_cols and cat_cols:
        suggestions.append(
            f"Which {cat_cols[0]} has the highest {num_cols[0]}?"
        )
    if len(num_cols) > 1 and cat_cols:
        suggestions.append(
            f"Which {cat_cols[0]} has the best {num_cols[1]}?"
        )
    if len(cat_cols) > 1 and num_cols:
        suggestions.append(
            f"Compare {cat_cols[0]} and {cat_cols[1]} by {num_cols[0]}"
        )
    if col_types['datetime'] and num_cols:
        suggestions.append("What is the trend over time?")
    suggestions.append("Give me the top 5 insights from this data")
    suggestions.append("What are the biggest outliers or anomalies?")

    st.markdown("**💡 Try asking:**")
    btn_cols = st.columns(3)
    for i, s in enumerate(suggestions[:6]):
        if btn_cols[i % 3].button(s, key=f"sug_{i}",
                                   use_container_width=True):
            st.session_state['pending_q'] = s

    st.markdown("---")

    # Session state init
    for key, default in [
        ('chat_history', []),
        ('chart_history', []),
        ('followup_history', []),
        ('pending_q', None)
    ]:
        if key not in st.session_state:
            st.session_state[key] = default

    typed = st.chat_input("Ask anything about your data...")

    # Determine active question
    active_q = None
    if typed:
        active_q = typed
        st.session_state['pending_q'] = None
    elif st.session_state.get('pending_q'):
        active_q = st.session_state['pending_q']
        st.session_state['pending_q'] = None

    if active_q:
        st.session_state['chat_history'].append(
            {'role': 'user', 'content': active_q}
        )

        with st.spinner("Analyzing..."):

            # Compute dynamic aggregations from real data
            agg_text = []
            for cat in cat_cols[:3]:
                for num in num_cols[:2]:
                    try:
                        t = (
                            df.groupby(cat)[num]
                            .agg(['sum','mean','count'])
                            .round(1)
                            .reset_index()
                            .sort_values('sum', ascending=False)
                            .head(8)
                            .to_string(index=False)
                        )
                        agg_text.append(f"{num} by {cat}:\n{t}")
                    except:
                        pass

            profile  = get_data_profile(df, col_types)
            agg_data = "\n\n".join(agg_text[:4])

            prompt = f"""You are an expert data analyst.
A user uploaded a CSV dataset and is asking a question.

DATASET PROFILE:
{profile}

AGGREGATED DATA:
{agg_data}

USER QUESTION: "{active_q}"

Answer rules:
- First sentence: direct answer with key number
- 2-3 supporting facts from the data above
- End with one actionable business recommendation
- Under 100 words, no bullet points, natural paragraph
- Use actual column names and values from the data"""

            ai_answer = ask_ai(prompt)

            # Auto generate chart based on question
            chart = None
            try:
                if num_cols and cat_cols:
                    q_low = active_q.lower()

                    # Match mentioned category column
                    m_cat = next(
                        (c for c in cat_cols if c.lower() in q_low),
                        cat_cols[0]
                    )
                    # Match mentioned numeric column
                    m_num = next(
                        (c for c in num_cols if c.lower() in q_low),
                        num_cols[0]
                    )

                    cd = (
                        df.groupby(m_cat)[m_num]
                        .sum()
                        .reset_index()
                        .sort_values(m_num, ascending=False)
                        .head(10)
                    )

                    # Line chart if trend question
                    if any(w in q_low for w in
                           ['trend','over time','month','year','growth']):
                        if col_types['datetime']:
                            dc = col_types['datetime'][0]
                            df_t = df.copy()
                            df_t['_p'] = (
                                df_t[dc].dt.to_period('M').astype(str)
                            )
                            cd = (
                                df_t.groupby('_p')[m_num]
                                .sum()
                                .reset_index()
                            )
                            chart = px.line(
                                cd, x='_p', y=m_num, markers=True,
                                title=f'{m_num} Trend',
                                color_discrete_sequence=['#6366f1']
                            )
                    else:
                        chart = px.bar(
                            cd,
                            x=m_cat, y=m_num,
                            color=m_num,
                            color_continuous_scale='Blues',
                            title=f'{m_num} by {m_cat}',
                            text_auto='.2s'
                        )

                    if chart:
                        chart.update_layout(
                            height=300,
                            showlegend=False,
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)'
                        )
            except:
                chart = None

            # Follow-up suggestions
            fu_raw = ask_ai(
                f"The user asked: '{active_q}' about a dataset with "
                f"columns: {', '.join(df.columns.tolist())}. "
                f"Suggest 3 short follow-up questions (numbered list, "
                f"under 10 words each, no extra text):"
            )
            followups = [
                line[2:].strip()
                for line in fu_raw.strip().splitlines()
                if line.strip() and line.strip()[0].isdigit()
                and len(line) > 2
            ][:3]

        st.session_state['chat_history'].append(
            {'role': 'ai', 'content': ai_answer}
        )
        st.session_state['chart_history'].append(chart)
        st.session_state['followup_history'].append(followups)

    # ── DISPLAY CHAT HISTORY ──
    ai_idx = 0
    for msg in st.session_state['chat_history']:
        if msg['role'] == 'user':
            st.markdown(
                f'<div class="chat-user">🧑 You: {msg["content"]}</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="chat-ai">🤖 AI Analyst: {msg["content"]}</div>',
                unsafe_allow_html=True
            )
            # Chart below AI answer
            if ai_idx < len(st.session_state['chart_history']):
                c = st.session_state['chart_history'][ai_idx]
                if c:
                    st.plotly_chart(c, use_container_width=True)

            # Follow-up buttons
            if ai_idx < len(st.session_state['followup_history']):
                fqs = st.session_state['followup_history'][ai_idx]
                if fqs:
                    st.markdown("💡 *You might also ask:*")
                    fu_c = st.columns(len(fqs))
                    for j, fq in enumerate(fqs):
                        if fu_c[j].button(
                            fq, key=f"fu_{ai_idx}_{j}",
                            use_container_width=True
                        ):
                            st.session_state['pending_q'] = fq
                            st.rerun()
            ai_idx += 1

    if st.session_state['chat_history']:
        st.markdown("---")
        if st.button("🗑️ Clear conversation", key="clear"):
            st.session_state['chat_history']    = []
            st.session_state['chart_history']   = []
            st.session_state['followup_history'] = []
            st.rerun()


# ============================================================
# MAIN
# ============================================================
def main():
    st.title("📊 AI Data Analyst")
    st.caption(
        "Upload **any** CSV file — get instant KPIs, charts & AI insights"
    )

    # ── SIDEBAR ──────────────────────────────────────────
    with st.sidebar:
        st.markdown("### 📂 Upload Data")
        uploaded_file = st.file_uploader(
            "Upload any CSV file",
            type=['csv'],
            help="Works with sales, HR, finance, marketing data — any CSV"
        )

        if uploaded_file is not None:
            df_raw    = load_data(uploaded_file)
            col_types = detect_columns(df_raw)

            st.markdown("---")
            st.markdown("### 🔽 Filters")

            selected_filters = {}

            # Build filter list — prioritise text categoricals first,
            # then add numeric-based categoricals (few unique values)
            # This ensures filters always appear for any dataset type
            text_cat_cols = [
                c for c in col_types['categorical']
                if df_raw[c].dtype == object
            ]
            num_cat_cols = [
                c for c in col_types['categorical']
                if df_raw[c].dtype != object
                and df_raw[c].nunique() <= 15
            ]
            filter_cols = (text_cat_cols + num_cat_cols)[:6]

            if filter_cols:
                for cat in filter_cols:
                    # Convert values to string for display
                    raw_opts = sorted(
                        df_raw[cat].dropna().unique().tolist()
                    )
                    str_opts = [str(o) for o in raw_opts]
                    sel = st.multiselect(
                        f"🏷️ {cat}",
                        options=str_opts,
                        default=str_opts,
                        key=f"f_{cat}"
                    )
                    selected_filters[cat] = {
                        'selected': sel,
                        'original': raw_opts
                    }
            else:
                st.info(
                    "No filterable columns detected.\n"
                    "Filters appear for columns with repeated values "
                    "(like Region, Category, Status)."
                )

            st.markdown("---")
            st.info(
                f"📋 Rows: {len(df_raw):,}\n"
                f"📐 Columns: {len(df_raw.columns)}\n"
                f"🔢 Numeric: {len(col_types['numeric'])}\n"
                f"🏷️ Categories: {len(col_types['categorical'])}\n"
                f"📅 Dates: {len(col_types['datetime'])}"
            )

    # ── WELCOME SCREEN ────────────────────────────────────
    if uploaded_file is None:
        st.markdown("---")
        st.info("👈 Upload any CSV file from the sidebar to get started")

        c1, c2, c3 = st.columns(3)
        with c1:
            st.success(
                "**Works with any CSV**\n\n"
                "Sales · HR · Finance\n"
                "Marketing · Operations"
            )
        with c2:
            st.success(
                "**Auto-detects columns**\n\n"
                "Numbers · Categories\n"
                "Dates · all handled"
            )
        with c3:
            st.success(
                "**Ask in plain English**\n\n"
                "\"Which region is best?\"\n"
                "\"Show me the trend\""
            )

        st.markdown("---")
        st.markdown("#### 📁 Sample CSVs to try:")
        s1, s2, s3 = st.columns(3)
        with s1:
            st.info(
                "**Sales Data**\n"
                "Order_Date, Region, Category\n"
                "Sales, Profit, Discount"
            )
        with s2:
            st.info(
                "**HR Data**\n"
                "Department, Role, Gender\n"
                "Salary, Experience, Rating"
            )
        with s3:
            st.info(
                "**Finance Data**\n"
                "Month, Category, Budget\n"
                "Actual, Variance"
            )
        return

    # ── APPLY FILTERS ─────────────────────────────────────
    df = df_raw.copy()
    for cat, filter_data in selected_filters.items():
        selected_str = filter_data['selected']
        original     = filter_data['original']
        if selected_str and len(selected_str) < len(original):
            # Convert selected strings back to original dtype for filtering
            try:
                dtype = df[cat].dtype
                if pd.api.types.is_numeric_dtype(dtype):
                    selected_vals = [
                        pd.to_numeric(v) for v in selected_str
                    ]
                else:
                    selected_vals = selected_str
                df = df[df[cat].isin(selected_vals)]
            except:
                pass  # if conversion fails, skip this filter

    col_types = detect_columns(df)

    if df.empty:
        st.warning("⚠️ No data matches your filters. Please adjust.")
        return

    st.caption(
        f"Showing **{len(df):,}** rows × **{len(df.columns)}** columns "
        f"| Filtered from {len(df_raw):,} total rows"
    )

    # ── RUN ALL SECTIONS ──────────────────────────────────
    show_column_explorer(df, col_types)
    st.markdown("---")
    generate_kpi_cards(df, col_types)
    st.markdown("---")
    generate_ai_summary(df, col_types)
    st.markdown("---")
    show_charts(df, col_types)
    st.markdown("---")
    ai_chat_section(df, col_types)
    # ── PDF EXPORT ─────────────────────────────────────
    st.markdown("---")
    st.markdown('<p class="section-title">📄 Export Report</p>',
                unsafe_allow_html=True)

    col_exp1, col_exp2 = st.columns([2, 3])

    with col_exp1:
        if PDF_AVAILABLE:
            generate_clicked = st.button(
                "📥 Generate PDF Report",
                use_container_width=True,
                type="primary",
                help="Exports KPIs, charts, AI summary as a PDF"
            )
            if generate_clicked:
                ai_text = st.session_state.get('last_ai_summary', '')
                with st.spinner("Building your PDF report..."):
                    try:
                        # Pass chat history so Q&A appears in PDF
                        chat_hist = st.session_state.get('chat_history', [])
                        pdf_bytes = generate_pdf_report(
                            df=df,
                            col_types=col_types,
                            ai_summary=ai_text,
                            chat_history=chat_hist,
                        )
                        st.download_button(
                            label="⬇️ Click to save PDF",
                            data=pdf_bytes,
                            file_name=f"report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.pdf",
                            mime="application/pdf",
                            use_container_width=True,
                        )
                        st.success("PDF ready! Click the button above.")
                    except Exception as e:
                        st.error(f"Error: {e}")
        else:
            st.warning(
                "PDF export unavailable. "
                "Ensure pdf_export.py is in the same folder as app.py."
            )

    with col_exp2:
        st.info(
            "**The PDF report includes:**\n"
            "Page 1: Dataset overview + KPIs + AI summary + stats table\n"
            "Page 2: 4 business charts\n"
            "Page 3: First 20 rows data sample"
        )

    st.markdown("---")
    st.caption(
        "Built with Streamlit · Pandas · Plotly · "
        "Groq LLaMA 3.3 70B · Works on any CSV"
    )


if __name__ == "__main__":
    main()