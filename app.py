# ============================================================
# AI BUSINESS ANALYST DASHBOARD — FULLY GENERIC VERSION
# Works on ANY CSV dataset — no hardcoded column names
#
# HOW IT WORKS:
#   1. User uploads any CSV
#   2. We auto-detect column types (numbers, categories, dates)
#   3. Gemini reads column names + samples to understand context
#   4. KPIs, charts, and AI insights are built dynamically
#   5. Chat answers questions about whatever data is uploaded
# ============================================================

import streamlit as st
import pandas as pd
import plotly.express as px
from dotenv import load_dotenv
import os
from google import genai

# ── SETUP ──────────────────────────────────────────────────
load_dotenv()
client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY", "")
)
MODEL = "gemini-2.0-flash-001"

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
        line-height: 1.75;
    }
    .section-title {
        font-size: 1.15rem;
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
        line-height: 1.75;
    }
    .col-badge {
        display: inline-block;
        font-size: 11px;
        padding: 2px 8px;
        border-radius: 999px;
        margin: 2px;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# CORE AI FUNCTION WITH CACHING
# @st.cache_data means: same prompt = cached result (no API call)
# This saves quota — critical for free tier
# ============================================================
@st.cache_data(ttl=600, show_spinner=False)
def ask_gemini(prompt):
    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=prompt
        )
        return response.text
    except Exception as e:
        return f"AI temporarily unavailable: {str(e)}"


# ============================================================
# COLUMN DETECTION — THE HEART OF GENERIC MODE
#
# This function scans every column and classifies it as:
#   - numeric    → numbers (Sales, Age, Score, Price...)
#   - categorical → text with few unique values (Region, Status...)
#   - datetime   → dates (Order_Date, Created_At...)
#   - text       → free text (Name, Description...) — skip for charts
#
# WHY WE NEED THIS:
# Every dataset has different column names.
# Instead of assuming "Sales" or "Region" exist,
# we detect what's available and build analysis from that.
# ============================================================
def detect_columns(df):
    """
    Auto-detects column types in any DataFrame.
    Returns a dict with lists of column names by type.
    """
    cols = {
        'numeric'     : [],   # number columns → KPIs, y-axis of charts
        'categorical' : [],   # text with <30 unique values → filters, x-axis
        'datetime'    : [],   # date columns → trend charts
        'text'        : [],   # high-cardinality text → skip (names, IDs)
    }

    for col in df.columns:

        # ── DATE DETECTION ──
        # First check if it's already a datetime type
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            cols['datetime'].append(col)
            continue

        # Also try converting object columns to datetime
        # Some CSVs store dates as text like "2021-01-15"
        if df[col].dtype == object:
            try:
                pd.to_datetime(df[col], infer_datetime_format=True)
                cols['datetime'].append(col)
                continue
            except:
                pass

        # ── NUMERIC DETECTION ──
        if pd.api.types.is_numeric_dtype(df[col]):
            cols['numeric'].append(col)
            continue

        # ── CATEGORICAL vs TEXT DETECTION ──
        # If a text column has fewer than 30 unique values,
        # it's probably a category (Region, Status, Department)
        # If it has many unique values, it's free text (Name, ID)
        if df[col].dtype == object:
            unique_count = df[col].nunique()
            if unique_count <= 30:
                cols['categorical'].append(col)
            else:
                cols['text'].append(col)

    return cols


# ============================================================
# DATA LOADING — with auto date conversion
# ============================================================
def load_data(uploaded_file):
    """
    Reads any CSV and auto-converts date columns to datetime.

    GENERIC APPROACH:
    We don't know which column is the date in advance.
    So we try converting every object column to datetime.
    If it works → mark it as datetime.
    If it fails → leave it as text.
    """
    df = pd.read_csv(uploaded_file)

    # Try to auto-convert any column that looks like a date
    for col in df.columns:
        if df[col].dtype == object:
            try:
                converted = pd.to_datetime(df[col], infer_datetime_format=True)
                df[col] = converted
            except:
                pass  # not a date column, leave as-is

    return df


# ============================================================
# DATA PROFILE — text summary sent to Gemini as context
#
# GENERIC VERSION:
# Instead of hardcoding which columns to summarize,
# we dynamically summarize whatever columns exist.
# ============================================================
def get_data_profile(df, col_types):
    """
    Builds a comprehensive text profile of any dataset.
    This is passed to Gemini so it understands the data context.
    """
    lines = []
    lines.append(f"Dataset shape: {len(df):,} rows × {len(df.columns)} columns")
    lines.append(f"Columns: {', '.join(df.columns.tolist())}")

    # Date range (if any date columns exist)
    if col_types['datetime']:
        date_col = col_types['datetime'][0]
        lines.append(f"Date range: {df[date_col].min()} to {df[date_col].max()}")

    # Numeric column stats
    if col_types['numeric']:
        lines.append("\nNUMERIC COLUMNS SUMMARY:")
        for col in col_types['numeric'][:6]:  # max 6 to keep prompt short
            lines.append(
                f"  {col}: min={df[col].min():,.1f}, "
                f"max={df[col].max():,.1f}, "
                f"mean={df[col].mean():,.1f}, "
                f"sum={df[col].sum():,.1f}"
            )

    # Categorical column stats
    if col_types['categorical']:
        lines.append("\nCATEGORY COLUMNS SUMMARY:")
        for col in col_types['categorical'][:5]:  # max 5
            top_val = df[col].value_counts().index[0]
            unique_n = df[col].nunique()
            lines.append(f"  {col}: {unique_n} unique values, most common = '{top_val}'")

    return "\n".join(lines)


# ============================================================
# GENERIC KPI CARDS
#
# GENERIC APPROACH:
# Show the top 4 numeric columns as KPI cards.
# No assumption about which columns exist.
# ============================================================
def generate_kpi_cards(df, col_types):
    st.markdown('<p class="section-title">📈 Key Metrics</p>',
                unsafe_allow_html=True)

    numeric_cols = col_types['numeric']

    if not numeric_cols:
        st.info("No numeric columns found in this dataset.")
        return

    # Show up to 4 numeric columns as KPI cards
    display_cols = numeric_cols[:4]
    cols = st.columns(len(display_cols))

    for i, col in enumerate(display_cols):
        total = df[col].sum()
        avg   = df[col].mean()
        with cols[i]:
            # Format large numbers nicely
            if abs(total) >= 1e7:
                formatted = f"{total/1e7:.2f} Cr"
            elif abs(total) >= 1e5:
                formatted = f"{total/1e5:.1f} L"
            elif abs(total) >= 1e3:
                formatted = f"{total/1e3:.1f} K"
            else:
                formatted = f"{total:,.2f}"

            st.metric(
                label=f"Total {col}",
                value=formatted,
                help=f"Average: {avg:,.2f} | Min: {df[col].min():,.2f} | Max: {df[col].max():,.2f}"
            )


# ============================================================
# GENERIC AI EXECUTIVE SUMMARY
#
# GENERIC APPROACH:
# We pass the full data profile (whatever columns exist) to Gemini.
# The AI figures out what kind of data it is and writes accordingly.
# ============================================================
def generate_ai_summary(df, col_types):
    st.markdown('<p class="section-title">🤖 AI Executive Summary</p>',
                unsafe_allow_html=True)

    with st.spinner("AI is analyzing your data..."):
        profile = get_data_profile(df, col_types)

        # Sample rows help the AI understand context better
        sample = df.head(3).to_string(index=False)

        prompt = f"""
        You are a senior data analyst. A user has uploaded a dataset to you.
        Analyze it and write a professional executive summary.

        DATASET PROFILE:
        {profile}

        SAMPLE ROWS (first 3):
        {sample}

        Write a 4-5 sentence executive summary that:
        1. First identifies what type of data this is (sales, HR, finance, etc.)
        2. States the most important top-line finding (biggest number, key trend)
        3. Highlights one strong performer (top category, region, product etc.)
        4. Flags one concern or anomaly worth investigating
        5. Ends with one specific actionable recommendation

        Rules:
        - Write in flowing paragraph form — NO bullet points
        - Be specific: use actual column names and values from the data
        - Sound like a professional analyst presenting to leadership
        - Under 130 words
        - If numbers are large, use K/L/Cr format for Indian data or M/B for others
        """

        ai_text = ask_gemini(prompt)
        st.markdown(
            f'<div class="insight-box">{ai_text}</div>',
            unsafe_allow_html=True
        )


# ============================================================
# GENERIC CHARTS — 4 charts built from whatever columns exist
#
# CHART LOGIC:
#   Chart 1: Best numeric col grouped by best categorical col → Bar
#   Chart 2: Best numeric col over time → Line (if date exists)
#             OR second categorical col → Bar (if no date)
#   Chart 3: All numeric cols distribution → Pie/Donut of first cat col
#   Chart 4: Two numeric cols → Scatter (if 2+ numeric exist)
#             OR bar chart of second cat col
# ============================================================
def show_charts(df, col_types):
    st.markdown('<p class="section-title">📊 Visual Analysis</p>',
                unsafe_allow_html=True)

    numeric_cols     = col_types['numeric']
    categorical_cols = col_types['categorical']
    datetime_cols    = col_types['datetime']

    if not numeric_cols:
        st.warning("No numeric columns found — cannot generate charts.")
        return

    # Pick the best columns to visualize
    # "Best" = the one with highest variance (most interesting to chart)
    primary_num = numeric_cols[0]   # first numeric = main KPI
    second_num  = numeric_cols[1] if len(numeric_cols) > 1 else None
    primary_cat = categorical_cols[0] if categorical_cols else None
    second_cat  = categorical_cols[1] if len(categorical_cols) > 1 else None
    date_col    = datetime_cols[0] if datetime_cols else None

    col1, col2 = st.columns(2)

    # ── CHART 1: Primary numeric grouped by primary category ──
    with col1:
        if primary_cat:
            # groupby the category and sum the main numeric column
            chart_df = (
                df.groupby(primary_cat)[primary_num]
                .sum()
                .reset_index()
                .sort_values(primary_num, ascending=False)
                .head(10)  # top 10 only so chart isn't crowded
            )
            fig = px.bar(
                chart_df,
                x=primary_cat,
                y=primary_num,
                color=primary_num,
                color_continuous_scale='Blues',
                title=f'📊 {primary_num} by {primary_cat}',
                text_auto='.2s'
            )
            fig.update_layout(height=370, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

            # Smart caption using Pandas (no AI call needed)
            top_val = chart_df.loc[chart_df[primary_num].idxmax(), primary_cat]
            top_num = chart_df[primary_num].max()
            st.caption(
                f"📌 {top_val} leads with {primary_num} = "
                f"{top_num:,.1f} — the highest in this category."
            )
        else:
            # No categorical column — show histogram of main numeric
            fig = px.histogram(
                df, x=primary_num,
                title=f'📊 Distribution of {primary_num}',
                color_discrete_sequence=['#3b82f6']
            )
            fig.update_layout(height=370)
            st.plotly_chart(fig, use_container_width=True)

    # ── CHART 2: Trend over time OR second category ──
    with col2:
        if date_col:
            # Time series trend
            df_copy = df.copy()
            df_copy['_period'] = df_copy[date_col].dt.to_period('M').astype(str)
            trend_df = (
                df_copy.groupby('_period')[primary_num]
                .sum()
                .reset_index()
            )
            fig = px.line(
                trend_df,
                x='_period',
                y=primary_num,
                title=f'📅 {primary_num} Over Time',
                markers=True
            )
            fig.update_layout(height=370)
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)

            peak_period = trend_df.loc[trend_df[primary_num].idxmax(), '_period']
            st.caption(f"📌 Peak {primary_num} was in {peak_period}.")

        elif second_cat:
            # No date — use second categorical column instead
            chart_df2 = (
                df.groupby(second_cat)[primary_num]
                .sum()
                .reset_index()
                .sort_values(primary_num, ascending=False)
                .head(10)
            )
            fig = px.bar(
                chart_df2,
                x=second_cat,
                y=primary_num,
                color=primary_num,
                color_continuous_scale='Greens',
                title=f'📊 {primary_num} by {second_cat}',
                text_auto='.2s'
            )
            fig.update_layout(height=370, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Add a date column for trend analysis.")

    col3, col4 = st.columns(2)

    # ── CHART 3: Category share (donut) ──
    with col3:
        if primary_cat:
            donut_df = (
                df.groupby(primary_cat)[primary_num]
                .sum()
                .reset_index()
                .sort_values(primary_num, ascending=False)
                .head(8)  # max 8 slices for readability
            )
            fig = px.pie(
                donut_df,
                values=primary_num,
                names=primary_cat,
                title=f'🍩 {primary_num} Share by {primary_cat}',
                hole=0.45,
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            fig.update_layout(height=370)
            st.plotly_chart(fig, use_container_width=True)

            top_pct = donut_df[primary_num].max() / donut_df[primary_num].sum() * 100
            top_cat = donut_df.loc[donut_df[primary_num].idxmax(), primary_cat]
            st.caption(
                f"📌 {top_cat} accounts for {top_pct:.1f}% of total {primary_num}. "
                f"{'High concentration.' if top_pct > 40 else 'Fairly distributed.'}"
            )
        else:
            st.info("Add a categorical column for share analysis.")

    # ── CHART 4: Scatter (two numerics) OR second cat bar ──
    with col4:
        if second_num and primary_cat:
            # Scatter: relationship between two numeric columns
            scatter_df = (
                df.groupby(primary_cat)
                .agg(
                    x_val=(primary_num, 'sum'),
                    y_val=(second_num, 'sum')
                )
                .reset_index()
            )
            fig = px.scatter(
                scatter_df,
                x='x_val',
                y='y_val',
                hover_name=primary_cat,
                color='y_val',
                color_continuous_scale='RdYlGn',
                title=f'🔵 {primary_num} vs {second_num}',
                labels={'x_val': primary_num, 'y_val': second_num}
            )
            fig.update_layout(height=370, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                f"📌 Each point = one {primary_cat}. "
                f"Points in the top-right corner are highest in both {primary_num} and {second_num}."
            )
        elif second_cat:
            chart_df3 = (
                df.groupby(second_cat)[primary_num]
                .sum()
                .reset_index()
                .sort_values(primary_num, ascending=False)
                .head(10)
            )
            fig = px.bar(
                chart_df3,
                x=second_cat,
                y=primary_num,
                color=primary_num,
                color_continuous_scale='Purples',
                title=f'📊 {primary_num} by {second_cat}',
                text_auto='.2s'
            )
            fig.update_layout(height=370, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            # Fallback: histogram of second numeric
            if second_num:
                fig = px.histogram(df, x=second_num,
                                   title=f'Distribution of {second_num}')
                fig.update_layout(height=370)
                st.plotly_chart(fig, use_container_width=True)


# ============================================================
# GENERIC AI CHAT
#
# GENERIC APPROACH:
# Instead of pre-computing region/category/segment stats,
# we compute stats dynamically based on whatever columns exist.
# The chat works on any dataset.
# ============================================================
def ai_chat_section(df, col_types):
    st.markdown('<p class="section-title">💬 Ask the AI Analyst</p>',
                unsafe_allow_html=True)

    numeric_cols     = col_types['numeric']
    categorical_cols = col_types['categorical']

    # Auto-generate suggested questions based on actual columns
    suggestions = []
    if numeric_cols and categorical_cols:
        suggestions.append(f"Which {categorical_cols[0]} has highest {numeric_cols[0]}?")
        suggestions.append(f"What is the average {numeric_cols[0]}?")
    if len(categorical_cols) > 1:
        suggestions.append(f"Compare {categorical_cols[0]} vs {categorical_cols[1]}")
    if len(numeric_cols) > 1:
        suggestions.append(f"Which {categorical_cols[0] if categorical_cols else 'group'} has best {numeric_cols[1]}?")
    if col_types['datetime']:
        suggestions.append("What is the trend over time?")
    suggestions.append("Give me the top 5 insights from this data")

    if suggestions:
        st.markdown("**Try asking:**")
        cols = st.columns(min(len(suggestions), 3))
        for i, s in enumerate(suggestions[:6]):
            if cols[i % 3].button(s, key=f"sug_{i}", use_container_width=True):
                st.session_state['pending_question'] = s

    st.markdown("---")

    # Initialize session state
    for key in ['chat_history', 'chart_history', 'followup_history', 'pending_question']:
        if key not in st.session_state:
            st.session_state[key] = [] if key != 'pending_question' else None

    typed_question = st.chat_input("Ask anything about your data...")

    active_question = None
    if typed_question:
        active_question = typed_question
        st.session_state['pending_question'] = None
    elif st.session_state.get('pending_question'):
        active_question = st.session_state['pending_question']
        st.session_state['pending_question'] = None

    if active_question:
        st.session_state['chat_history'].append({
            'role': 'user', 'content': active_question
        })

        with st.spinner("Analyzing..."):
            # Build dynamic context from whatever columns exist
            profile = get_data_profile(df, col_types)

            # Compute aggregation tables dynamically
            agg_tables = []
            for cat in categorical_cols[:3]:
                for num in numeric_cols[:2]:
                    agg = (
                        df.groupby(cat)[num]
                        .agg(['sum', 'mean', 'count'])
                        .round(2)
                        .reset_index()
                        .sort_values('sum', ascending=False)
                        .head(10)
                    )
                    agg_tables.append(f"\n{num} by {cat}:\n{agg.to_string(index=False)}")

            agg_context = "\n".join(agg_tables[:4])  # max 4 tables

            prompt = f"""
            You are an expert data analyst. A user uploaded a dataset and is asking a question.

            DATASET PROFILE:
            {profile}

            AGGREGATED DATA:
            {agg_context}

            USER QUESTION: "{active_question}"

            Answer rules:
            - First sentence: direct answer with the key number
            - Support with 2-3 specific facts from the data above
            - End with one actionable recommendation
            - Under 100 words. No bullet points. Natural paragraph.
            - Use actual column names and values from the data
            - If numbers are large use K/L/Cr format
            """

            ai_answer = ask_gemini(prompt)

            # Generate chart for the answer dynamically
            chart = None
            if numeric_cols and categorical_cols:
                q_lower = active_question.lower()
                # Find which categorical column the question is about
                matched_cat = None
                for cat in categorical_cols:
                    if cat.lower() in q_lower:
                        matched_cat = cat
                        break
                matched_cat = matched_cat or categorical_cols[0]

                # Find which numeric column the question is about
                matched_num = None
                for num in numeric_cols:
                    if num.lower() in q_lower:
                        matched_num = num
                        break
                matched_num = matched_num or numeric_cols[0]

                try:
                    chart_df = (
                        df.groupby(matched_cat)[matched_num]
                        .sum()
                        .reset_index()
                        .sort_values(matched_num, ascending=False)
                        .head(10)
                    )
                    chart = px.bar(
                        chart_df,
                        x=matched_cat, y=matched_num,
                        color=matched_num,
                        color_continuous_scale='Blues',
                        title=f'{matched_num} by {matched_cat}',
                        text_auto='.2s'
                    )
                    chart.update_layout(height=300, showlegend=False)
                except:
                    chart = None

            # Follow-up suggestions
            fu_prompt = f"""
            User asked: "{active_question}" about a dataset with columns: {', '.join(df.columns.tolist())}
            Suggest 3 short follow-up questions (numbered list, under 10 words each, no extra text):
            """
            fu_raw = ask_gemini(fu_prompt)
            followups = []
            for line in fu_raw.strip().splitlines():
                line = line.strip()
                if line and line[0].isdigit() and len(line) > 2:
                    followups.append(line[2:].strip())
            followups = followups[:3]

        st.session_state['chat_history'].append({'role': 'ai', 'content': ai_answer})
        st.session_state['chart_history'].append(chart)
        st.session_state['followup_history'].append(followups)

    # Display conversation
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
            if ai_idx < len(st.session_state['chart_history']):
                chart = st.session_state['chart_history'][ai_idx]
                if chart:
                    st.plotly_chart(chart, use_container_width=True)
            if ai_idx < len(st.session_state['followup_history']):
                fqs = st.session_state['followup_history'][ai_idx]
                if fqs:
                    st.markdown("💡 *You might also ask:*")
                    fu_cols = st.columns(len(fqs))
                    for j, fq in enumerate(fqs):
                        if fu_cols[j].button(fq, key=f"fu_{ai_idx}_{j}",
                                              use_container_width=True):
                            st.session_state['pending_question'] = fq
                            st.rerun()
            ai_idx += 1

    if st.session_state['chat_history']:
        st.markdown("---")
        if st.button("🗑️ Clear conversation"):
            st.session_state['chat_history']    = []
            st.session_state['chart_history']   = []
            st.session_state['followup_history'] = []
            st.rerun()


# ============================================================
# COLUMN EXPLORER — shows user what columns were detected
# This is a transparency feature — user knows what the app found
# ============================================================
def show_column_explorer(df, col_types):
    with st.expander("🔍 Column Analysis — What the app detected in your data"):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**📊 Numeric Columns**")
            for col in col_types['numeric']:
                st.markdown(
                    f'<span class="col-badge" style="background:#dbeafe;color:#1e40af">'
                    f'{col}</span>', unsafe_allow_html=True
                )
        with c2:
            st.markdown("**🏷️ Category Columns**")
            for col in col_types['categorical']:
                st.markdown(
                    f'<span class="col-badge" style="background:#dcfce7;color:#166534">'
                    f'{col}</span>', unsafe_allow_html=True
                )
        with c3:
            st.markdown("**📅 Date Columns**")
            for col in col_types['datetime']:
                st.markdown(
                    f'<span class="col-badge" style="background:#fef9c3;color:#854d0e">'
                    f'{col}</span>', unsafe_allow_html=True
                )
        st.dataframe(df.head(5), use_container_width=True)
        st.caption(f"Showing first 5 of {len(df):,} rows")


# ============================================================
# MAIN
# ============================================================
def main():
    st.title("📊 AI Data Analyst")
    st.caption("Upload any CSV — get instant AI-powered insights, charts, and Q&A")

    with st.sidebar:
        st.markdown("### 📂 Upload Your Data")
        uploaded_file = st.file_uploader(
            "Upload any CSV file",
            type=['csv'],
            help="Works with sales, HR, finance, marketing, or any dataset"
        )

        if uploaded_file is not None:
            df_raw = load_data(uploaded_file)
            col_types = detect_columns(df_raw)

            st.markdown("---")
            st.markdown("### 🔽 Filters")

            # Dynamically build filters for all categorical columns
            # We don't know which columns exist in advance
            selected_filters = {}
            for cat_col in col_types['categorical'][:4]:  # max 4 filters
                options = sorted(df_raw[cat_col].dropna().unique().tolist())
                selected = st.multiselect(
                    f"🏷️ {cat_col}",
                    options=options,
                    default=options,
                    key=f"filter_{cat_col}"
                )
                selected_filters[cat_col] = selected

            st.markdown("---")
            st.info(
                f"Rows: {len(df_raw):,}\n"
                f"Columns: {len(df_raw.columns)}\n"
                f"Numeric: {len(col_types['numeric'])}\n"
                f"Categories: {len(col_types['categorical'])}\n"
                f"Dates: {len(col_types['datetime'])}"
            )

    if uploaded_file is None:
        st.markdown("---")
        st.info("👈 Upload any CSV file from the sidebar to get started")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.success("**Works with any CSV**\nSales, HR, Finance, Marketing...")
        with c2:
            st.success("**Auto-detects columns**\nNo setup or configuration needed")
        with c3:
            st.success("**Ask any question**\nIn plain English about your data")

        st.markdown("---")
        st.markdown("#### 📁 Sample datasets you can try:")
        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            st.info("**Sales Data**\nOrders, Revenue, Region, Product")
        with sc2:
            st.info("**HR Data**\nEmployees, Department, Salary, Tenure")
        with sc3:
            st.info("**Finance Data**\nExpenses, Category, Budget, Actual")
        return

    # Apply all dynamic filters
    df = df_raw.copy()
    for cat_col, selected_vals in selected_filters.items():
        if selected_vals:
            df = df[df[cat_col].isin(selected_vals)]

    col_types = detect_columns(df)

    if df.empty:
        st.warning("⚠️ No data matches your filters. Please adjust.")
        return

    st.caption(
        f"Analyzing **{len(df):,}** rows × **{len(df.columns)}** columns | "
        f"Filtered from {len(df_raw):,} total rows"
    )

    show_column_explorer(df, col_types)
    st.markdown("---")
    generate_kpi_cards(df, col_types)
    st.markdown("---")
    generate_ai_summary(df, col_types)
    st.markdown("---")
    show_charts(df, col_types)
    st.markdown("---")
    ai_chat_section(df, col_types)
    st.markdown("---")
    st.caption("Built with Streamlit · Pandas · Plotly · Google Gemini AI · Works on any CSV")


if __name__ == "__main__":
    main()