# ============================================================
# data_quality.py — Automated Data Quality Report Module
# Save this in the SAME folder as app.py
#
# WHAT IT DOES:
#   1. Missing Values Analysis — % missing per column + heatmap
#   2. Duplicate Rows Detection — exact and near duplicates
#   3. Outlier Detection — IQR method per numeric column
#   4. Data Type Summary — column types + sample values
#   5. AI Recommendations — Groq writes fix suggestions
#
# HOW IQR OUTLIER METHOD WORKS (know this for interviews):
#   Q1 = 25th percentile, Q3 = 75th percentile
#   IQR = Q3 - Q1  (the middle 50% spread)
#   Lower fence = Q1 - 1.5 * IQR
#   Upper fence = Q3 + 1.5 * IQR
#   Any value outside these fences = outlier
#   This is robust to skewed data — better than Z-score for most DA work
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ============================================================
# SECTION 1 — MISSING VALUES
# ============================================================
def analyze_missing(df):
    """
    Calculates missing value stats for every column.
    Returns a summary DataFrame.
    """
    total      = len(df)
    missing    = df.isnull().sum()
    pct        = (missing / total * 100).round(2)
    dtype      = df.dtypes.astype(str)

    summary = pd.DataFrame({
        'Column'       : missing.index,
        'Missing Count': missing.values,
        'Missing %'    : pct.values,
        'Data Type'    : dtype.values,
        'Non-Null Count': (total - missing.values),
    })
    summary = summary.sort_values('Missing %', ascending=False)
    summary = summary.reset_index(drop=True)
    return summary


def show_missing_section(df, ask_ai_fn):
    """Renders the full missing values section."""

    st.markdown("#### 🕳️ Missing Values Analysis")

    missing_summary = analyze_missing(df)
    cols_with_missing = missing_summary[
        missing_summary['Missing Count'] > 0
    ]
    total_missing = df.isnull().sum().sum()
    total_cells   = df.shape[0] * df.shape[1]
    overall_pct   = round(total_missing / total_cells * 100, 2)

    # ── Top metrics ──
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Missing Cells",  f"{total_missing:,}")
    m2.metric("Overall Missing %",    f"{overall_pct}%")
    m3.metric("Columns Affected",     f"{len(cols_with_missing)}")
    m4.metric("Complete Columns",
              f"{len(df.columns) - len(cols_with_missing)}")

    if total_missing == 0:
        st.success("✅ No missing values found — dataset is complete!")
        return missing_summary

    # ── Bar chart: missing % per column ──
    fig = px.bar(
        cols_with_missing,
        x='Column',
        y='Missing %',
        color='Missing %',
        color_continuous_scale='Reds',
        title='Missing Values % by Column',
        text='Missing %',
    )
    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig.update_layout(
        height=350,
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )
    fig.add_hline(
        y=20,
        line_dash='dash',
        line_color='orange',
        annotation_text='20% threshold — consider dropping'
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Missing value heatmap ──
    # Shows WHERE the missing values are across all rows
    st.markdown("**Missing Value Pattern Heatmap**")
    st.caption(
        "Each column = one row below. "
        "Dark = missing, Light = present. "
        "Shows if missingness is random or clustered."
    )

    # Sample max 200 rows so heatmap isn't too slow
    sample_df  = df[cols_with_missing['Column'].tolist()].head(200)
    # 1 = missing, 0 = present
    heatmap_df = sample_df.isnull().astype(int)

    fig2 = px.imshow(
        heatmap_df.T,           # Transpose: columns on y-axis
        color_continuous_scale=[[0, '#e0f2fe'], [1, '#dc2626']],
        aspect='auto',
        title='Missing Value Heatmap (first 200 rows)',
        labels=dict(x='Row Index', y='Column', color='Missing'),
    )
    fig2.update_layout(
        height=max(200, len(cols_with_missing) * 35 + 80),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        coloraxis_showscale=False,
    )
    st.plotly_chart(fig2, use_container_width=True)

    # ── Detail table ──
    with st.expander("📋 Full Missing Values Table"):
        # background_gradient works fine — just ensure numeric column
        try:
            styled_miss = missing_summary.style.background_gradient(
                subset=['Missing %'], cmap='Reds'
            )
            st.dataframe(styled_miss, use_container_width=True)
        except Exception:
            st.dataframe(missing_summary, use_container_width=True)

    # ── AI Recommendation ──
    with st.spinner("AI analyzing missing values..."):
        high_missing = cols_with_missing[
            cols_with_missing['Missing %'] > 5
        ]['Column'].tolist()
        low_missing  = cols_with_missing[
            cols_with_missing['Missing %'] <= 5
        ]['Column'].tolist()

        prompt = f"""You are a senior data analyst reviewing data quality.

Missing value analysis results:
- Total missing cells: {total_missing:,} ({overall_pct}% of all data)
- Columns with >5% missing: {high_missing if high_missing else 'None'}
- Columns with ≤5% missing: {low_missing if low_missing else 'None'}
- Dataset size: {df.shape[0]:,} rows × {df.shape[1]} columns

Give 3-4 specific, actionable recommendations to handle these missing values.
For each column with >5% missing, suggest whether to: drop the column,
impute with mean/median/mode, or flag as a separate category.
Use data analyst language. Under 120 words. No bullet symbols — use
numbered lines like 1. 2. 3."""

        ai_text = ask_ai_fn(prompt)

        st.markdown(
            f'<div style="background:#fef9c3;border-left:4px solid '
            f'#eab308;padding:12px 16px;border-radius:0 8px 8px 0;'
            f'font-size:0.9rem;color:#854d0e;line-height:1.7;'
            f'margin-top:8px">'
            f'🤖 <strong>AI Recommendation:</strong><br>{ai_text}</div>',
            unsafe_allow_html=True
        )

    return missing_summary


# ============================================================
# SECTION 2 — DUPLICATE ROWS
# ============================================================
def analyze_duplicates(df):
    """
    Finds exact duplicate rows.
    Returns count, percentage, and sample of duplicates.
    """
    dup_mask    = df.duplicated()
    dup_count   = dup_mask.sum()
    dup_pct     = round(dup_count / len(df) * 100, 2)
    dup_rows    = df[dup_mask]
    return dup_count, dup_pct, dup_rows


def show_duplicates_section(df, ask_ai_fn):
    """Renders the duplicates section."""

    st.markdown("#### 👥 Duplicate Rows Detection")

    dup_count, dup_pct, dup_rows = analyze_duplicates(df)

    d1, d2, d3 = st.columns(3)
    d1.metric("Duplicate Rows",   f"{dup_count:,}")
    d2.metric("Duplicate %",      f"{dup_pct}%")
    d3.metric("Unique Rows",      f"{len(df) - dup_count:,}")

    if dup_count == 0:
        st.success("✅ No duplicate rows found — dataset is unique!")
        return

    # Donut chart showing unique vs duplicate split
    fig = px.pie(
        values=[len(df) - dup_count, dup_count],
        names=['Unique Rows', 'Duplicate Rows'],
        color_discrete_sequence=['#22c55e', '#ef4444'],
        hole=0.55,
        title=f'Dataset Composition — {dup_pct}% duplicates'
    )
    fig.update_layout(
        height=300,
        paper_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig, use_container_width=True)

    # Show sample duplicates
    with st.expander(f"🔍 View {min(dup_count, 10)} sample duplicate rows"):
        st.dataframe(dup_rows.head(10), use_container_width=True)
        st.caption(
            f"These rows are exact copies of other rows in the dataset."
        )

    # AI Recommendation
    with st.spinner("AI analyzing duplicates..."):
        prompt = f"""You are a senior data analyst.

Duplicate row analysis:
- Dataset has {len(df):,} total rows
- {dup_count:,} rows ({dup_pct}%) are exact duplicates
- {len(df) - dup_count:,} rows are unique

Give 2-3 specific recommendations:
1. Should these duplicates be dropped or investigated first?
2. What might have caused them (data pipeline issues, manual entry, etc.)?
3. What to check before removing them?
Under 80 words. Numbered lines. Data analyst language."""

        ai_text = ask_ai_fn(prompt)
        st.markdown(
            f'<div style="background:#fef9c3;border-left:4px solid '
            f'#eab308;padding:12px 16px;border-radius:0 8px 8px 0;'
            f'font-size:0.9rem;color:#854d0e;line-height:1.7;'
            f'margin-top:8px">'
            f'🤖 <strong>AI Recommendation:</strong><br>{ai_text}</div>',
            unsafe_allow_html=True
        )

    # One-click clean button
    st.markdown("**Quick Fix:**")
    if st.button("🧹 Remove duplicates from analysis",
                 key="remove_dups"):
        st.session_state['dups_removed'] = True
        st.success(
            f"✅ {dup_count} duplicate rows will be excluded from "
            f"all charts and analysis. Refresh the page to apply."
        )
        st.info(
            "Note: This removes duplicates from the current session only. "
            "Your original file is unchanged."
        )


# ============================================================
# SECTION 3 — OUTLIER DETECTION (IQR METHOD)
# ============================================================
def detect_outliers_iqr(df, col):
    """
    Detects outliers in a single numeric column using IQR method.

    IQR METHOD EXPLAINED (for interviews):
        Q1  = 25th percentile (bottom quarter value)
        Q3  = 75th percentile (top quarter value)
        IQR = Q3 - Q1  (the spread of the middle 50%)
        Lower fence = Q1 - 1.5 × IQR
        Upper fence = Q3 + 1.5 × IQR
        Outlier = any value below lower or above upper fence

    WHY 1.5?
        John Tukey (inventor of box plots) chose 1.5 as the multiplier.
        It captures ~99.3% of normally distributed data inside the fences.
        Values outside are statistically unusual enough to investigate.
    """
    series = df[col].dropna()
    Q1  = series.quantile(0.25)
    Q3  = series.quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers     = series[(series < lower) | (series > upper)]
    outlier_pct  = round(len(outliers) / len(series) * 100, 2)

    return {
        'column'       : col,
        'Q1'           : round(Q1, 2),
        'Q3'           : round(Q3, 2),
        'IQR'          : round(IQR, 2),
        'lower_fence'  : round(lower, 2),
        'upper_fence'  : round(upper, 2),
        'outlier_count': len(outliers),
        'outlier_pct'  : outlier_pct,
        'min_outlier'  : round(outliers.min(), 2) if len(outliers) else None,
        'max_outlier'  : round(outliers.max(), 2) if len(outliers) else None,
        'outlier_values': outliers,
    }


def show_outliers_section(df, col_types, ask_ai_fn):
    """Renders the full outliers section."""

    st.markdown("#### 📦 Outlier Detection — IQR Method")

    num_cols = [
        c for c in col_types['numeric']
        if c not in col_types['categorical']
    ]

    if not num_cols:
        st.info("No pure numeric columns found for outlier analysis.")
        return

    # Run IQR on all numeric columns
    results = []
    for col in num_cols[:10]:   # max 10 columns
        r = detect_outliers_iqr(df, col)
        results.append(r)

    results_df = pd.DataFrame([{
        'Column'        : r['column'],
        'Outlier Count' : r['outlier_count'],
        'Outlier %'     : r['outlier_pct'],
        'Lower Fence'   : r['lower_fence'],
        'Upper Fence'   : r['upper_fence'],
        'Min Outlier'   : r['min_outlier'],
        'Max Outlier'   : r['max_outlier'],
    } for r in results])

    total_outliers = results_df['Outlier Count'].sum()

    o1, o2, o3 = st.columns(3)
    o1.metric("Total Outliers Found",
              f"{total_outliers:,}")
    o2.metric("Columns with Outliers",
              f"{len(results_df[results_df['Outlier Count'] > 0])}")
    o3.metric("Avg Outlier Rate",
              f"{results_df['Outlier %'].mean():.1f}%")

    # ── Outlier summary bar chart ──
    outlier_cols = results_df[results_df['Outlier Count'] > 0]
    if not outlier_cols.empty:
        fig = px.bar(
            outlier_cols.sort_values('Outlier %', ascending=False),
            x='Column',
            y='Outlier %',
            color='Outlier %',
            color_continuous_scale='Oranges',
            title='Outlier % by Column (IQR Method)',
            text='Outlier %',
        )
        fig.update_traces(
            texttemplate='%{text:.1f}%',
            textposition='outside'
        )
        fig.update_layout(
            height=350,
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
        )
        fig.add_hline(
            y=5,
            line_dash='dash',
            line_color='red',
            annotation_text='>5% — investigate'
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Box plots — best visual for outliers ──
    st.markdown("**Box Plots — Visualize Outlier Distribution**")
    st.caption(
        "Each dot beyond the whiskers = an outlier. "
        "The box = middle 50% of data (IQR). "
        "Line in box = median."
    )

    # Show box plots in a 2-column grid
    cols_to_plot = [
        r['column'] for r in results
        if r['outlier_count'] > 0
    ][:6]   # max 6 box plots

    if cols_to_plot:
        ncols = 2
        nrows = (len(cols_to_plot) + 1) // 2

        fig_box = make_subplots(
            rows=nrows, cols=ncols,
            subplot_titles=cols_to_plot
        )

        for idx, col in enumerate(cols_to_plot):
            row = idx // ncols + 1
            col_n = idx % ncols + 1
            fig_box.add_trace(
                go.Box(
                    y=df[col].dropna(),
                    name=col,
                    marker_color='#3b82f6',
                    boxpoints='outliers',   # show outlier dots
                    marker=dict(
                        outliercolor='#ef4444',
                        size=4
                    ),
                    line=dict(color='#1e40af'),
                ),
                row=row, col=col_n
            )

        fig_box.update_layout(
            height=max(300, nrows * 250),
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            title_text="Box Plots — Red dots are outliers"
        )
        st.plotly_chart(fig_box, use_container_width=True)

    # ── Detailed outlier table ──
    with st.expander("📋 Full Outlier Statistics Table"):
        try:
            styled_out = results_df.style.background_gradient(
                subset=['Outlier %'], cmap='Oranges'
            )
            st.dataframe(styled_out, use_container_width=True)
        except Exception:
            st.dataframe(results_df, use_container_width=True)
        st.caption(
            "IQR Method: Values below Q1-1.5×IQR or "
            "above Q3+1.5×IQR are flagged as outliers."
        )

    # ── Column-level deep dive ──
    st.markdown("**Deep Dive — Explore Individual Columns**")
    selected_col = st.selectbox(
        "Select a column to inspect outliers:",
        options=[r['column'] for r in results
                 if r['outlier_count'] > 0],
        key="outlier_col_select"
    )

    if selected_col:
        r = next(x for x in results if x['column'] == selected_col)

        dc1, dc2, dc3, dc4 = st.columns(4)
        dc1.metric("Q1 (25th pct)",     f"{r['Q1']:,}")
        dc2.metric("Q3 (75th pct)",     f"{r['Q3']:,}")
        dc3.metric("IQR",               f"{r['IQR']:,}")
        dc4.metric("Outliers Found",    f"{r['outlier_count']:,}")

        dc5, dc6 = st.columns(2)
        dc5.metric("Lower Fence",  f"{r['lower_fence']:,}",
                   help="Values below this = low outliers")
        dc6.metric("Upper Fence",  f"{r['upper_fence']:,}",
                   help="Values above this = high outliers")

        # Distribution histogram with fence lines
        fig_hist = px.histogram(
            df, x=selected_col,
            nbins=40,
            color_discrete_sequence=['#3b82f6'],
            title=f"Distribution of {selected_col} with Outlier Fences",
        )
        fig_hist.add_vline(
            x=r['lower_fence'],
            line_dash='dash',
            line_color='#ef4444',
            annotation_text=f"Lower fence: {r['lower_fence']:,}",
            annotation_position='top right'
        )
        fig_hist.add_vline(
            x=r['upper_fence'],
            line_dash='dash',
            line_color='#ef4444',
            annotation_text=f"Upper fence: {r['upper_fence']:,}",
            annotation_position='top left'
        )
        fig_hist.update_layout(
            height=350,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
        )
        st.plotly_chart(fig_hist, use_container_width=True)

        # Show actual outlier rows
        if r['outlier_count'] > 0:
            with st.expander(
                f"🔍 View {min(r['outlier_count'], 20)} "
                f"outlier rows for {selected_col}"
            ):
                outlier_idx = r['outlier_values'].index
                st.dataframe(
                    df.loc[outlier_idx].head(20),
                    use_container_width=True
                )

    # ── AI Recommendations ──
    with st.spinner("AI analyzing outliers..."):
        # Prepare summary for AI
        outlier_summary = []
        for r in results:
            if r['outlier_count'] > 0:
                outlier_summary.append(
                    f"{r['column']}: {r['outlier_count']} outliers "
                    f"({r['outlier_pct']}%), "
                    f"range [{r['lower_fence']} to {r['upper_fence']}], "
                    f"extreme values up to {r['max_outlier']}"
                )

        prompt = f"""You are a senior data analyst reviewing outlier analysis.

Outlier detection results (IQR method):
{chr(10).join(outlier_summary) if outlier_summary else 'No significant outliers found.'}

Dataset: {len(df):,} rows, {len(num_cols)} numeric columns.

Give 3-4 specific recommendations:
- Which columns need immediate attention and why
- Whether outliers are likely data errors or genuine extreme values
- Specific treatment: cap/floor, log transform, remove, or keep
- Any business implications of these outliers
Under 120 words. Use numbered lines 1. 2. 3. Data analyst language."""

        ai_text = ask_ai_fn(prompt)
        st.markdown(
            f'<div style="background:#fef9c3;border-left:4px solid '
            f'#eab308;padding:12px 16px;border-radius:0 8px 8px 0;'
            f'font-size:0.9rem;color:#854d0e;line-height:1.7;'
            f'margin-top:8px">'
            f'🤖 <strong>AI Recommendation:</strong><br>{ai_text}</div>',
            unsafe_allow_html=True
        )

    return results_df


# ============================================================
# SECTION 4 — DATA TYPE SUMMARY
# ============================================================
def show_dtype_section(df, col_types):
    """Shows column types, unique counts, sample values."""

    st.markdown("#### 🏷️ Column Type Summary")

    rows = []
    for col in df.columns:
        col_type = (
            'Numeric'     if col in col_types['numeric'] and
                            col not in col_types['categorical']
            else 'Numeric+Category' if col in col_types['numeric'] and
                            col in col_types['categorical']
            else 'Category'   if col in col_types['categorical']
            else 'DateTime'   if col in col_types['datetime']
            else 'Text/ID'
        )
        samples = df[col].dropna().unique()[:3]
        sample_str = ', '.join([str(s)[:15] for s in samples])

        rows.append({
            'Column'      : col,
            'Detected As' : col_type,
            'Raw Dtype'   : str(df[col].dtype),
            'Unique Values': df[col].nunique(),
            'Missing %'   : f"{df[col].isnull().mean()*100:.1f}%",
            'Sample Values': sample_str,
        })

    dtype_df = pd.DataFrame(rows)

    # Color code by type
    def color_type(val):
        colors = {
            'Numeric'          : 'background-color: #dbeafe',
            'Numeric+Category' : 'background-color: #e9d5ff',
            'Category'         : 'background-color: #dcfce7',
            'DateTime'         : 'background-color: #fef9c3',
            'Text/ID'          : 'background-color: #f3f4f6',
        }
        return colors.get(val, '')

    # applymap was renamed to map in pandas 2.1+
    # We use a safe wrapper that works on both versions
    try:
        styled = dtype_df.style.map(
            color_type, subset=['Detected As']
        )
    except AttributeError:
        styled = dtype_df.style.applymap(
            color_type, subset=['Detected As']
        )
    st.dataframe(styled, use_container_width=True, height=300)

    # Legend
    leg1, leg2, leg3, leg4, leg5 = st.columns(5)
    leg1.markdown("🔵 Numeric")
    leg2.markdown("🟣 Num+Category")
    leg3.markdown("🟢 Category")
    leg4.markdown("🟡 DateTime")
    leg5.markdown("⚪ Text/ID")


# ============================================================
# MAIN — DATA QUALITY DASHBOARD
# This is the function called from app.py
# ============================================================
def show_data_quality_report(df, col_types, ask_ai_fn):
    """
    Full Data Quality Report — call this from app.py.

    Parameters:
        df         : the filtered DataFrame
        col_types  : dict from detect_columns()
        ask_ai_fn  : the ask_ai() function from app.py (passed in)
                     so we don't duplicate the Groq client setup
    """
    st.markdown(
        '<p class="section-title">🔬 Data Quality Report</p>',
        unsafe_allow_html=True
    )
    st.caption(
        "Automated scan for missing values, duplicates, outliers "
        "and data type issues — with AI-powered fix recommendations."
    )

    # ── Overall health score ──────────────────────────────
    total_cells    = df.shape[0] * df.shape[1]
    missing_pct    = df.isnull().sum().sum() / total_cells * 100
    dup_pct        = df.duplicated().sum() / len(df) * 100

    # Count columns with >5% outliers
    num_cols = [c for c in col_types['numeric']
                if c not in col_types['categorical']]
    high_outlier_cols = 0
    for col in num_cols[:8]:
        r = detect_outliers_iqr(df, col)
        if r['outlier_pct'] > 5:
            high_outlier_cols += 1

    # Simple health score: start 100, deduct for issues
    score = 100
    score -= min(missing_pct * 2, 30)     # max -30 for missing
    score -= min(dup_pct * 2, 20)         # max -20 for duplicates
    score -= min(high_outlier_cols * 5, 25) # max -25 for outliers
    score = max(0, round(score))

    # Score display
    if score >= 80:
        score_color = '#22c55e'
        score_label = 'Good'
        score_emoji = '✅'
    elif score >= 60:
        score_color = '#f59e0b'
        score_label = 'Fair'
        score_emoji = '⚠️'
    else:
        score_color = '#ef4444'
        score_label = 'Poor'
        score_emoji = '❌'

    st.markdown(
        f'<div style="background:linear-gradient(135deg,#1e3a5f,#2563eb);'
        f'color:white;padding:20px 24px;border-radius:12px;'
        f'margin-bottom:16px;display:flex;align-items:center;gap:20px">'
        f'<div style="text-align:center">'
        f'<div style="font-size:3rem;font-weight:700;'
        f'color:{score_color}">{score}</div>'
        f'<div style="font-size:0.85rem;opacity:0.8">Quality Score / 100</div>'
        f'</div>'
        f'<div>'
        f'<div style="font-size:1.3rem;font-weight:600">'
        f'{score_emoji} Data Quality: {score_label}</div>'
        f'<div style="font-size:0.85rem;opacity:0.8;margin-top:4px">'
        f'Missing: {missing_pct:.1f}% | '
        f'Duplicates: {dup_pct:.1f}% | '
        f'High-outlier columns: {high_outlier_cols}</div>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True
    )

    # ── Tabs for each section ─────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "🕳️ Missing Values",
        "👥 Duplicates",
        "📦 Outliers",
        "🏷️ Column Types",
    ])

    with tab1:
        show_missing_section(df, ask_ai_fn)

    with tab2:
        show_duplicates_section(df, ask_ai_fn)

    with tab3:
        show_outliers_section(df, col_types, ask_ai_fn)

    with tab4:
        show_dtype_section(df, col_types)