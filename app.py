# ============================================================
# AI BUSINESS ANALYST DASHBOARD — Complete (Phase 2+3+4)
# Phase 4 adds:
#   → AI Chat section where user types any business question
#   → Suggested question buttons for easy demo
#   → Auto chart generation based on the answer
#   → Follow-up question suggestions
#   → Full conversation history
# ============================================================

import streamlit as st
import pandas as pd
import plotly.express as px
from dotenv import load_dotenv
import os
from google import genai

# ── SETUP ──────────────────────────────────────────────────
load_dotenv()
# client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
api_key = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY", "")
client = genai.Client(api_key=api_key)
MODEL  = "gemini-2.0-flash"

st.set_page_config(
    page_title="AI Business Analyst",
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
    /* User message bubble — right aligned feel, blue tint */
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
    /* AI message bubble — left aligned feel, grey tint */
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
    /* Follow-up question pill buttons */
    .followup-label {
        font-size: 0.78rem;
        color: #6b7280;
        margin: 0.5rem 0 0.3rem 0;
        font-style: italic;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# CORE HELPERS
# ============================================================

def ask_gemini(prompt):
    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=prompt
        )
        return response.text
    except Exception as e:
        return f"AI temporarily unavailable: {str(e)}"


def load_data(uploaded_file):
    df = pd.read_csv(uploaded_file)
    if 'Order_Date' in df.columns:
        df['Order_Date'] = pd.to_datetime(df['Order_Date'])
    return df


def get_data_summary(df):
    """
    Text description of the dataset passed to AI as context.
    Pandas does ALL calculations — AI only reads the results.
    """
    top_region    = df.groupby('Region')['Sales'].sum().idxmax()
    worst_region  = df.groupby('Region')['Profit'].sum().idxmin()
    top_cat_sales = df.groupby('Category')['Sales'].sum().idxmax()
    top_cat_profit= df.groupby('Category')['Profit'].sum().idxmax()
    worst_cat     = df.groupby('Category')['Profit'].sum().idxmin()
    top_segment   = df.groupby('Segment')['Sales'].sum().idxmax()

    yearly_sales  = df.groupby('Year')['Sales'].sum()
    if len(yearly_sales) >= 2:
        growth = ((yearly_sales.iloc[-1] - yearly_sales.iloc[0])
                  / yearly_sales.iloc[0] * 100)
        growth_str = f"{growth:.1f}%"
    else:
        growth_str = "N/A"

    return f"""
    BUSINESS DATA SUMMARY (Indian Retail Company):
    Total Orders       : {len(df):,}
    Date Range         : {df['Order_Date'].min().strftime('%b %Y')} to {df['Order_Date'].max().strftime('%b %Y')}
    Total Sales        : ₹{df['Sales'].sum()/1e7:.2f} Crores
    Total Profit       : ₹{df['Profit'].sum()/1e5:.1f} Lakhs
    Profit Margin      : {(df['Profit'].sum()/df['Sales'].sum()*100):.1f}%
    Avg Order Value    : ₹{df['Sales'].mean():,.0f}
    Overall YoY Growth : {growth_str}
    Top Region (Sales) : {top_region}
    Weak Region (Profit): {worst_region}
    Top Category (Sales): {top_cat_sales}
    Top Category (Profit): {top_cat_profit}
    Worst Category      : {worst_cat}
    Avg Discount Given  : {df['Discount_Percent'].mean():.1f}%
    Top Segment (Sales) : {top_segment}
    Segments Present    : {', '.join(df['Segment'].unique())}
    Regions Present     : {', '.join(df['Region'].unique())}
    Categories Present  : {', '.join(df['Category'].unique())}

    REGIONAL BREAKDOWN:
    {df.groupby('Region').agg(Sales=('Sales','sum'), Profit=('Profit','sum'), Orders=('Order_ID','count')).round(0).to_string()}

    CATEGORY BREAKDOWN:
    {df.groupby('Category').agg(Sales=('Sales','sum'), Profit=('Profit','sum'), Avg_Discount=('Discount_Percent','mean')).round(1).to_string()}

    SEGMENT BREAKDOWN:
    {df.groupby('Segment').agg(Sales=('Sales','sum'), Profit=('Profit','sum'), Orders=('Order_ID','count')).round(0).to_string()}

    YEARLY BREAKDOWN:
    {df.groupby('Year').agg(Sales=('Sales','sum'), Profit=('Profit','sum')).round(0).to_string()}
    """


# ============================================================
# PHASE 2 — KPI CARDS
# ============================================================

def generate_kpi_cards(df):
    st.markdown('<p class="section-title">📈 Business KPIs</p>',
                unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    total_sales   = df['Sales'].sum()
    total_profit  = df['Profit'].sum()
    profit_margin = (total_profit / total_sales) * 100
    avg_order_val = total_sales / len(df)
    with col1: st.metric("💰 Total Sales",      f"₹{total_sales/1e7:.2f} Cr")
    with col2: st.metric("📊 Total Profit",     f"₹{total_profit/1e5:.1f} L")
    with col3: st.metric("📉 Profit Margin",    f"{profit_margin:.1f}%")
    with col4: st.metric("🛒 Avg Order Value",  f"₹{avg_order_val:,.0f}")


# ============================================================
# PHASE 3 — AI EXECUTIVE SUMMARY
# ============================================================

def generate_ai_summary(df):
    st.markdown('<p class="section-title">🤖 AI Executive Summary</p>',
                unsafe_allow_html=True)
    with st.spinner("Gemini AI is analyzing your data..."):
        prompt = f"""
        You are a senior data analyst presenting to leadership of an Indian retail company.

        Based on this business data:
        {get_data_summary(df)}

        Write a concise executive summary (4-5 sentences):
        1. Open with overall business performance (revenue + growth)
        2. Highlight the strongest region and category by name
        3. Flag one specific concern with a number to support it
        4. End with one clear, actionable recommendation

        Rules: Indian currency format (Crores/Lakhs). No bullet points.
        Flowing paragraph. Mention actual names. Under 120 words.
        """
        st.markdown(
            f'<div class="insight-box">{ask_gemini(prompt)}</div>',
            unsafe_allow_html=True
        )


# ============================================================
# PHASE 3 — CHARTS
# ============================================================

def show_charts(df):
    st.markdown('<p class="section-title">📊 Visual Analysis</p>',
                unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        region_data = df.groupby('Region')[['Sales','Profit']].sum().reset_index()
        fig1 = px.bar(region_data, x='Region', y='Sales', color='Profit',
                      color_continuous_scale='Greens',
                      title='💹 Sales & Profit by Region', text_auto='.2s')
        fig1.update_layout(height=370, showlegend=False)
        st.plotly_chart(fig1, use_container_width=True)
        top_r = region_data.loc[region_data['Sales'].idxmax(), 'Region']
        low_r = region_data.loc[region_data['Profit'].idxmin(), 'Region']
        st.caption("🤖 " + ask_gemini(
            f"One sentence for a manager: top sales region is {top_r} "
            f"but lowest profit region is {low_r}. Under 35 words."
        ))

    with col2:
        df_copy = df.copy()
        df_copy['YearMonth'] = df_copy['Order_Date'].dt.to_period('M').astype(str)
        monthly = df_copy.groupby('YearMonth')['Sales'].sum().reset_index()
        fig2 = px.line(monthly, x='YearMonth', y='Sales',
                       title='📅 Monthly Sales Trend', markers=True)
        fig2.update_layout(height=370)
        fig2.update_xaxes(tickangle=45)
        st.plotly_chart(fig2, use_container_width=True)
        peak = monthly.loc[monthly['Sales'].idxmax(), 'YearMonth']
        st.caption("🤖 " + ask_gemini(
            f"One sentence for a manager: sales peaked in {peak}. "
            f"Latest month: ₹{monthly['Sales'].iloc[-1]:,.0f}. Under 35 words."
        ))

    col3, col4 = st.columns(2)

    with col3:
        cat_data = df.groupby('Category')['Sales'].sum().reset_index()
        fig3 = px.pie(cat_data, values='Sales', names='Category',
                      title='🏷️ Sales Share by Category', hole=0.45,
                      color_discrete_sequence=px.colors.qualitative.Set2)
        fig3.update_layout(height=370)
        st.plotly_chart(fig3, use_container_width=True)
        top_cat = cat_data.loc[cat_data['Sales'].idxmax(), 'Category']
        top_pct = cat_data['Sales'].max() / cat_data['Sales'].sum() * 100
        st.caption("🤖 " + ask_gemini(
            f"One sentence: {top_cat} is {top_pct:.1f}% of sales. "
            f"Risk or healthy? Under 35 words."
        ))

    with col4:
        subcat = df.groupby('Sub_Category').agg(
            Avg_Discount=('Discount_Percent','mean'),
            Total_Profit=('Profit','sum'),
            Total_Sales=('Sales','sum')
        ).reset_index()
        fig4 = px.scatter(subcat, x='Avg_Discount', y='Total_Profit',
                          size='Total_Sales', color='Total_Profit',
                          color_continuous_scale='RdYlGn',
                          hover_name='Sub_Category',
                          title='🎯 Discount % vs Profit by Sub-Category')
        fig4.update_layout(height=370, showlegend=False)
        fig4.add_hline(y=0, line_dash="dash", line_color="red",
                       annotation_text="Break-even")
        st.plotly_chart(fig4, use_container_width=True)
        loss_n = len(subcat[subcat['Total_Profit'] < 0])
        hi_disc = subcat.nlargest(1,'Avg_Discount')['Sub_Category'].values[0]
        st.caption("🤖 " + ask_gemini(
            f"One sentence: {hi_disc} has highest avg discount. "
            f"{loss_n} sub-categories are loss-making. Action? Under 35 words."
        ))


# ============================================================
# ★ PHASE 4 — AI CHAT SECTION
#
# HOW THE WHOLE THING WORKS (learn this cold):
#
#   1. User types a question OR clicks a suggestion button
#   2. We compute fresh stats from the dataframe using Pandas
#   3. We build a detailed prompt: question + all the data stats
#   4. Gemini reads the data and writes a business answer
#   5. We detect what type of chart fits the answer
#   6. We auto-generate the chart using Plotly
#   7. We ask Gemini for 3 follow-up question suggestions
#   8. We save everything to session_state (chat history)
#   9. We display the full conversation on screen
#
# KEY CONCEPT — st.session_state:
#   Streamlit reruns the ENTIRE script top to bottom every time
#   the user does anything (types, clicks, uploads).
#   Without session_state, all variables reset on every rerun.
#   session_state = persistent storage that survives reruns.
#   Think of it like RAM for your Streamlit app.
# ============================================================

def detect_chart_type(question, result_df):
    """
    Decides which chart to show based on what the user asked.

    WHY THIS FUNCTION:
    Different questions need different charts.
    "Compare regions" → bar chart
    "Show trend"      → line chart
    "Show share"      → donut chart
    We detect the intent from keywords in the question.

    This is a simple rule-based classifier — no AI needed here
    because it's a structured decision, not a language task.
    """
    q = question.lower()

    # Time/trend keywords → line chart
    if any(word in q for word in
           ['trend', 'month', 'year', 'over time', 'growth',
            'quarterly', 'weekly', 'timeline']):
        return 'line'

    # Proportion keywords → donut chart
    if any(word in q for word in
           ['share', 'proportion', 'percent', '%',
            'contribution', 'breakdown', 'distribution']):
        return 'donut'

    # Default → bar chart (works for most comparisons)
    return 'bar'


def generate_answer_chart(question, result_df):
    """
    Auto-generates a Plotly chart from the query result DataFrame.

    HOW IT WORKS:
    After Pandas computes the answer, result_df is a small summary table.
    We look at the column types to figure out what to plot:
      - First text column  → x-axis (categories)
      - First number column → y-axis (values)
    Then we pick the chart type from detect_chart_type().

    This is called "dynamic chart generation" — the chart builds
    itself based on the data shape, not hardcoded column names.
    """
    if result_df is None or result_df.empty:
        return None

    try:
        # Find which columns are text (categories) and which are numbers
        # select_dtypes() filters columns by data type
        text_cols   = result_df.select_dtypes(include=['object']).columns.tolist()
        number_cols = result_df.select_dtypes(include=['number']).columns.tolist()

        # We need at least one text column and one number column to make a chart
        if not text_cols or not number_cols:
            return None

        x_col = text_cols[0]    # first text column = x-axis
        y_col = number_cols[0]  # first number column = y-axis

        chart_type = detect_chart_type(question, result_df)

        if chart_type == 'line':
            fig = px.line(result_df, x=x_col, y=y_col, markers=True,
                          title=f"📈 {y_col} by {x_col}")

        elif chart_type == 'donut':
            fig = px.pie(result_df, names=x_col, values=y_col,
                         hole=0.45, title=f"🍩 {y_col} by {x_col}",
                         color_discrete_sequence=px.colors.qualitative.Set2)

        else:  # bar (default)
            fig = px.bar(result_df, x=x_col, y=y_col,
                         color=y_col, color_continuous_scale='Blues',
                         title=f"📊 {y_col} by {x_col}",
                         text_auto='.2s')

        fig.update_layout(height=320, showlegend=False)
        return fig

    except Exception:
        # If chart fails for any reason, just return None
        # The answer text will still show — chart is bonus
        return None


def compute_question_data(question, df):
    """
    Runs the most relevant Pandas calculation based on the question.

    WHY THIS EXISTS:
    The AI gets better, more accurate answers when we give it
    pre-computed data tables instead of raw CSV data.
    We detect keywords and run the matching Pandas groupby.

    This function returns TWO things:
      1. stats_text  → text table pasted into the AI prompt
      2. result_df   → DataFrame used to generate the chart
    """
    q = question.lower()

    # ── REGION questions ─────────────────────────────────
    if any(w in q for w in ['region', 'north', 'south', 'east', 'west']):
        result_df = df.groupby('Region').agg(
            Sales=('Sales','sum'),
            Profit=('Profit','sum'),
            Orders=('Order_ID','count'),
            Margin=('Profit', lambda x: round(x.sum()/df.loc[x.index,'Sales'].sum()*100, 1))
        ).reset_index().sort_values('Sales', ascending=False)
        return result_df.to_string(index=False), result_df

    # ── CATEGORY questions ────────────────────────────────
    elif any(w in q for w in ['category', 'categories', 'electronics',
                               'furniture', 'clothing', 'office', 'kitchen']):
        result_df = df.groupby('Category').agg(
            Sales=('Sales','sum'),
            Profit=('Profit','sum'),
            Avg_Discount=('Discount_Percent','mean'),
            Orders=('Order_ID','count')
        ).reset_index().sort_values('Sales', ascending=False)
        return result_df.to_string(index=False), result_df

    # ── SEGMENT questions ─────────────────────────────────
    elif any(w in q for w in ['segment', 'consumer', 'corporate',
                               'home office', 'small business', 'customer']):
        result_df = df.groupby('Segment').agg(
            Sales=('Sales','sum'),
            Profit=('Profit','sum'),
            Orders=('Order_ID','count')
        ).reset_index().sort_values('Sales', ascending=False)
        return result_df.to_string(index=False), result_df

    # ── YEAR / TREND questions ────────────────────────────
    elif any(w in q for w in ['year', 'yearly', 'annual', 'growth',
                               'trend', 'over time', '2021','2022','2023','2024']):
        result_df = df.groupby('Year').agg(
            Sales=('Sales','sum'),
            Profit=('Profit','sum'),
            Orders=('Order_ID','count')
        ).reset_index()
        # Add YoY growth % column
        result_df['YoY_Growth_%'] = result_df['Sales'].pct_change() * 100
        result_df['YoY_Growth_%'] = result_df['YoY_Growth_%'].round(1)
        return result_df.to_string(index=False), result_df

    # ── MONTH / SEASONAL questions ────────────────────────
    elif any(w in q for w in ['month', 'monthly', 'quarter',
                               'seasonal', 'festival', 'diwali']):
        df_copy = df.copy()
        df_copy['YearMonth'] = df_copy['Order_Date'].dt.to_period('M').astype(str)
        result_df = df_copy.groupby('YearMonth').agg(
            Sales=('Sales','sum'),
            Orders=('Order_ID','count')
        ).reset_index()
        return result_df.to_string(index=False), result_df

    # ── DISCOUNT questions ────────────────────────────────
    elif any(w in q for w in ['discount', 'offer', 'deal', 'margin', 'profit']):
        result_df = df.groupby('Category').agg(
            Avg_Discount=('Discount_Percent','mean'),
            Total_Profit=('Profit','sum'),
            Sales=('Sales','sum')
        ).reset_index().sort_values('Avg_Discount', ascending=False)
        return result_df.to_string(index=False), result_df

    # ── PRODUCT / SUB-CATEGORY questions ─────────────────
    elif any(w in q for w in ['product', 'sub', 'item', 'best selling',
                               'worst', 'top product', 'laptop', 'phone']):
        result_df = df.groupby('Sub_Category').agg(
            Sales=('Sales','sum'),
            Profit=('Profit','sum'),
            Orders=('Order_ID','count')
        ).reset_index().sort_values('Sales', ascending=False).head(10)
        return result_df.to_string(index=False), result_df

    # ── PAYMENT questions ─────────────────────────────────
    elif any(w in q for w in ['payment', 'upi', 'credit', 'debit',
                               'cash', 'net banking']):
        result_df = df.groupby('Payment_Mode').agg(
            Sales=('Sales','sum'),
            Orders=('Order_ID','count')
        ).reset_index().sort_values('Sales', ascending=False)
        return result_df.to_string(index=False), result_df

    # ── DEFAULT: general summary ──────────────────────────
    else:
        result_df = df.groupby('Category').agg(
            Sales=('Sales','sum'),
            Profit=('Profit','sum')
        ).reset_index()
        return get_data_summary(df), result_df


def generate_followup_questions(question, answer):
    """
    Asks Gemini to suggest 3 natural follow-up questions.

    WHY THIS FEATURE:
    Follow-up suggestions guide the user to explore more data
    and show off more of the app's capabilities during a demo.
    When a recruiter is watching, these suggestions make the app
    feel like a complete product, not a student project.

    TECHNIQUE — asking AI to return structured data:
    We ask Gemini to return ONLY a numbered list with no extra text.
    Then we parse the response by splitting on newlines.
    This is a simple form of structured AI output.
    """
    prompt = f"""
    A user asked a data analyst chatbot: "{question}"
    The chatbot answered with business insights about Indian retail data.

    Suggest exactly 3 short follow-up questions the user might ask next.
    Rules:
    - Each question on its own line, numbered: 1. 2. 3.
    - Each question under 10 words
    - About sales, profit, region, category, segment, or trends
    - No explanations, no extra text — just the 3 numbered questions
    """
    raw = ask_gemini(prompt)

    # Parse the numbered list into a Python list
    # splitlines() splits text by line breaks
    # strip() removes extra spaces
    # We filter only lines that start with a number
    questions = []
    for line in raw.strip().splitlines():
        line = line.strip()
        if line and line[0].isdigit():
            # Remove the number and dot prefix: "1. Question?" → "Question?"
            clean = line[2:].strip() if len(line) > 2 else line
            if clean:
                questions.append(clean)

    # Return max 3 questions
    return questions[:3]


def ai_chat_section(df):
    """
    The main Phase 4 function.
    Renders the complete chat interface.
    """
    st.markdown('<p class="section-title">💬 Ask the AI Analyst</p>',
                unsafe_allow_html=True)

    # ── SUGGESTED QUESTION BUTTONS ───────────────────────
    # These buttons help users know what the app can do
    # They're especially useful during demos — one click shows everything
    st.markdown("**Try asking:**")

    suggestions = [
        "Which region has highest profit?",
        "What is our worst performing category?",
        "How did sales grow year over year?",
        "Which customer segment brings most revenue?",
        "Which sub-category has the most discount?",
        "Show monthly sales trend",
    ]

    # Display buttons in 3 columns, 2 per row
    # We use a key= argument so each button has a unique ID
    cols = st.columns(3)
    for i, suggestion in enumerate(suggestions):
        if cols[i % 3].button(suggestion, key=f"sug_{i}",
                               use_container_width=True):
            # When clicked, store the question in session_state
            # The app reruns, picks it up below, and processes it
            st.session_state['pending_question'] = suggestion

    st.markdown("---")

    # ── INITIALIZE SESSION STATE ─────────────────────────
    # session_state variables survive across Streamlit reruns
    # We check if they exist first — only create them once

    # chat_history: list of dicts, each with role + content
    # [{'role': 'user', 'content': '...'}, {'role': 'ai', ...}]
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []

    # chart_history: stores charts matching each AI response
    if 'chart_history' not in st.session_state:
        st.session_state['chart_history'] = []

    # followup_history: stores follow-up suggestions per response
    if 'followup_history' not in st.session_state:
        st.session_state['followup_history'] = []

    # pending_question: set by suggestion buttons or follow-up clicks
    if 'pending_question' not in st.session_state:
        st.session_state['pending_question'] = None

    # ── CHAT INPUT BOX ───────────────────────────────────
    # st.chat_input() = the text box at the bottom of a chat
    # It returns None if user hasn't submitted anything
    # It returns the text string when user presses Enter
    typed_question = st.chat_input(
        "Ask any business question about your data..."
    )

    # ── DETERMINE ACTIVE QUESTION ────────────────────────
    # Priority: typed question > pending (from button click)
    active_question = None

    if typed_question:
        active_question = typed_question
        st.session_state['pending_question'] = None

    elif st.session_state.get('pending_question'):
        active_question = st.session_state['pending_question']
        st.session_state['pending_question'] = None

    # ── PROCESS THE QUESTION ─────────────────────────────
    if active_question:

        # Step 1: Add user question to chat history
        st.session_state['chat_history'].append({
            'role' : 'user',
            'content': active_question
        })

        # Step 2: Compute relevant data using Pandas
        # This gives the AI accurate numbers to work with
        with st.spinner("Analyzing your data..."):
            stats_text, result_df = compute_question_data(
                active_question, df
            )

            # Step 3: Build the AI prompt
            # We give Gemini:
            #   - its role
            #   - the full dataset summary
            #   - the specific computed stats for this question
            #   - the user's exact question
            #   - strict formatting rules
            prompt = f"""
            You are an expert data analyst at an Indian retail company.
            Answer the analyst's question using the data provided below.

            FULL DATASET SUMMARY:
            {get_data_summary(df)}

            SPECIFIC DATA FOR THIS QUESTION:
            {stats_text}

            ANALYST'S QUESTION: "{active_question}"

            ANSWER RULES:
            - First sentence: direct answer with the key number
            - Then 2-3 supporting facts from the specific data above
            - End with one business recommendation
            - Use ₹ and Indian format (Lakhs/Crores)
            - Under 100 words total
            - No bullet points — natural paragraph form
            - Be specific: use actual names from the data
            """

            ai_answer = ask_gemini(prompt)

            # Step 4: Generate the supporting chart
            chart = generate_answer_chart(active_question, result_df)

            # Step 5: Generate follow-up question suggestions
            followups = generate_followup_questions(
                active_question, ai_answer
            )

        # Step 6: Save everything to session state
        st.session_state['chat_history'].append({
            'role'   : 'ai',
            'content': ai_answer
        })
        st.session_state['chart_history'].append(chart)
        st.session_state['followup_history'].append(followups)

    # ── DISPLAY CHAT HISTORY ─────────────────────────────
    # We walk through the history list and render each message
    # The chart and follow-ups align with each AI response
    ai_response_index = 0

    for message in st.session_state['chat_history']:

        if message['role'] == 'user':
            # User message — blue bubble
            st.markdown(
                f'<div class="chat-user">🧑 You: {message["content"]}</div>',
                unsafe_allow_html=True
            )

        else:
            # AI message — grey bubble
            st.markdown(
                f'<div class="chat-ai">🤖 AI Analyst: {message["content"]}</div>',
                unsafe_allow_html=True
            )

            # Show the auto-generated chart below the AI answer
            if ai_response_index < len(st.session_state['chart_history']):
                chart = st.session_state['chart_history'][ai_response_index]
                if chart is not None:
                    st.plotly_chart(chart, use_container_width=True)

            # Show follow-up question buttons
            if ai_response_index < len(st.session_state['followup_history']):
                followups = st.session_state['followup_history'][ai_response_index]
                if followups:
                    st.markdown(
                        '<p class="followup-label">💡 You might also ask:</p>',
                        unsafe_allow_html=True
                    )
                    fu_cols = st.columns(len(followups))
                    for j, fq in enumerate(followups):
                        if fu_cols[j].button(
                            fq,
                            key=f"fu_{ai_response_index}_{j}",
                            use_container_width=True
                        ):
                            st.session_state['pending_question'] = fq
                            st.rerun()  # immediately rerun to process it

            ai_response_index += 1

    # ── CLEAR CHAT BUTTON ────────────────────────────────
    if st.session_state['chat_history']:
        st.markdown("---")
        if st.button("🗑️ Clear conversation", key="clear_chat"):
            # Reset all three history lists at once
            st.session_state['chat_history']   = []
            st.session_state['chart_history']  = []
            st.session_state['followup_history'] = []
            st.rerun()


# ============================================================
# MAIN — WIRES EVERYTHING TOGETHER
# ============================================================

def main():
    st.title("📊 AI Business Analyst Dashboard")
    st.caption("Upload your sales CSV and get instant AI-powered insights")

    with st.sidebar:
        st.markdown("### 📂 Upload Your Data")
        uploaded_file = st.file_uploader(
            "Upload CSV file", type=['csv'],
            help="Upload superstore_india_sales.csv"
        )

        if uploaded_file is not None:
            df_raw = load_data(uploaded_file)
            st.markdown("---")
            st.markdown("### 🔽 Filters")

            years = sorted(df_raw['Year'].unique().tolist())
            selected_years = st.multiselect(
                "📅 Years", options=years, default=years
            )
            regions = sorted(df_raw['Region'].unique().tolist())
            selected_regions = st.multiselect(
                "🗺️ Regions", options=regions, default=regions
            )
            categories = sorted(df_raw['Category'].unique().tolist())
            selected_categories = st.multiselect(
                "🏷️ Categories", options=categories, default=categories
            )

            st.markdown("---")
            st.info(
                f"Rows: {len(df_raw):,}\n"
                f"Period: {df_raw['Year'].min()}–{df_raw['Year'].max()}\n"
                f"Columns: {len(df_raw.columns)}"
            )

    if uploaded_file is None:
        st.markdown("---")
        st.info("👈 Upload your CSV from the sidebar to get started")
        c1, c2, c3 = st.columns(3)
        with c1: st.success("**Step 1:** Upload CSV from sidebar")
        with c2: st.success("**Step 2:** Use filters to slice data")
        with c3: st.success("**Step 3:** Ask any business question")
        return

    df = df_raw[
        (df_raw['Year'].isin(selected_years)) &
        (df_raw['Region'].isin(selected_regions)) &
        (df_raw['Category'].isin(selected_categories))
    ]

    if df.empty:
        st.warning("⚠️ No data matches your filters. Please adjust.")
        return

    st.caption(
        f"Analyzing **{len(df):,}** orders | "
        f"Filtered from {len(df_raw):,} total records"
    )

    with st.expander("🔍 View Raw Data Table"):
        st.dataframe(df.head(100), use_container_width=True)

    st.markdown("---")
    generate_kpi_cards(df)
    st.markdown("---")
    generate_ai_summary(df)
    st.markdown("---")
    show_charts(df)
    st.markdown("---")
    ai_chat_section(df)         # ← Phase 4 added here

    st.markdown("---")
    st.caption(
        "Built with Streamlit · Pandas · Plotly · Google Gemini 2.5 Flash"
    )


if __name__ == "__main__":
    main()