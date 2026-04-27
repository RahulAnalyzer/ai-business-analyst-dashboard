# 📊 AI Business Analyst Dashboard

An interactive business intelligence dashboard powered by **Google Gemini 2.5 Flash AI** that lets anyone upload a sales CSV and instantly get KPI summaries, visual analysis, and plain-English answers to business questions.

🔗 **[Live Demo →](https://ai-business-analyst.streamlit.app)**

---

## 🎯 What It Does

| Feature | Description |
|---|---|
| 📈 KPI Cards | Auto-calculates Total Sales, Profit, Margin, Avg Order Value |
| 🤖 AI Executive Summary | Gemini writes a business narrative in analyst language |
| 📊 4 Business Charts | Region bar, monthly trend line, category donut, discount scatter |
| 💬 AI Chat | Ask any business question — get data-backed answers + auto chart |
| 💡 Follow-up Suggestions | AI suggests the next 3 questions after every answer |

---

## 🛠️ Tech Stack

| Layer | Tool | Why |
|---|---|---|
| UI | Streamlit | Rapid dashboard building in pure Python |
| Data | Pandas | Groupby, filtering, aggregation — all analysis |
| Charts | Plotly Express | Interactive, hoverable, zoomable charts |
| AI | Google Gemini 2.5 Flash | Free API, fast responses, business narrative generation |
| Secrets | python-dotenv | Secure API key management |

---

## 🧠 Architecture — How AI Is Used

```
User Question
     ↓
Pandas computes accurate stats (groupby, agg, pct_change)
     ↓
Stats passed as context to Gemini AI
     ↓
Gemini writes business narrative (never calculates — only explains)
     ↓
Plotly generates chart from Pandas result
     ↓
Gemini suggests 3 follow-up questions
```

> **Key design decision:** Pandas handles ALL calculations. Gemini handles ALL communication.
> This separation guarantees accuracy while delivering business-ready language.

---

## 🚀 Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/RahulAnalyzer/ai-business-analyst-dashboard.git
cd ai-business-analyst-dashboard

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your Gemini API key
echo "GEMINI_API_KEY=your_key_here" > .env

# 5. Run the app
streamlit run app.py
```

Get your **free** Gemini API key at [aistudio.google.com](https://aistudio.google.com)

---

## 📁 Project Structure

```
ai-business-analyst-dashboard/
├── app.py                        # Main application (all phases)
├── requirements.txt              # Python dependencies
├── .gitignore                    # Protects secrets from GitHub
├── README.md                     # This file
└── data/
    └── superstore_india_sales.csv  # Sample dataset
```

---

## 📊 Sample Dataset

The included dataset simulates 4 years (2021–2024) of Indian retail sales:
- **3,000 orders** across 5 categories and 4 regions
- **22 columns** including Sales, Profit, Discount, Region, Segment, Payment Mode
- Built to answer real business questions a DA faces daily

---

## 💡 Key Learnings & Interview Points

- **Prompt Engineering** — Structured prompts with role, context, format rules, and constraints
- **Separation of Concerns** — Pandas for computation, AI for communication
- **Session State Management** — Persistent chat history across Streamlit reruns
- **Dynamic Chart Generation** — Auto chart type detection from question keywords
- **Secure Credential Handling** — Environment variables + Streamlit secrets

---

## 👤 Built By

**Rahul Singh** — Aspiring Data Analyst  
[LinkedIn](www.linkedin.com/in/rahul-singh-analyst) · [GitHub](https://github.com/RahulAnalyzer)