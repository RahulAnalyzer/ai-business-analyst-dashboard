# 📊 AI Data Analyst Dashboard

> Upload **any** CSV file and instantly get KPIs, AI insights, business charts, data quality report, and a downloadable PDF — powered by Groq LLaMA 3.3 70B.

🔗 **[Live Demo → ai-business-analyst.streamlit.app](https://ai-business-analysis.streamlit.app/)**

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red?style=flat-square&logo=streamlit)
![Groq](https://img.shields.io/badge/Groq-LLaMA_3.3_70B-orange?style=flat-square)
![Pandas](https://img.shields.io/badge/Pandas-2.0+-green?style=flat-square&logo=pandas)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

---

## 🎯 What It Does

This tool works like an AI data analyst sitting next to you. Upload any CSV — sales data, HR data, medical data, finance data — and the app instantly:

- **Calculates KPIs** automatically from whatever numeric columns exist
- **Writes an executive summary** in professional business language
- **Generates 4 interactive charts** — bar, trend line, donut, scatter
- **Runs a data quality scan** — missing values, duplicates, outliers
- **Answers business questions** in plain English with supporting charts
- **Exports a PDF report** with everything compiled in one document

---

## ✨ Features

### 📈 Auto KPI Cards
Detects all numeric columns and displays totals, averages, and max values — formatted in Indian number format (Crores/Lakhs) or standard K/M format.

### 🤖 AI Executive Summary
Gemini reads the dataset profile and writes a 4-5 sentence business narrative — identifying top performers, flagging concerns, and giving one actionable recommendation.

### 📊 4 Dynamic Business Charts
Charts are built from whatever columns exist — no hardcoded assumptions:
| Chart | Type | Business Question Answered |
|---|---|---|
| Primary metric by category | Bar | Who is performing best? |
| Metric over time | Line | Is the business growing? |
| Category share | Donut | What % does each group contribute? |
| Two metrics relationship | Scatter | Do these two things correlate? |

### 🔬 Automated Data Quality Report
A complete data health scan with a **Quality Score (0–100)**:
- **Missing Values** — % missing per column + pattern heatmap + AI fix recommendation
- **Duplicate Rows** — count, %, sample rows, one-click remove
- **Outlier Detection** — IQR method with box plots, fence lines, and column deep-dive
- **Column Type Summary** — auto-detected types with sample values

### 💬 AI Chat — Ask Anything
Type any business question in plain English. The app:
1. Detects which columns are relevant
2. Computes the answer using Pandas (accurate numbers)
3. Asks Groq to write the business narrative
4. Auto-generates a supporting chart
5. Suggests 3 follow-up questions

### 📄 One-Click PDF Export
Exports a 4-page professional report:
- Page 1: Dataset overview + KPI cards + AI summary + stats table
- Page 2: All 4 business charts
- Page 3: Chat Q&A history (every question + answer from the session)
- Page 4: First 20 rows data sample

### 🔽 Dynamic Filters
Auto-detects categorical columns (including numeric columns with few unique values like status codes, ratings, flags) and builds sidebar filters — no configuration needed.

---

## 🏗️ Architecture

```
User uploads CSV
       ↓
Pandas — auto-detects column types (numeric / categorical / datetime)
       ↓
Pandas — computes all aggregations, KPIs, quality metrics (accurate)
       ↓
Groq LLaMA 3.3 70B — receives pre-computed data as context
       ↓
Groq — writes narrative (never calculates — only explains)
       ↓
Plotly — renders interactive charts from Pandas results
       ↓
ReportLab + Matplotlib — compiles everything into PDF
```

**Key design decision:** Pandas handles ALL calculations. Groq handles ALL communication. This separation guarantees accuracy while producing professional business language — and eliminates AI hallucination risk entirely.

---

## 🛠️ Tech Stack

| Layer | Tool | Why |
|---|---|---|
| UI | Streamlit | Rapid dashboard building in pure Python |
| Data Engine | Pandas | groupby, agg, IQR, correlation — all analysis |
| Charts (web) | Plotly Express | Interactive, zoomable, hoverable |
| Charts (PDF) | Matplotlib | Universal, no extra dependencies |
| AI | Groq LLaMA 3.3 70B | 14,400 free req/day, GPT-4 quality |
| PDF | ReportLab | Professional multi-page report generation |
| Secrets | python-dotenv | Secure API key management |
| Deploy | Streamlit Cloud | Free hosting, auto-deploy from GitHub |

---

## 🚀 Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/ai-business-analyst-dashboard.git
cd ai-business-analyst-dashboard

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your free Groq API key
echo "GROQ_API_KEY=your_key_here" > .env

# 5. Run
streamlit run app.py
```

Get your **free** Groq API key (no credit card) at [console.groq.com](https://console.groq.com)

---

## 📁 Project Structure

```
ai-business-analyst-dashboard/
├── app.py                  # Main app — all phases wired together
├── data_quality.py         # Data Quality Report module
├── pdf_export.py           # PDF generation module
├── requirements.txt        # Dependencies
├── .gitignore              # Protects .env and secrets
└── README.md               # This file
```

---

## 📊 Works With Any CSV

| Dataset Type | Example Columns | What It Shows |
|---|---|---|
| Sales | Region, Category, Sales, Profit | Revenue by region, profit trends |
| HR | Department, Salary, Experience | Headcount by dept, salary distribution |
| Medical | Diagnosis, Age, Procedures | Patient metrics, procedure frequency |
| Finance | Month, Budget, Actual, Variance | Budget vs actual, cost trends |
| Marketing | Channel, Spend, Conversions | ROI by channel, campaign trends |

---

## 💡 Key Technical Decisions

**Why Groq instead of OpenAI?**
Groq provides 14,400 free requests per day with no credit card required. For a portfolio project, this means zero cost with enough capacity for real demos.

**Why separate Pandas computation from AI narration?**
AI models can hallucinate numbers. By computing everything with Pandas first and passing results as context, the AI only handles language — guaranteeing factual accuracy.

**Why IQR for outlier detection instead of Z-score?**
IQR is robust to skewed distributions — it doesn't assume normal distribution. Z-score can miss outliers in skewed data or flag too many in heavy-tailed distributions. IQR is the standard in exploratory data analysis.

**Why ReportLab + Matplotlib for PDF instead of Plotly + kaleido?**
kaleido (needed to export Plotly charts as images) has inconsistent installation across platforms and Python versions. Matplotlib is pre-installed everywhere and produces clean static charts — the right tool for PDF rendering.

---

## 🎓 What I Learned Building This

- **Prompt Engineering** — Structured prompts with role, context, format constraints, and length limits produce consistent, professional AI output
- **Generic Data Architecture** — Using `select_dtypes()` and uniqueness ratios to classify any CSV without hardcoded column names
- **Session State Management** — Persisting chat history and AI summaries across Streamlit reruns
- **Pandas-AI Separation** — The most important pattern: compute with code, communicate with AI
- **Dependency Management** — Choosing libraries based on deployment compatibility, not just features
- **Breaking Change Handling** — Migrated from deprecated `google-generativeai` to new `google-genai` SDK, and handled `pandas.Styler.applymap` → `.map` deprecation

---

## 🔮 Planned Features

- [ ] Forecasting — predict next 3 months using linear regression
- [ ] Correlation Heatmap — show relationships between all numeric columns
- [ ] Anomaly Detection — flag unusual rows with Z-score + IQR
- [ ] Compare Two CSVs — side-by-side dataset comparison
- [ ] Google Sheets Integration — paste URL, read live data

---

## 👤 About

Built by **[Rahul Singh]** — Aspiring Data Analyst

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat-square&logo=linkedin)](www.linkedin.com/in/rahul-singh-analyst)

---

## 📄 License

MIT License — free to use, modify, and distribute.
