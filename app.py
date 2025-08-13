import streamlit as st
from dotenv import load_dotenv
import os
from groq import Groq
import json, time, re
import pandas as pd
from PyPDF2 import PdfReader



# Helpers
@st.cache_data(show_spinner=False)
def load_json(path: str, default):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default

def extract_text_from_pdf(file) -> str:
    try:
        reader = PdfReader(file)
        return "\n".join([(p.extract_text() or "") for p in reader.pages])
    except Exception:
        return ""

def relevant_snippets(text: str, keywords=None, window=1, max_chunks=6) -> str:
    """Light relevance filter for unstructured docs."""
    if not text: return ""
    if keywords is None:
        keywords = ["salary","income","epf","nps","gratuity","deduction","basic pay",
                    "hra","ctc","contribution","pension","bonus","allowance","tax"]
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    hits = []
    for i, ln in enumerate(lines):
        low = ln.lower()
        if any(k in low for k in keywords):
            start = max(0, i-window); end = min(len(lines), i+window+1)
            chunk = "\n".join(lines[start:end])
            if chunk not in hits:
                hits.append(chunk)
    return "\n---\n".join(hits[:max_chunks])

def parse_structured_df(df: pd.DataFrame) -> dict:
    """Infer income/expenses/savings from CSV headers."""
    cols = {c.lower().strip(): c for c in df.columns}
    out = {}
    for k in ["income","monthly income","salary","net salary","take home"]:
        if k in cols:
            try: out["income"] = float(df[cols[k]].iloc[0]); break
            except Exception: pass
    for k in ["expenses","monthly expenses","spend","outflow","total expenses"]:
        if k in cols:
            try: out["expenses"] = float(df[cols[k]].iloc[0]); break
            except Exception: pass
    for k in ["investments","current investments","savings","corpus"]:
        if k in cols:
            try: out["savings_now"] = float(df[cols[k]].iloc[0]); break
            except Exception: pass
    return out

def try_parse_json(s: str):
    """Parse JSON from LLM (tolerate fenced code blocks)."""
    if not s: return None
    try:
        return json.loads(s)
    except Exception:
        # strip code fences if present
        m = re.search(r"\{.*\}", s, flags=re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return None
        return None

# ------------ Session retention ------------
if "profile" not in st.session_state:
    st.session_state.profile = {}

# ------------ Sidebar Inputs ------------
with st.sidebar:
    st.markdown("### ⚙️ Inputs")
    # defaults may get overridden by uploaded CSV (below)
    age = st.number_input("Your Age", min_value=18, max_value=70, value=st.session_state.profile.get("age", 30))
    retire_age = st.number_input("Target Retirement Age", min_value=40, max_value=70, value=st.session_state.profile.get("retire_age", 60))
    income = st.number_input("Monthly Income (₹)", min_value=0, value=int(st.session_state.profile.get("income", 100000)), step=5000)
    expenses = st.number_input("Monthly Expenses (₹)", min_value=0, value=int(st.session_state.profile.get("expenses", 50000)), step=5000)
    savings_now = st.number_input("Current Investments (₹)", min_value=0, value=int(st.session_state.profile.get("savings_now", 0)), step=10000)
    risk = st.radio("Risk Profile", ["Low", "Medium", "High"], index={"Low":0,"Medium":1,"High":2}[st.session_state.profile.get("risk","Medium")], horizontal=True)

    if st.button("💾 Save inputs to session"):
        st.session_state.profile = {
            "age": age, "retire_age": retire_age, "income": income,
            "expenses": expenses, "savings_now": savings_now, "risk": risk
        }
        st.success("Saved for this session.")

    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
    with st.expander("Disclaimer"):
        st.caption("Educational tool, not investment advice. Consult a SEBI-registered Investment Adviser for paid guidance.")


def fmt_val(v, suffix=""):
    if v is None: return "Not available"
    return f"{v}{suffix}"

# ----------- Page & Theme------------
st.set_page_config(
    page_title="AI FI Planner 🇮🇳",
    page_icon="₹",
    layout="wide",
    menu_items={"About": "RAG-grounded retirement planner MVP for Indian professionals."}
)

# Global styles (lightweight, no extra deps)
st.markdown("""
<style>
:root {
  --bg: #0b1221;
  --card: rgba(255,255,255,0.08);
  --border: rgba(255,255,255,0.12);
  --text: #e7ecf3;
  --muted: #93a2b8;
  --accent: #6ee7b7;
  --accent2: #4ade80;
}

html, body, [data-testid="stAppViewContainer"] {
  background: radial-gradient(1200px 800px at 10% 0%, #0f1b36 0%, #0b1221 35%, #0a0f1c 100%);
  color: var(--text);
}

h1, h2, h3, h4, h5, h6, .stMarkdown p, .stMarkdown li, label, .stTextInput label, .stNumberInput label {
  color: var(--text) !important;
}

.hero {
  background: linear-gradient(90deg, rgba(110,231,183,0.18), rgba(59,130,246,0.18));
  border: 1px solid var(--border);
  border-radius: 18px;
  padding: 18px 22px;
  box-shadow: 0 8px 30px rgba(0,0,0,0.35);
}

.card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 16px 16px;
  box-shadow: 0 6px 20px rgba(0,0,0,0.25);
}

.small {
  color: var(--muted); font-size: 0.9rem;
}

.hr {
  height: 1px; background: var(--border); border: none; margin: 10px 0 2px 0;
}

.stButton > button {
  background: linear-gradient(135deg, var(--accent), var(--accent2));
  color: #0b1221;
  border: none;
  padding: 0.6rem 1rem;
  border-radius: 10px;
  font-weight: 700;
  box-shadow: 0 6px 20px rgba(78,222,128,0.25);
}
.stDownloadButton > button {
  border-radius: 10px;
  font-weight: 600;
}
.sidebar-title {
  font-weight: 700;
  font-size: 1.0rem;
  margin-top: 0.25rem;
}
</style>
""", unsafe_allow_html=True)

# Load environment variables
load_dotenv()
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("GROQ_API_KEY not found. Add it in Streamlit Cloud → Manage app → Settings → Secrets.")
    st.stop()

client = Groq(api_key=GROQ_API_KEY)

# ------------ Sidebar Inputs ------------
with st.sidebar:
    st.markdown("### ⚙️ Inputs")
    age = st.number_input("Your Age", min_value=18, max_value=70, value=30)
    retire_age = st.number_input("Target Retirement Age", min_value=40, max_value=70, value=60)
    income = st.number_input("Monthly Income (₹)", min_value=0, value=100000, step=5000)
    expenses = st.number_input("Monthly Expenses (₹)", min_value=0, value=50000, step=5000)
    risk = st.radio("Risk Profile", ["Low", "Medium", "High"], index=1, horizontal=True)
    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
    with st.expander("Disclaimer"):
        st.caption(
            "This is an educational tool and **not investment advice**. "
            "For personalized recommendations, consult a SEBI-registered Investment Adviser."
        )

# ------------ Hero ------------
st.markdown("""
<div class="hero">
  <h2>🇮🇳 AI Financial Independence Planner</h2>
  <p class="small">Grounded with live PPF/EPF facts and concise rules. Built with Groq (LLaMA 3) + Streamlit.</p>
</div>
""", unsafe_allow_html=True)
st.write("")

# ------------ Live Facts ------------
live = load_json("data/live.json", default={})
ppf_rate = live.get("small_savings", {}).get("ppf_rate")
ppf_as_of = live.get("small_savings", {}).get("as_of")
ppf_src   = live.get("small_savings", {}).get("source")

epf_rate = live.get("epf", {}).get("epf_rate")
epf_as_of = live.get("epf", {}).get("as_of")
epf_src   = live.get("epf", {}).get("source")

colA, colB, colC = st.columns([1.1,1.1,1])
with colA:
    st.markdown("<div class='card'>🧾 <b>PPF rate</b>", unsafe_allow_html=True)
    st.markdown(f"**{fmt_val(ppf_rate, '%')}**  \n<span class='small'>As of {fmt_val(ppf_as_of)} · Source: {fmt_val(ppf_src)}</span>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with colB:
    st.markdown("<div class='card'>🏦 <b>EPF rate</b>", unsafe_allow_html=True)
    st.markdown(f"**{fmt_val(epf_rate, '%')}**  \n<span class='small'>As of {fmt_val(epf_as_of)} · Source: {fmt_val(epf_src)}</span>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with colC:
    st.markdown("<div class='card'>👤 <b>Your profile</b>", unsafe_allow_html=True)
    st.markdown(f"**Age:** {age}  \n**Retire:** {retire_age}  \n**Risk:** {risk}", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

st.write("")

# ------------ Reference Rules (RAG snippets) ------------
snippets = load_json("kb/seed_snippets.json", default=[])
snippet_text = "\n".join(
    [f"- {s.get('text','').strip()} (Source: {s.get('source','Unknown')})" for s in snippets]
) or "- (No snippets found. Add kb/seed_snippets.json)"

with st.expander("📚 Reference Rules (snippets)"):
    st.markdown(snippet_text)

# ------------ Uploads: structured & unstructured ------------
st.markdown("### 📥 Optional: Upload CSV/PDF/TXT (budgets, payslips, benefits)")
upload = st.file_uploader("CSV → auto-fill numbers; PDF/TXT → extract relevant lines", type=["csv","pdf","txt"])

structured_overrides = {}
unstructured_snips = ""

if upload is not None:
    if upload.type == "text/csv":
        try:
            df = pd.read_csv(upload)
            structured_overrides = parse_structured_df(df)
            # Apply overrides to sidebar inputs
            if "income" in structured_overrides: income = int(structured_overrides["income"])
            if "expenses" in structured_overrides: expenses = int(structured_overrides["expenses"])
            if "savings_now" in structured_overrides: savings_now = int(structured_overrides["savings_now"])
            st.success(f"CSV loaded. Parsed: {', '.join(structured_overrides.keys()) or 'none'}")
        except Exception as e:
            st.warning(f"Could not read CSV: {e}")

    elif upload.type == "application/pdf":
        text = extract_text_from_pdf(upload)
        unstructured_snips = relevant_snippets(text)
        st.success("PDF loaded. Extracted relevant snippets below.")
        with st.expander("View extracted snippets"):
            st.code(unstructured_snips or "(no keyword matches)")

    else:  # txt
        try:
            text = upload.getvalue().decode("utf-8", errors="ignore")
            unstructured_snips = relevant_snippets(text)
            st.success("Text loaded. Extracted relevant snippets below.")
            with st.expander("View extracted snippets"):
                st.code(unstructured_snips or "(no keyword matches)")
        except Exception as e:
            st.warning(f"Could not read text: {e}")

# ------------ Generate Button Row ------------
left, right = st.columns([1,1])
with left:
    generate = st.button("🚀 Generate Plan", use_container_width=True)
with right:
    st.caption("Tip: tweak inputs from the sidebar and re-generate.")

# ------------ Generation ------------
if generate:
    with st.spinner("Thinking through your plan…"):
        # Build grounded prompt
        prompt = f"""
You are a financial planner for Indian salaried professionals.

Use the provided live facts as the **only** source of truth for PPF and EPF rates.
Do not invent or guess rates; always use them exactly as shown.

USER INPUTS:
- Age: {age}
- Monthly Income: ₹{income}
- Monthly Expenses: ₹{expenses}
- Target Retirement Age: {retire_age}
- Risk Profile: {risk}

LIVE FACTS:
- PPF rate: {ppf_rate}% (as of {ppf_as_of}, {ppf_src})
- EPF rate: {epf_rate}% (as of {epf_as_of}, {epf_src})

REFERENCE RULES (snippets):
{snippet_text}

TASKS:
1) Estimate the monthly savings needed to reach retirement at age {retire_age}.
2) Propose a simple asset allocation (Equity/Debt/Gold) suitable for these inputs and risk: {risk}.
3) Provide 3 starter action steps for the next 30 days.
4) State the assumptions you used (e.g., inflation, expected returns, equity/debt glide path).
5) Cite the live facts inline exactly as shown above.

Constraints:
- Keep currency in INR (₹).
- Be concise, practical, and avoid jargon.
- If any live fact is missing, clearly say what's missing instead of guessing.
"""

        system_msg = (
            "You are a cautious, India-focused retirement planning assistant. "
            "Use only the provided live facts for PPF/EPF. If data is missing, say so."
        )

        try:
            chat = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": prompt}
                ],
                model="llama3-8b-8192",
                temperature=0.2
            )
            plan_text = chat.choices[0].message.content

            st.success("Plan generated")
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("### 📈 Your Plan (Grounded)")
            st.write(plan_text)
            st.markdown("</div>", unsafe_allow_html=True)

            st.download_button(
                "⬇️ Download Plan (Markdown)",
                data=plan_text,
                file_name=f"fi-plan-{int(time.time())}.md",
                mime="text/markdown",
                use_container_width=True
            )

        except Exception as e:
            st.error(f"Model error: {e}")