import streamlit as st
from dotenv import load_dotenv
import os
from groq import Groq
import json

# Load environment variables
load_dotenv()
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("GROQ_API_KEY not found. Add it in Streamlit Cloud → Manage app → Settings → Secrets.")
    st.stop()

client = Groq(api_key=GROQ_API_KEY)

# Streamlit app test
st.title("AI Financial Independence Planner 🇮🇳")
st.write("Enter your details to get a basic retirement savings plan.")

# User inputs
age = st.number_input("Your Age", min_value=18, max_value=70, value=30)
income = st.number_input("Monthly Income (₹)", min_value=0, value=100000, step=5000)
expenses = st.number_input("Monthly Expenses (₹)", min_value=0, value=50000, step=5000)
retire_age = st.number_input("Target Retirement Age", min_value=40, max_value=70, value=60)


live = {}
try:
    with open("data/live.json", "r", encoding="utf-8") as f:
        live = json.load(f)
except Exception:
    live = {}

ppf_rate = live.get("small_savings", {}).get("ppf_rate")
ppf_as_of = live.get("small_savings", {}).get("as_of")
ppf_src   = live.get("small_savings", {}).get("source")

epf_rate = live.get("epf", {}).get("epf_rate")
epf_as_of = live.get("epf", {}).get("as_of")
epf_src   = live.get("epf", {}).get("source")

st.markdown("### Live Facts (India)")
colA, colB = st.columns(2)
with colA:
    st.write(f"**PPF rate:** {ppf_rate}%  \n_As of:_ {ppf_as_of}  \nSource: {ppf_src}")
with colB:
    st.write(f"**EPF rate:** {epf_rate}%  \n_As of:_ {epf_as_of}  \nSource: {epf_src}")

snippets = []
try:
    with open("kb/seed_snippets.json", "r", encoding="utf-8") as f:
        snippets = json.load(f)
except Exception:
    snippets = []

snippet_text = "\n".join([f"- {s['text']} (Source: {s['source']})" for s in snippets])


if st.button("Generate Plan"):
    prompt = f"""
You are a financial planner for Indian salaried professionals.
Use the provided live facts as the single source of truth for PPF/EPF rates.

USER INPUTS:
- Age: {age}
- Monthly Income: ₹{income}
- Monthly Expenses: ₹{expenses}
- Target Retirement Age: {retire_age}

LIVE FACTS:
- PPF rate: {ppf_rate}% (as of {ppf_as_of}, {ppf_src})
- EPF rate: {epf_rate}% (as of {epf_as_of}, {epf_src})

REFERENCE RULES (snippets):
{snippet_text}

TASKS:
1) Estimate monthly savings needed for retirement at {retire_age}.
2) Propose a simple asset allocation (Equity/Debt/Gold) appropriate for the inputs.
3) Provide 3 starter actions for the next 30 days.
4) State the assumptions you used (inflation, expected returns).
5) Cite the live facts inline exactly as shown above.

Keep currency in INR (₹). Be concise and practical.
"""

    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system",
             "content": "You are a cautious, India-focused retirement planning assistant. Use only provided live facts for PPF/EPF."},
            {"role": "user", "content": prompt}
        ],
        model="llama3-8b-8192",
        temperature=0.2
    )

    st.subheader("Your Plan (Grounded)")
    st.write(chat_completion.choices[0].message.content)