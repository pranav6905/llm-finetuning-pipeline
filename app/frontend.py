import streamlit as st
import requests

API_URL = "http://localhost:8000"

st.set_page_config(
    page_title = "LLM Finetuning Pipeline",
    page_icon  = "🦙",
    layout     = "centered"
)

st.title("🦙 LLaMA 3.2-1B — SFT + DPO")
st.caption("QLoRA fine-tuned · Aligned with DPO · Deployed with FastAPI + Docker")

# ── Results table ─────────────────────────────────────────
st.markdown("### Evaluation Results")

col0, col1, col2, col3 = st.columns([1.2, 1, 1, 1])
col0.markdown("**Benchmark**")
col1.markdown("**Base Model**")
col2.markdown("**After SFT**")
col3.markdown("**After DPO**")

col0, col1, col2, col3 = st.columns([1.2, 1, 1, 1])
col0.markdown("ARC-Easy")
col1.metric("", "59.81%")
col2.metric("", "63.26%", "+3.45%")
col3.metric("", "62.88%", "-0.38%")

col0, col1, col2, col3 = st.columns([1.2, 1, 1, 1])
col0.markdown("TruthfulQA")
col1.metric("", "25.21%")
col2.metric("", "26.32%", "+1.11%")
col3.metric("", "26.93%", "+0.61%")

st.divider()

# ── API health check ──────────────────────────────────────
try:
    health = requests.get(f"{API_URL}/health", timeout=5)
    if health.status_code == 200:
        st.success("Model API is running")
    else:
        st.error("API not responding")
except:
    st.warning("API is still starting up — please refresh in 30 seconds")
    st.stop()

# ── Session state init ────────────────────────────────────
# This is the KEY fix — text_area reads directly from this key
if "prompt_text" not in st.session_state:
    st.session_state["prompt_text"] = ""

# ── Example buttons (BEFORE text_area) ───────────────────
st.markdown("### Try the Model")
st.markdown("**Quick examples — click to load:**")

ex1, ex2, ex3 = st.columns(3)

if ex1.button("ML vs DL"):
    st.session_state["prompt_text"] = "Explain the difference between machine learning and deep learning."

if ex2.button("Communication"):
    st.session_state["prompt_text"] = "What are the key principles of effective communication?"

if ex3.button("Logic"):
    st.session_state["prompt_text"] = "If all roses are flowers and some flowers fade quickly, do all roses fade quickly?"

# ── Text area reads from session state ───────────────────
# IMPORTANT: key="prompt_text" links it directly to session state
instruction = st.text_area(
    "Enter your instruction",
    placeholder = "Type here or click an example above...",
    height      = 120,
    key         = "prompt_text",   # ← this is what makes buttons work
)

with st.expander("Generation Settings"):
    max_tokens  = st.slider("Max new tokens", 50, 400, 200)
    temperature = st.slider("Temperature", 0.1, 1.0, 0.7)

# ── Generate ──────────────────────────────────────────────
if st.button("Generate Response", type="primary", use_container_width=True):
    if not instruction.strip():
        st.warning("Please enter an instruction first.")
    else:
        with st.spinner("Generating... (30-60s on CPU)"):
            try:
                res = requests.post(
                    f"{API_URL}/generate",
                    json = {
                        "instruction"   : instruction,
                        "max_new_tokens": max_tokens,
                        "temperature"   : temperature,
                    },
                    timeout = 120,
                )
                if res.status_code == 200:
                    data = res.json()
                    st.markdown("### Response")
                    st.write(data["response"])
                    st.caption(
                        f"⏱️ {data['latency_seconds']}s · "
                        f"Model: `{data['model_id']}`"
                    )
                else:
                    st.error(f"API error {res.status_code}: {res.text}")
            except requests.exceptions.Timeout:
                st.error("Timed out. Try reducing max tokens.")
            except Exception as e:
                st.error(f"Error: {str(e)}")

# ── Footer ────────────────────────────────────────────────
st.divider()
st.markdown(
    "🔗 [GitHub](https://github.com/pranav6905/llm-finetuning-pipeline) · "
    "[HuggingFace Model](https://huggingface.co/pranav6905/llama-1b-sft-dpo-final)"
)