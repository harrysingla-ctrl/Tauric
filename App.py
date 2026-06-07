"""
TradingAgents Streamlit UI
Run with: streamlit run app.py
Place this file inside the TradingAgents project root (same level as main.py).
"""

import streamlit as st
import threading
import queue
import sys
import io
import os
import datetime

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TradingAgents",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Dark terminal background */
.stApp {
    background: #0d0f14;
    color: #e2e8f0;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #111318;
    border-right: 1px solid #1e2330;
}
[data-testid="stSidebar"] * {
    color: #c9d1e0 !important;
}

/* Header */
.ta-header {
    font-family: 'Space Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    color: #00e5a0;
    letter-spacing: -1px;
    margin-bottom: 0;
    line-height: 1;
}
.ta-sub {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.85rem;
    color: #4a5568;
    margin-top: 4px;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}

/* Cards */
.card {
    background: #111318;
    border: 1px solid #1e2330;
    border-radius: 8px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1rem;
}

/* Agent log box */
.log-box {
    background: #090b0f;
    border: 1px solid #1e2330;
    border-radius: 6px;
    padding: 1rem;
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    color: #64ffb0;
    white-space: pre-wrap;
    max-height: 420px;
    overflow-y: auto;
    line-height: 1.6;
}

/* Decision output */
.decision-box {
    background: #0a1f14;
    border: 1px solid #00e5a0;
    border-radius: 8px;
    padding: 1.5rem;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.92rem;
    color: #d4f5e8;
    line-height: 1.8;
    white-space: pre-wrap;
}

/* Signal badge */
.badge-buy    { background:#003d20; color:#00e5a0; border:1px solid #00e5a0; padding:4px 14px; border-radius:20px; font-family:'Space Mono',monospace; font-size:0.8rem; font-weight:700; }
.badge-sell   { background:#3d0000; color:#ff6b6b; border:1px solid #ff6b6b; padding:4px 14px; border-radius:20px; font-family:'Space Mono',monospace; font-size:0.8rem; font-weight:700; }
.badge-hold   { background:#1a1a00; color:#f5c518; border:1px solid #f5c518; padding:4px 14px; border-radius:20px; font-family:'Space Mono',monospace; font-size:0.8rem; font-weight:700; }

/* Inputs */
.stTextInput input, .stDateInput input, .stSelectbox div[data-baseweb="select"] {
    background: #090b0f !important;
    border-color: #1e2330 !important;
    color: #e2e8f0 !important;
    font-family: 'Space Mono', monospace !important;
}

/* Primary button */
.stButton > button[kind="primary"] {
    background: #00e5a0;
    color: #060809;
    font-family: 'Space Mono', monospace;
    font-weight: 700;
    font-size: 0.82rem;
    border: none;
    border-radius: 6px;
    padding: 0.6rem 2rem;
    letter-spacing: 0.5px;
    transition: all 0.2s;
}
.stButton > button[kind="primary"]:hover {
    background: #00ffb3;
    transform: translateY(-1px);
}

/* Divider */
hr { border-color: #1e2330 !important; }

/* Labels */
.stSelectbox label, .stTextInput label, .stDateInput label,
.stSlider label, .stNumberInput label {
    color: #6b7a99 !important;
    font-size: 0.75rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.8px !important;
}

/* Metric */
[data-testid="metric-container"] {
    background: #111318;
    border: 1px solid #1e2330;
    border-radius: 8px;
    padding: 0.75rem 1rem;
}
[data-testid="stMetricValue"] { color: #00e5a0 !important; font-family: 'Space Mono', monospace !important; }
[data-testid="stMetricLabel"] { color: #4a5568 !important; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="ta-header">📈 TradingAgents</div>', unsafe_allow_html=True)
st.markdown('<div class="ta-sub">Multi-Agent LLM Financial Analysis Framework</div>', unsafe_allow_html=True)
st.markdown("---")

# ── Provider / model maps ─────────────────────────────────────────────────────
PROVIDER_MODELS = {
    "openai":     {"deep": ["gpt-5.5", "gpt-5.4", "gpt-4.1"],        "quick": ["gpt-5.4-mini", "gpt-5.4-nano", "gpt-4.1-mini"]},
    "anthropic":  {"deep": ["claude-opus-4-6", "claude-sonnet-4-6"],   "quick": ["claude-sonnet-4-6", "claude-haiku-4-5-20251001"]},
    "google":     {"deep": ["gemini-2.5-pro", "gemini-2.0-pro"],     "quick": ["gemini-3.0-flash", "gemini-3.0-flash-lite"]},
    "xai":        {"deep": ["grok-4", "grok-3"],                       "quick": ["grok-3-fast", "grok-3-mini-fast"]},
    "deepseek":   {"deep": ["deepseek-reasoner", "deepseek-chat"],     "quick": ["deepseek-chat"]},
    "ollama":     {"deep": ["llama3.3", "mistral", "qwen2.5"],         "quick": ["llama3.2", "phi4", "gemma3"]},
    "openrouter": {"deep": ["custom"],                                  "quick": ["custom"]},
}

PROVIDER_KEY_MAP = {
    "openai":     "OPENAI_API_KEY",
    "anthropic":  "ANTHROPIC_API_KEY",
    "google":     "GOOGLE_API_KEY",
    "xai":        "XAI_API_KEY",
    "deepseek":   "DEEPSEEK_API_KEY",
    "ollama":     None,
    "openrouter": "OPENROUTER_API_KEY",
}

# ── Sidebar: configuration ─────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuration")

    # Provider
    provider = st.selectbox(
        "LLM Provider",
        list(PROVIDER_MODELS.keys()),
        index=0,
    )

    models = PROVIDER_MODELS[provider]

    deep_model = st.selectbox("Deep-Think Model", models["deep"])
    quick_model = st.selectbox("Quick-Think Model", models["quick"])

    # API key
    key_env = PROVIDER_KEY_MAP[provider]
    if key_env:
        api_key = st.text_input(
            f"API Key ({key_env})",
            value=os.environ.get(key_env, ""),
            type="password",
            help=f"Will be set as {key_env}",
        )
        if api_key:
            os.environ[key_env] = api_key
    else:
        st.info("Ollama uses no API key. Ensure Ollama is running locally.")

    st.markdown("---")
    st.markdown("### 🔬 Research Depth")

    max_debate_rounds = st.slider("Bull/Bear Debate Rounds", 1, 5, 1)
    max_risk_rounds = st.slider("Risk Discussion Rounds", 1, 3, 1)
    temperature = st.slider(
        "Temperature (0 = deterministic)",
        0.0, 1.0, 0.3, step=0.05,
        help="Lower = more consistent outputs across runs",
    )

    st.markdown("---")
    st.markdown("### 💾 Options")
    checkpoint_enabled = st.toggle("Checkpoint Resume", value=False,
        help="Save state so interrupted runs can resume")
    debug_mode = st.toggle("Debug Output", value=False,
        help="Stream verbose agent logs to the log panel")

    st.markdown("---")
    st.markdown(
        '<span style="font-size:0.7rem;color:#2d3748;">TradingAgents v0.2.5 · For research only · Not financial advice</span>',
        unsafe_allow_html=True,
    )

# ── Main panel ────────────────────────────────────────────────────────────────
col_left, col_right = st.columns([1, 2], gap="large")

with col_left:
    st.markdown("### 🎯 Analysis Target")

    ticker = st.text_input(
        "Ticker Symbol",
        value="NVDA",
        placeholder="AAPL, BTC-USD, 0700.HK …",
        help="Any Yahoo Finance ticker. US: AAPL · HK: 0700.HK · Crypto: BTC-USD",
    ).strip().upper()

    analysis_date = st.date_input(
        "Analysis Date",
        value=datetime.date.today() - datetime.timedelta(days=7),
        max_value=datetime.date.today(),
        help="Historical date to run the analysis for",
    )

    st.markdown("#### 📊 Quick Examples")
    example_cols = st.columns(3)
    examples = [("AAPL", "🍎"), ("NVDA", "🤖"), ("BTC-USD", "₿")]
    for i, (ex_ticker, emoji) in enumerate(examples):
        with example_cols[i]:
            if st.button(f"{emoji} {ex_ticker}", use_container_width=True):
                ticker = ex_ticker
                st.rerun()

    st.markdown("---")

    # Validate before running
    can_run = bool(ticker)
    if key_env and not os.environ.get(key_env):
        can_run = False
        st.warning(f"⚠️ Set your {key_env} in the sidebar to run.")

    run_btn = st.button(
        "▶  Run Analysis",
        type="primary",
        disabled=not can_run,
        use_container_width=True,
    )

    if not can_run and ticker:
        st.caption("Add your API key in the sidebar to enable analysis.")

with col_right:
    st.markdown("### 📋 Agent Activity")

    log_placeholder = st.empty()
    log_placeholder.markdown('<div class="log-box">Waiting to start…</div>', unsafe_allow_html=True)

    st.markdown("### 🏁 Decision")
    decision_placeholder = st.empty()
    decision_placeholder.markdown('<div class="decision-box" style="color:#2d3748;">No analysis run yet.</div>', unsafe_allow_html=True)

# ── Run analysis ──────────────────────────────────────────────────────────────
if run_btn and ticker:
    # Import lazily so the app loads fast without the heavy deps
    try:
        from tradingagents.graph.trading_graph import TradingAgentsGraph
        from tradingagents.default_config import DEFAULT_CONFIG
    except ImportError as e:
        st.error(f"Could not import TradingAgents: {e}\n\nMake sure you are running this script from inside the TradingAgents project root and have run `pip install .`")
        st.stop()

    config = DEFAULT_CONFIG.copy()
    config["llm_provider"]           = provider
    config["deep_think_llm"]         = deep_model
    config["quick_think_llm"]        = quick_model
    config["max_debate_rounds"]      = max_debate_rounds
    config["max_risk_discuss_rounds"]= max_risk_rounds
    config["temperature"]            = temperature if temperature > 0 else None
    config["checkpoint_enabled"]     = checkpoint_enabled

    date_str = analysis_date.strftime("%Y-%m-%d")

    # ── Stream stdout/stderr into the log box ──────────────────────────────
    log_lines: list[str] = []
    result_queue: queue.Queue = queue.Queue()

    class StreamCapture(io.TextIOBase):
        def write(self, text):
            if text.strip():
                log_lines.append(text.rstrip())
            return len(text)
        def flush(self): pass

    def run_agents():
        old_stdout, old_stderr = sys.stdout, sys.stderr
        if debug_mode:
            cap = StreamCapture()
            sys.stdout = cap
            sys.stderr = cap
        try:
            ta = TradingAgentsGraph(debug=debug_mode, config=config)
            _, decision = ta.propagate(ticker, date_str)
            result_queue.put(("ok", decision))
        except Exception as exc:
            result_queue.put(("err", str(exc)))
        finally:
            if debug_mode:
                sys.stdout = old_stdout
                sys.stderr = old_stderr

    thread = threading.Thread(target=run_agents, daemon=True)
    thread.start()

    # ── Poll and update UI while thread runs ───────────────────────────────
    log_placeholder.markdown(
        f'<div class="log-box">🔄 Starting analysis for <b>{ticker}</b> on {date_str}…\n'
        f'Provider: {provider} | Deep: {deep_model} | Quick: {quick_model}\n'
        f'Debate rounds: {max_debate_rounds} | Risk rounds: {max_risk_rounds}\n'
        + ("─" * 60) + "</div>",
        unsafe_allow_html=True,
    )
    decision_placeholder.markdown(
        '<div class="decision-box" style="color:#2d3748;">⏳ Agents are deliberating…</div>',
        unsafe_allow_html=True,
    )

    import time
    spinner_frames = ["⠋","⠙","⠹","⠸","⠼","⠴","⠦","⠧","⠇","⠏"]
    frame_i = 0

    while thread.is_alive():
        frame = spinner_frames[frame_i % len(spinner_frames)]
        frame_i += 1

        log_content = (
            f"🔄 Starting analysis for {ticker} on {date_str}…\n"
            f"Provider: {provider} | Deep: {deep_model} | Quick: {quick_model}\n"
            f"Debate rounds: {max_debate_rounds} | Risk rounds: {max_risk_rounds}\n"
            + "─" * 60 + "\n"
        )
        if debug_mode and log_lines:
            log_content += "\n".join(log_lines[-80:])  # keep last 80 lines
        else:
            log_content += f"\n{frame} Running agents, please wait…\n\nThis typically takes 1–5 minutes depending on the model and depth."

        log_placeholder.markdown(
            f'<div class="log-box">{log_content}</div>',
            unsafe_allow_html=True,
        )
        time.sleep(0.3)

    # ── Collect result ─────────────────────────────────────────────────────
    status, payload = result_queue.get()

    if status == "err":
        log_placeholder.markdown(
            f'<div class="log-box" style="color:#ff6b6b;">❌ Error:\n{payload}</div>',
            unsafe_allow_html=True,
        )
        decision_placeholder.markdown(
            f'<div class="decision-box" style="border-color:#ff6b6b;color:#ff6b6b;">Analysis failed.\n\n{payload}</div>',
            unsafe_allow_html=True,
        )
    else:
        decision_text = str(payload)

        # Detect signal keyword for badge
        upper = decision_text.upper()
        if "BUY" in upper:
            badge = '<span class="badge-buy">▲ BUY</span>'
        elif "SELL" in upper:
            badge = '<span class="badge-sell">▼ SELL</span>'
        else:
            badge = '<span class="badge-hold">◆ HOLD</span>'

        final_log = (
            f"✅ Analysis complete for {ticker} on {date_str}\n"
            + "─" * 60 + "\n"
        )
        if debug_mode and log_lines:
            final_log += "\n".join(log_lines[-80:])
        else:
            final_log += "Agents completed successfully."

        log_placeholder.markdown(
            f'<div class="log-box">{final_log}</div>',
            unsafe_allow_html=True,
        )
        decision_placeholder.markdown(
            f'{badge}<br><br><div class="decision-box">{decision_text}</div>',
            unsafe_allow_html=True,
        )

        # Metrics row
        st.markdown("---")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Ticker", ticker)
        m2.metric("Date", date_str)
        m3.metric("Debate Rounds", max_debate_rounds)
        m4.metric("Provider", provider.capitalize())

