# noiPA_chatbot.py  ── run with:  streamlit run noiPA_chatbot.py
import streamlit as st
import asyncio, json, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import Any
from Agents_Reply import triage_with_memory, triage_agent


# ─────────────────────────── 0) HELPERS ───────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_dataset(name: str) -> pd.DataFrame:
    path_map = {
        "Accesso":      "Accesso_English.csv",
        "Stipendi":     "Stipendi_English.csv",
        "Reddito":      "Reddito_English.csv",
        "Pendolarismo": "Pendolarismo_English.csv",
    }
    return pd.read_csv(path_map[name])


def reset_conversation() -> None:        # NEW
    """Clear everything user-chat related and restart the app."""
    st.session_state.memory_buffer = []
    st.session_state.chat_history = []
    st.session_state.ds_choice = "None"   # reset dataset selector
    plt.close("all")                      # close any lingering figures
    st.rerun()                            # non-experimental rerun

# ----------  HEADER  ----------
c1, c2, c3 = st.columns([1, 4, 1])   # centre the image
with c2:
    st.image("Reply Noipa Logo.png",use_container_width=True) 

st.markdown("<h1 style='text-align:center; margin-top:-0.4rem;'>NoiPA Agent</h1>",
            unsafe_allow_html=True)
st.markdown("<p style='text-align:center; margin-top:-0.5rem;'>"
            "I’m here to help. Ask me anything!</p>",
            unsafe_allow_html=True)

# ───────────────────────── 1) SESSION STATE ───────────────────────────────────
st.session_state.setdefault("memory_buffer", [])
st.session_state.setdefault("chat_history", [])

# ────────────────────────── 2) SIDEBAR UI ─────────────────────────────────────
st.sidebar.title("⚙️ Settings")

# 🔄 RESET (uses on_click callback)
st.sidebar.button("🔄 Reset Conversation", on_click=reset_conversation)

# 📊 DATA-VIEWER
with st.sidebar.expander("📊 Explore data", expanded=False):
    ds_choice = st.selectbox(
        "Select a dataset",
        ["None", "Accesso", "Stipendi", "Reddito", "Pendolarismo"],
        key="ds_choice",
    )
    if ds_choice != "None":
        st.dataframe(load_dataset(ds_choice), use_container_width=True)

# 📥 DOWNLOAD CHAT HISTORY
if st.session_state.chat_history:
    safe_history = json.dumps(
        st.session_state.chat_history,
        indent=2,
        default=lambda o: "<Figure>" if isinstance(o, Figure) else str(o),
    )
    st.sidebar.download_button(
        "📥 Download Chat as JSON",
        data=safe_history,
        file_name="chat_history.json",
        mime="application/json",
    )

# ───────────────────────── 3) DYNAMIC THEMING ─────────────────────────────────
bg_color, fg_color, input_bg = "#87CEFA", "#333333", "white"

st.markdown(
    f"""
    <style>
      .stApp {{ background-color: {bg_color}; color: {fg_color}; }}s
      .stTextInput input {{ background-color: {input_bg}; color: {fg_color}; }}
      .stChatMessage, .stButton {{ color: {fg_color} !important; }}
      .stMarkdown p,
      .stMarkdown ul li,
      .stChatMessage .stMarkdown p,
      .stChatMessage .stMarkdown ul li {{ color: {fg_color} !important; }}
    </style>
    """,
    unsafe_allow_html=True,
)

plt.rcParams.update(
    {
        "figure.facecolor": bg_color,
        "axes.facecolor": bg_color,
        "axes.edgecolor": fg_color,
        "axes.labelcolor": fg_color,
        "xtick.color": fg_color,
        "ytick.color": fg_color,
        "text.color": fg_color,
        "legend.facecolor": bg_color,
        "legend.edgecolor": fg_color,
    }
)

# ───────────────────────── 4) CORE LOGIC ──────────────────────────────────────
max_memory_length = 6


def run_sync(query: str) -> Any:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    async def _internal():
        return await triage_with_memory(
            query, triage_agent, st.session_state.memory_buffer, max_memory_length
        )

    result = loop.run_until_complete(_internal())

    if (
        not (
            isinstance(result, dict)
            and any(isinstance(v, Figure) for v in result.values())
        )
        and plt.get_fignums()
    ):
        fig = plt.gcf()
        result = {"fig": fig, **(result if isinstance(result, dict) else {"output": result})}

    st.session_state.chat_history.append({"user": query, "assistant": result})
    return result


# ───────────────────────── 6) RENDER ONE MSG ──────────────────────────────────
def render_assistant(chat, idx):
    resp = chat["assistant"]
    if isinstance(resp, dict):
        for k, v in resp.items():
            if k == "fig" and isinstance(v, Figure):
                st.pyplot(v, clear_figure=False)
            elif k != "memory_buffer":
                st.markdown(f"*{k}:*")
                st.write(v)
    else:
        st.write(resp)

    col1, col2, _ = st.columns([1, 1, 6])
    if col1.button("↻ Regenerate", key=f"regen_{idx}"):
        run_sync(chat["user"])
        st.rerun()
    if col2.button("✏️ Summarize", key=f"summ_{idx}"):
        last = resp.get("output", "") if isinstance(resp, dict) else str(resp)
        run_sync(f"Please summarize the following:\n\n{last}")
        st.rerun()


# ───────────────────────── 7) PREVIOUS CHAT ───────────────────────────────────
for i, msg in enumerate(st.session_state.chat_history):
    with st.chat_message("user", avatar="🧑"):
        st.write(msg["user"])
    with st.chat_message("assistant", avatar="🐉"):
        render_assistant(msg, i)

# ───────────────────────── 8) NEW INPUT LOOP ──────────────────────────────────
query = st.chat_input("Insert your query here")
if query:
    with st.chat_message("user", avatar="🧑"):
        st.write(query)

    with st.spinner("🐉 Thinking…"):
        result = run_sync(query)

    with st.chat_message("assistant", avatar="🐉"):
        if isinstance(result, dict):
            if "fig" in result and isinstance(result["fig"], Figure):
                st.pyplot(result["fig"], clear_figure=False)
            output = result.get("Output") or result.get("output")
            if output:
                st.markdown("*Output:*")
                st.write(output)
        else:
            st.write(result)