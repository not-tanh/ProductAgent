import os
import requests
import streamlit as st

st.set_page_config(page_title="Product Agent Chat", layout="wide")

DEFAULT_API_BASE = os.getenv("CHAT_API_BASE_URL", "http://localhost:8000")
CHAT_ENDPOINT = "/chat"

if "api_base" not in st.session_state:
    st.session_state.api_base = DEFAULT_API_BASE
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []  # [{"role": "user"/"assistant", "content": "..."}]


def reset_chat():
    st.session_state.session_id = None
    st.session_state.messages = []


def call_chat_api(message: str, api_base: str, session_id: str | None, timeout: int = 60):
    url = api_base.rstrip("/") + CHAT_ENDPOINT
    headers = {}
    if session_id:
        headers["X-Session-Id"] = session_id

    return requests.post(
        url,
        json={"message": message},
        headers=headers,
        timeout=timeout,
    )


with st.sidebar:
    st.header("Settings")
    st.session_state.api_base = st.text_input("CHAT_API_BASE_URL", value=st.session_state.api_base)
    timeout_s = st.number_input("Timeout (seconds)", min_value=5, max_value=300, value=120, step=5)
    show_debug = st.checkbox("Show debug response", value=False)

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        if st.button("New chat", use_container_width=True):
            reset_chat()
    with col2:
        if st.button("Clear UI only", use_container_width=True):
            st.session_state.messages = []

    st.divider()
    st.caption("Session")
    st.code(st.session_state.session_id or "(none)", language="text")

st.title("Chat UI (Streamlit → FastAPI /chat)")

# Render history
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

prompt = st.chat_input("Enter your message...")

if prompt:
    # Show user message immediately
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                resp = call_chat_api(
                    message=prompt,
                    api_base=st.session_state.api_base,
                    session_id=st.session_state.session_id,
                    timeout=int(timeout_s),
                )

                if resp.status_code == 409:
                    st.warning("Something is wrong. Please try again later.")
                    if show_debug:
                        st.code(resp.text, language="text")
                elif resp.status_code >= 400:
                    st.error(f"API error: {resp.status_code}")
                    if show_debug:
                        st.code(resp.text, language="text")
                else:
                    data = resp.json()
                    # Persist session_id from server
                    st.session_state.session_id = data.get("session_id") or st.session_state.session_id

                    reply = data.get("reply", "")
                    st.markdown(reply if reply else "(empty reply)")

                    st.session_state.messages.append({"role": "assistant", "content": reply if reply else ""})

                    if show_debug:
                        st.json(data)

            except requests.exceptions.RequestException as e:
                st.error(f"Request failed: {e}")
