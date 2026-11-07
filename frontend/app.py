import streamlit as st
from components.chat_interface import render_chat_interface

st.set_page_config(page_title="Internal Knowledge Assistant", page_icon="💬", layout="centered")

dark_mode = st.sidebar.toggle("Dark mode", key="dark_mode")
if dark_mode:
	page_bg, text, surface, border, muted = "#111713", "#edf2ed", "#1b241e", "#344139", "#a5b2a8"
else:
	page_bg, text, surface, border, muted = "#f5f7f3", "#1d2821", "#ffffff", "#dce4dc", "#65736a"

st.markdown(f"""
<style>
.stApp {{ background: {page_bg}; color: {text}; }}
[data-testid="stHeader"] {{ background: {page_bg}; }}
[data-testid="stSidebar"] {{ background: {surface}; border-right: 1px solid {border}; }}
.block-container {{ max-width: 850px; padding-top: 2rem; }}
[data-testid="stChatMessage"] {{ background: {surface}; border: 1px solid {border}; border-radius: 10px; }}
[data-testid="stChatInput"] textarea {{ background: {surface}; color: {text}; border-color: {border}; }}
[data-testid="stExpander"] {{ background: {surface}; border-color: {border}; }}
.subtitle {{ color: {muted}; text-align: center; margin-bottom: 2rem; }}
</style>
""", unsafe_allow_html=True)

st.title("Internal Knowledge Assistant")
st.markdown('<p class="subtitle">Ask questions about your internal documents.</p>', unsafe_allow_html=True)
render_chat_interface()
