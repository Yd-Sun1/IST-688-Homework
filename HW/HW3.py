import streamlit as st
from openai import OpenAI
from google import genai
from google.genai import types
import requests
from bs4 import BeautifulSoup

st.title("HW 3 - URL Chatbot")
st.write(
    "This chatbot answers questions using content from up to two URLs. "
    "You can choose between OpenAI GPT-5.6 Sol and Google Gemini 3.1 Pro. "
    "The chatbot keeps the last 6 conversation messages as short-term memory."
)

openai_client = OpenAI(api_key=st.secrets["openai_api_key"])
gemini_client = genai.Client(api_key=st.secrets["gemini_api_key"])

def read_url_content(url):
    try:
        headers = {
            "User-Agent": "Mozilla/5.0"
        }

        response = requests.get(
            url,
            headers=headers,
            timeout=10
        )

        response.raise_for_status()

        soup = BeautifulSoup(response.content, "html.parser")
        text = soup.get_text(" ", strip=True)

        # Check if the website returned a verification page
        blocked_words = [
            "one moment",
            "verification",
            "just a moment",
            "enable javascript"
        ]

        if any(word in text.lower() for word in blocked_words):
            reader_url = "https://r.jina.ai/" + url

            reader_response = requests.get(
                reader_url,
                timeout=20
            )

            reader_response.raise_for_status()
            text = reader_response.text

        return text

    except requests.RequestException as e:
        print(f"Error reading URL: {e}")
        return None
    st.sidebar.header("Website Options")

url1 = st.sidebar.text_input("URL 1")
url2 = st.sidebar.text_input("URL 2")

model_choice = st.sidebar.selectbox(
    "Choose a model",
    ["OpenAI GPT-5.6 Sol", "Google Gemini 3.1 Pro"]
)


context_parts = []

if url1:
    content1 = read_url_content(url1)
    if content1:
        context_parts.append(f"Content from URL 1:\n{content1}")

if url2:
    content2 = read_url_content(url2)
    if content2:
        context_parts.append(f"Content from URL 2:\n{content2}")

url_context = "\n\n".join(context_parts)

def get_buffer(messages):
    system_messages = [
        message for message in messages
        if message["role"] == "system"
    ]

    conversation_messages = [
        message for message in messages
        if message["role"] != "system"
    ]

    recent_messages = conversation_messages[-6:]

    return system_messages + recent_messages

system_prompt = f"""
You are a helpful chatbot that answers questions about the provided website content.

Use the website content below as context when answering the user's questions.

Website content:
{url_context}
"""

system_prompt = f"""
You are a helpful chatbot that answers questions about the provided website content.

Use the website content below as context when answering the user's questions.

Website content:
{url_context}
"""

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "system",
            "content": system_prompt
        }
    ]
else:
    st.session_state.messages[0] = {
        "role": "system",
        "content": system_prompt
    }

for message in st.session_state.messages:
    if message["role"] != "system":
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
if prompt := st.chat_input("Ask me a question"):

    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )

    with st.chat_message("user"):
        st.markdown(prompt)

        buffer_messages = get_buffer(st.session_state.messages)

    with st.chat_message("assistant"):

        if model_choice == "OpenAI GPT-5.6 Sol":
            stream = openai_client.chat.completions.create(
                model="gpt-5.6-sol",
                messages=buffer_messages,
                stream=True
            )

            response = st.write_stream(stream)

        else:
            conversation_text = "\n".join(
                f"{message['role']}: {message['content']}"
                for message in buffer_messages
                if message["role"] != "system"
            )

            gemini_stream = gemini_client.models.generate_content_stream(
                model="gemini-3.1-pro-preview",
                contents=conversation_text,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt
                )
            )

            def stream_gemini():
                for chunk in gemini_stream:
                    if chunk.text:
                        yield chunk.text

            response = st.write_stream(stream_gemini())

    st.session_state.messages.append(
        {"role": "assistant", "content": response}
    )

    st.session_state.messages.append(
        {"role": "assistant", "content": response}
    )