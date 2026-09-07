import streamlit as st
from openai import OpenAI
import requests
from bs4 import BeautifulSoup

def read_url_content(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")
        return soup.get_text()

    except requests.RequestException as e:
        print(f"Error reading {url}: {e}")
        return None

openai_client = OpenAI(
    api_key=st.secrets["openai_api_key"]
)

gemini_client = OpenAI(
    api_key=st.secrets["gemini_api_key"],
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

language = st.sidebar.selectbox(
    "Select summary language",
    ["English", "Chinese", "Spanish", "French"]
)

summary_type = st.sidebar.selectbox(
    "Select summary type",
    [
        "100 words",
        "2 connecting paragraphs",
        "5 bullet points"
    ]
)

llm_provider = st.sidebar.selectbox(
    "Select LLM",
    ["OpenAI", "Gemini"]
)

use_advanced_model = st.sidebar.checkbox("Use advanced model")

if llm_provider == "OpenAI":
    if use_advanced_model:
        model_to_use = "gpt-5-mini"
    else:
        model_to_use = "gpt-5-nano"

else:
    if use_advanced_model:
        model_to_use = "gemini-3.5-flash"
    else:
        model_to_use = "gemini-3.5-flash-lite"

if summary_type == "100 words":
    summary_instruction = "Summarize the document in approximately 100 words."

elif summary_type == "2 connecting paragraphs":
    summary_instruction = "Summarize the document in 2 connecting paragraphs."

else:
    summary_instruction = "Summarize the document in 5 bullet points."

# Show title and description.
st.title("HW 2 - URL Summarizer")
st.write(
    "Enter a web page URL below to generate a summary."
)

# Ask user for their OpenAI API key via `st.text_input`.
# Alternatively, you can store the API key in `./.streamlit/secrets.toml` and access it
# via `st.secrets`, see https://docs.streamlit.io/develop/concepts/connections/secrets-management

# Let the user upload a file via `st.file_uploader`.
url = st.text_input("Enter a web page URL")



if url:
    document = read_url_content(url)

    if document:
        messages = [
            {
                "role": "user",
                "content": f"""
Please summarize the following content in {language}.

{summary_instruction}

Document:
{document}
"""
            }
        ]

        if llm_provider == "OpenAI":
            stream = openai_client.chat.completions.create(
                model=model_to_use,
                messages=messages,
                stream=True,
            )

        else:
            stream = gemini_client.chat.completions.create(
                model=model_to_use,
                messages=messages,
                stream=True,
            )

        st.write_stream(stream)