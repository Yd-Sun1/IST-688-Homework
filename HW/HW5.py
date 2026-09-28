import streamlit as st
from openai import OpenAI
import sys
import json
from pathlib import Path
from bs4 import BeautifulSoup

# Fix SQLite for ChromaDB on Streamlit Cloud
__import__("pysqlite3")
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

import chromadb


# Page title
st.title("HW 5 -- Intelligent Organization Chatbot")


# OpenAI client
client = OpenAI(
    api_key=st.secrets["openai_api_key"]
)


# Create ChromaDB client
chroma_client = chromadb.PersistentClient(
    path="./ChromaDB_for_HW4"
)


# Read one HTML file and convert it to text
def extract_text_from_html(html_path):

    with open(html_path, "r", encoding="utf-8", errors="ignore") as file:

        soup = BeautifulSoup(file, "html.parser")

        text = soup.get_text(
            separator=" ",
            strip=True
        )

    return text


# Split each organization page into two chunks
def chunk_document(text):

    words = text.split()

    middle = len(words) // 2

    chunk1 = " ".join(words[:middle])

    chunk2 = " ".join(words[middle:])

    return [chunk1, chunk2]


# Create an embedding and add the document to ChromaDB
def add_to_collection(collection, text, file_name):

    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )

    embedding = response.data[0].embedding

    collection.add(
        documents=[text],
        ids=[file_name],
        embeddings=[embedding]
    )


# Read HTML files and add them to ChromaDB
def load_html_to_collection(collection, folder_path):

    folder = Path(folder_path)

    html_files = folder.glob("*.html")

    for html_file in html_files:

        text = extract_text_from_html(html_file)

        chunks = chunk_document(text)

        file_name = html_file.name

        for i, chunk in enumerate(chunks):

            chunk_id = f"{file_name}_chunk_{i + 1}"

            add_to_collection(
                collection,
                chunk,
                chunk_id
            )


# Create the vector database
def create_vector_db():

    collection = chroma_client.get_or_create_collection(
        name="HW4Collection"
    )

    # Only create the vector database if it is empty
    if collection.count() == 0:

        load_html_to_collection(
            collection,
            "su_orgs"
        )

    return collection


# Load the vector database
if "HW4_VectorDB" not in st.session_state:

    st.session_state.HW4_VectorDB = create_vector_db()


collection = st.session_state.HW4_VectorDB


# Tool function for finding relevant organization information
def relevant_club_info(query):

    response = client.embeddings.create(
        input=query,
        model="text-embedding-3-small"
    )

    query_embedding = response.data[0].embedding

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3
    )

    context = ""

    for i in range(len(results["documents"][0])):

        file_name = results["ids"][0][i]
        document_text = results["documents"][0][i]

        context += f"\nSource: {file_name}\n"
        context += document_text
        context += "\n"

    return context


# Define the tool for the LLM
club_tool = {
    "type": "function",
    "function": {
        "name": "relevant_club_info",
        "description": "Get relevant information about Syracuse University student organizations.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "A search query for student organization information"
                }
            },
            "required": ["query"]
        }
    }
}


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []


# Display previous chat messages
for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.write(message["content"])


# Get user input
prompt = st.chat_input(
    "Ask a question about Syracuse University student organizations"
)


if prompt:

    # Display user message
    with st.chat_message("user"):
        st.write(prompt)

    # Add user message to memory
    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )

    # Use recent conversation history as short-term memory
    recent_messages = st.session_state.messages[-11:]

    # First LLM call with the tool
    response = client.chat.completions.create(
        model="gpt-5-mini",
        messages=recent_messages,
        tools=[club_tool],
        tool_choice="auto"
    )

    message = response.choices[0].message

    # If the LLM calls the tool
    if message.tool_calls:

        tool_call = message.tool_calls[0]

        arguments = json.loads(
            tool_call.function.arguments
        )

        query = arguments["query"]

        # Search ChromaDB
        context = relevant_club_info(query)

        system_message = f"""
You are a helpful assistant for Syracuse University student organizations.

Use the retrieved organization information below to answer the user's question.

Answer using the retrieved information when possible.

If the retrieved information does not contain enough information,
say that the information was not found in the provided documents.

Do not offer to search the web or look up outside information.
Only answer using the retrieved documents.

Retrieved organization information:

{context}
"""

        # Second LLM call without tools
        final_response = client.chat.completions.create(
            model="gpt-5-mini",
            messages=[
                {
                    "role": "system",
                    "content": system_message
                }
            ] + recent_messages
        )

        answer = final_response.choices[0].message.content

    else:

        answer = message.content


    # Display assistant response
    with st.chat_message("assistant"):
        st.write(answer)


    # Add assistant response to memory
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )


    # Keep only the last 5 interactions
    st.session_state.messages = st.session_state.messages[-10:]