import streamlit as st
from openai import OpenAI
import sys
from pathlib import Path
from bs4 import BeautifulSoup

# Fix SQLite for ChromaDB on Streamlit Cloud
__import__("pysqlite3")
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

import chromadb


# Page title
st.title("HW 4 -- iSchool Organization Chatbot")


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

def chunk_document(text):

    words = text.split()

    middle = len(words) // 2

    # I use a simple half-document chunking method.
    # Each organization page is divided into two similar-sized chunks.
    # this keeps both chunks similar in size.

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


if "HW4_VectorDB" not in st.session_state:

    st.session_state.HW4_VectorDB = create_vector_db()


collection = st.session_state.HW4_VectorDB



def get_relevant_context(question):

    # Create an embedding for the user's question
    response = client.embeddings.create(
        input=question,
        model="text-embedding-3-small"
    )

    query_embedding = response.data[0].embedding

    # Search ChromaDB for the 3 most relevant documents
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3
    )

    # Combine the returned document text
    context = ""

    for i in range(len(results["documents"][0])):

        file_name = results["ids"][0][i]
        document_text = results["documents"][0][i]

        context += f"\nSource: {file_name}\n"
        context += document_text
        context += "\n"

    return context

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

    # Retrieve relevant information from ChromaDB
    context = get_relevant_context(prompt)

    # Instructions for the LLM
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

    # Use previous conversation history as memory
    recent_messages = st.session_state.messages[-11:]

    messages_for_llm = [
        {
            "role": "system",
            "content": system_message
        }
    ] + recent_messages

    # Send conversation to the LLM
    response = client.chat.completions.create(
        model="gpt-5-mini",
        messages=messages_for_llm
    )

    answer = response.choices[0].message.content

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
    # One interaction = one user message + one assistant response
    st.session_state.messages = st.session_state.messages[-10:]