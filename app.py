import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI

# Setup
load_dotenv()

st.set_page_config(page_title="PDF Chat", layout="wide")

st.title("PDF Chat (RAG)")
st.header("Upload a PDF and chat with it", divider="green")

# Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

if "vectorstore_ready" not in st.session_state:
    st.session_state.vectorstore_ready = False


# Helpers
def format_chat_history(messages):
    history = ""
    for msg in messages:
        history += f"{msg['role'].capitalize()}: {msg['content']}\n"
    return history


# Models
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    add_start_index=True
)

embedding_model = HuggingFaceEmbeddings(
    model_name="intfloat/e5-small-v2"
)

vectorstore = Chroma(
    collection_name="my_docs",
    embedding_function=embedding_model,
    persist_directory="./chroma/db"
)

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0
)

# Sidebar
with st.sidebar:
    st.subheader("Controls")

    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

# File Upload
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file is not None:
    with st.spinner("Processing PDF..."):
        # save file
        temp_path = uploaded_file.name
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # load pdf
        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        # split
        chunks = text_splitter.split_documents(docs)

        # IMPORTANT: E5 requires "passage:" prefix
        for chunk in chunks:
            chunk.page_content = f"passage: {chunk.page_content}"

        # reset DB for new PDF
        vectorstore.reset_collection()

        # add to vector store
        vectorstore.add_documents(chunks)

        st.session_state.vectorstore_ready = True

        st.success("PDF processed. You can now chat!")

#history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# -------------------------
# Chat Input
# -------------------------
if st.session_state.vectorstore_ready:
    if prompt := st.chat_input("Ask a question about the PDF"):
        # store user message
        st.session_state.messages.append({
            "role": "user",
            "content": prompt
        })

        # retriever (E5 requires query: prefix)
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}
        )

        docs_retrieved = retriever.invoke(f"query: {prompt}")

        if not docs_retrieved:
            response_text = "I can't answer that."
        else:
            context = "\n\n".join(doc.page_content for doc in docs_retrieved)

            prompt_text = f"""
You are a helpful assistant.

Answer the question using ONLY the context below.
If the answer is not in the context, say "I can't answer that."

Context:
{context}

Conversation so far:
{format_chat_history(st.session_state.messages)}

Question:
{prompt}
"""
            response = llm.invoke(prompt_text)
            response_text = response.content

        # store assistant message
        st.session_state.messages.append({
            "role": "assistant",
            "content": response_text
        })

        st.chat_message("assistant").write(response_text)
else:
    st.info("Upload a PDF to start chatting.")
