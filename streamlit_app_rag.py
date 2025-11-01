import os
from dotenv import load_dotenv
import streamlit as st
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
import tempfile

# Load environment variables
load_dotenv()

# Azure Config
azure_config = {
    "EMBEDDINGS_API_KEY": os.getenv("AZURE_OPENAI_EMBEDDING_API_KEY"),
    "EMBEDDINGS_ENDPOINT": os.getenv("AZURE_OPENAI_EMBEDDING_ENDPOINT"),
    "EMBEDDINGS_API_VERSION": os.getenv("AZURE_OPENAI_EMBEDDING_API_VERSION"),
    "EMBEDDINGS_DEPLOYMENT": os.getenv("AZURE_OPENAI_EMBEDDING_API_MODEL"),
    "CHAT_API_KEY": os.getenv("AZURE_OPENAI_CHAT_API_KEY"),
    "CHAT_ENDPOINT": os.getenv("AZURE_OPENAI_CHAT_ENDPOINT"),
    "CHAT_API_VERSION": os.getenv("AZURE_OPENAI_CHAT_API_VERSION"),
    "CHAT_DEPLOYMENT": os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"),
}

CHROMA_PATH = os.path.join(os.getcwd(), "chroma_db")

st.set_page_config(page_title="PDF Chatbot", layout="wide")
st.title("📘Chat with PDFs – A RAG-Powered AI Assistant")

st.markdown("""
1. **Upload Your PDFs** — Each PDF page is treated as one document chunk.
2. **Ask Questions** — Query the content directly from your uploaded PDFs.
""")


def main():
    st.header("Ask a Question")

    # Validate config
    missing = [k for k, v in azure_config.items() if not v]
    if missing:
        st.error(f"Missing Azure config: {', '.join(missing)}. Please set these environment variables.")
        return

    # Initialize Embeddings and Vector Store
    embeddings = AzureOpenAIEmbeddings(
        azure_deployment=azure_config["EMBEDDINGS_DEPLOYMENT"],
        api_key=azure_config["EMBEDDINGS_API_KEY"],
        azure_endpoint=azure_config["EMBEDDINGS_ENDPOINT"],
        openai_api_version=azure_config["EMBEDDINGS_API_VERSION"]
    )

    vectorstore = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    with st.sidebar:
        st.title("Menu")
        uploaded_files = st.file_uploader(
            "Upload PDF Files", accept_multiple_files=True, key="pdf_uploader"
        )

        if st.button("Submit & Process", key="process_button") and uploaded_files:
            with st.spinner("Processing PDFs..."):
                for uploaded_file in uploaded_files:
                    try:
                        # Save uploaded PDF temporarily
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                            tmp_file.write(uploaded_file.getvalue())
                            tmp_path = tmp_file.name

                        # Load PDF pages
                        loader = PyPDFLoader(tmp_path)
                        pages = loader.load()

                        for i, page in enumerate(pages):
                            page.metadata["file_name"] = uploaded_file.name
                            page.metadata["page_number"] = i + 1

                        # No text splitting — directly use page documents
                        vectorstore.add_documents(pages)
                        os.unlink(tmp_path)

                    except Exception as e:
                        st.error(f"Error processing {uploaded_file.name}: {str(e)}")
                        continue

                vectorstore.persist()
                st.success("✅ PDF(s) processed and stored in Chroma DB!")

        if st.button("Clear Chat", key="clear_chat"):
            st.session_state.chat_history = []

    st.markdown("---")
    st.subheader("💬 Chat with your documents")

    # Display chat history
    for chat in st.session_state.chat_history:
        if chat["role"] == "user":
            st.markdown(
                f"<div style='text-align: right; color: #1a73e8;'><b>You:</b> {chat['content']}</div>",
                unsafe_allow_html=True)
        else:
            st.markdown(
                f"<div style='text-align: left; color: #34a853;'><b>Bot:</b> {chat['content']}</div>",
                unsafe_allow_html=True)
            if "sources" in chat:
                for i, meta in enumerate(chat["sources"], 1):
                    st.markdown(
                        f"<div style='font-size: 0.9em; color: #888;'>Source [{i}]: "
                        f"<b>{meta.get('file_name', 'unknown')}</b> | Page: {meta.get('page_number', 'n/a')}</div>",
                        unsafe_allow_html=True,
                    )

    # Input box for user question
    user_question = st.text_input("Type your question and press Enter", key="user_question")

    if user_question:
        # Append user query to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_question})

        # Retrieve top-k relevant pages (no text splitter)
        retriever = vectorstore.as_retriever(search_type='similarity', search_kwargs={'k': 5})
        docs = retriever.invoke(user_question)
        context = "\n\n".join([doc.page_content for doc in docs])

        # Azure Chat LLM
        llm = AzureChatOpenAI(
            azure_deployment=azure_config["CHAT_DEPLOYMENT"],
            api_key=azure_config["CHAT_API_KEY"],
            azure_endpoint=azure_config["CHAT_ENDPOINT"],
            openai_api_version=azure_config["CHAT_API_VERSION"],
            temperature=0
        )

        # Manual prompt for retrieval QA
        prompt = f"""
You are a helpful assistant. Use the provided context to answer the question.
If the context doesn’t contain enough information, say "I don't have enough information."

CONTEXT:
{context}

QUESTION:
{user_question}
"""

        with st.spinner("Thinking..."):
            response = llm.invoke(prompt)

        # Save sources metadata
        sources = [doc.metadata for doc in docs]

        st.session_state.chat_history.append({
            "role": "bot",
            "content": response.content.strip(),
            "sources": sources
        })

        # ✅ Fix for infinite loop — safely clear input
        if "user_question" in st.session_state:
            st.session_state.pop("user_question")

        st.rerun()


if __name__ == "__main__":
    main()
