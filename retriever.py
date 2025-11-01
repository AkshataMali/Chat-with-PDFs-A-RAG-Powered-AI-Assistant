import os
import warnings
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
load_dotenv()

CHROMA_PATH = os.path.join(os.getcwd(), "chroma_db")

def _first_env(names, default=None, required=False, label=None):
    for n in names:
        v = os.getenv(n)
        if v:
            return v
    if default is not None:
        return default
    if required:
        pretty = ", ".join(names)
        if label:
            raise EnvironmentError(f"Missing env var for {label}: {pretty}")
        raise EnvironmentError(f"Missing env var: {pretty}")
    return None


def load_embeddings_chroma(persist_directory=CHROMA_PATH):
    from langchain_chroma import Chroma
    from langchain_openai import AzureOpenAIEmbeddings

    emb_api_key = _first_env(["AZURE_OPENAI_EMBEDDING_API_KEY"], required=True)
    emb_endpoint = _first_env(["AZURE_OPENAI_EMBEDDING_ENDPOINT"], required=True)
    emb_api_version = _first_env(["AZURE_OPENAI_EMBEDDING_API_VERSION"])
    emb_deployment = _first_env(["AZURE_OPENAI_EMBEDDING_API_MODEL"], required=True)

    embeddings = AzureOpenAIEmbeddings(
        azure_deployment=emb_deployment,
        api_key=emb_api_key,
        azure_endpoint=emb_endpoint,
        openai_api_version=emb_api_version
    )

    print("📦 Loading Chroma DB from:", persist_directory)
    vector_store = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )
    return vector_store


def manual_retrieval_qa(vector_store, query, k=3, search_type="similarity", search_kwargs=None):
    from langchain_openai import AzureChatOpenAI

    chat_api_key = _first_env(["AZURE_OPENAI_CHAT_API_KEY"], required=True)
    chat_endpoint = _first_env(["AZURE_OPENAI_CHAT_ENDPOINT"], required=True)
    chat_api_version = _first_env(["AZURE_OPENAI_CHAT_API_VERSION"])
    chat_deployment = _first_env(["AZURE_OPENAI_CHAT_DEPLOYMENT"], required=True)

    llm = AzureChatOpenAI(
        azure_deployment=chat_deployment,
        api_key=chat_api_key,
        azure_endpoint=chat_endpoint,
        openai_api_version=chat_api_version,
        temperature=0
    )

    if search_kwargs is None:
        search_kwargs = {"k": k}

    retriever = vector_store.as_retriever(search_type=search_type, search_kwargs=search_kwargs)
    docs = retriever.invoke(query)


    # Combine all retrieved chunks into one context
    context = "\n\n".join([d.page_content for d in docs])

    prompt = f"""
You are a helpful assistant. Use the following context to answer the user's question.

CONTEXT:
{context}

QUESTION: {query}

Provide a concise and factual answer based only on the given context.
If the context does not contain the answer, reply with "I don’t have enough information."
    """

    print("🧠 Sending prompt to Azure Chat model...")
    response = llm.invoke(prompt)

    # Return in a similar structure as RetrievalQA
    return {
        "result": response.content,
        "source_documents": docs,
        "query": query
    }


if __name__ == "__main__":
    vs = load_embeddings_chroma()
    question = "What can the health guide help me with?"
    resp = manual_retrieval_qa(vs, question, k=3)

    print("\n=== ANSWER ===")
    print(resp["result"])

    print("\n=== SOURCES ===")
    for i, doc in enumerate(resp.get("source_documents", []), 1):
        meta = doc.metadata or {}
        print(f"[{i}] {meta.get('source', 'unknown')}  |  page={meta.get('page', 'n/a')}")
