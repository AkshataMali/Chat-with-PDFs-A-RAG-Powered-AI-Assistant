# indexer.py
import os
import sys
import glob
import warnings
from typing import List
from dotenv import load_dotenv

load_dotenv()
warnings.filterwarnings("ignore")

ROOT = os.getcwd()
DATA_ROOT = os.getenv("DATA_ROOT", os.path.join(ROOT, "data"))
CHROMA_PATH = os.getenv("CHROMA_PATH", os.path.join(ROOT, "chroma_db"))
ALLOWED_EXTS = {".pdf", ".docx", ".txt"}

def assert_dir_exists(path, label="directory"):
    if not os.path.isdir(path):
        raise FileNotFoundError(f"{label} not found: {path}")

def discover_files(root: str, exts=ALLOWED_EXTS) -> List[str]:
    """
    Recursively discovers files under `root` whose extension is in `exts` (case-insensitive).
    """
    all_paths = glob.glob(os.path.join(root, "**", "*"), recursive=True)
    files = [
        p for p in all_paths
        if os.path.isfile(p) and os.path.splitext(p)[1].lower() in exts
    ]
    files.sort()
    return files

def load_document(file_path: str):
    """
    Loads a document into a list[Document].
    - PDF: try PyPDFLoader, fallback to PyMuPDFLoader for tough files.
    - DOCX: Docx2txtLoader
    - TXT: TextLoader
    Adds source metadata (relative path, file name, parent folder).
    """
    name, ext = os.path.splitext(file_path)
    ext = ext.lower()

    def attach_meta(docs):
        rel = os.path.relpath(file_path, DATA_ROOT)
        for d in docs:
            d.metadata.setdefault("source", rel)
            d.metadata.setdefault("file_name", os.path.basename(file_path))
            d.metadata.setdefault("parent", os.path.basename(os.path.dirname(file_path)))
        return docs

    try:
        if ext == ".pdf":
            from langchain_community.document_loaders import PyPDFLoader
            print(f"Loading PDF via PyPDFLoader: {file_path}")
            loader = PyPDFLoader(file_path, extract_images=False)
            docs = loader.load()
            if not docs:
                raise ValueError("PyPDFLoader returned no pages.")
            return attach_meta(docs)

        elif ext == ".docx":
            from langchain_community.document_loaders import Docx2txtLoader
            print(f"Loading DOCX: {file_path}")
            docs = Docx2txtLoader(file_path).load()
            return attach_meta(docs)

        elif ext == ".txt":
            from langchain_community.document_loaders import TextLoader
            print(f"Loading TXT: {file_path}")
            docs = TextLoader(file_path, encoding="utf-8").load()
            return attach_meta(docs)

        else:
            print(f"Unsupported file type: {ext} ({file_path})")
            return []

    except Exception as e:
        if ext == ".pdf":
            print(f"PyPDFLoader failed: {e}\nTrying PyMuPDFLoader ...")
            try:
                from langchain_community.document_loaders import PyMuPDFLoader
                docs = PyMuPDFLoader(file_path).load()
                return attach_meta(docs)
            except Exception as e2:
                print(f"PyMuPDFLoader also failed: {e2}")
        else:
            print(f"Error loading {file_path}: {e}")
        return []

def chunk_data(data, chunk_size=256, chunk_overlap=32):
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError:
        from langchain.text_splitter import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(data)
    print(f"Created {len(chunks)} chunks (size={chunk_size}, overlap={chunk_overlap})")
    return chunks

def create_embeddings_chroma(chunks, persist_directory=CHROMA_PATH):
    """
    Builds a Chroma DB with Azure OpenAI embeddings.
    Env vars required:
      AZURE_OPENAI_API_KEY
      AZURE_OPENAI_ENDPOINT
      AZURE_OPENAI_API_VERSION         (default: 2024-05-01-preview)
      AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT (your Azure *deployment name* for embeddings)
    """
    from langchain_openai import AzureOpenAIEmbeddings
    from langchain_community.vectorstores import Chroma

    api_key = os.getenv("AZURE_OPENAI_EMBEDDING_API_KEY")
    endpoint = os.getenv("AZURE_OPENAI_EMBEDDING_ENDPOINT")
    api_version = os.getenv("AZURE_OPENAI_EMBEDDING_API_VERSION")
    deployment = os.getenv("AZURE_OPENAI_EMBEDDING_API_MODEL")  # e.g., "embeddings"

    missing = [k for k, v in {
        "AZURE_OPENAI_API_KEY": api_key,
        "AZURE_OPENAI_ENDPOINT": endpoint,
        "AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT": deployment
    }.items() if not v]
    if missing:
        raise EnvironmentError(f"Missing Azure env vars: {', '.join(missing)}")

    embeddings = AzureOpenAIEmbeddings(
        azure_deployment=deployment,
        api_key=api_key,
        azure_endpoint=endpoint,
        openai_api_version=api_version
    )

    print(f"Building Chroma at: {persist_directory}")
    vs = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory,
    )
    vs.persist()
    print(f"Chroma vector DB persisted. ~{len(chunks)} chunks stored.")

def main():
    assert_dir_exists(DATA_ROOT, "Data root")
    print(f"Data root: {DATA_ROOT}")

    files = discover_files(DATA_ROOT, ALLOWED_EXTS)
    print(f"Found {len(files)} files:")
    for p in files:
        print(" -", os.path.relpath(p, DATA_ROOT))

    if not files:
        print("No files found. Check the path/structure and allowed extensions.")
        sys.exit(1)

    documents = []
    for f in files:
        docs = load_document(f)
        if docs:
            documents.extend(docs)
            print(f"Loaded {os.path.basename(f)} -> {len(docs)} page(s)/doc(s)")
        else:
            print(f"Skipped (no content): {f}")

    if not documents:
        print("No documents loaded. Exiting.")
        sys.exit(1)

    print("Splitting into chunks ...")
    chunks = chunk_data(documents, chunk_size=256, chunk_overlap=32)
    if not chunks:
        print("No chunks created. Exiting.")
        sys.exit(1)

    print("Creating Chroma index ...")
    create_embeddings_chroma(chunks, persist_directory=CHROMA_PATH)
    print("✅ Done.")

if __name__ == "__main__":
    main()
