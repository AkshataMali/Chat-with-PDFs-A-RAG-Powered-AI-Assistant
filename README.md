#### Chat with PDFs – A RAG-Powered AI Assistant
This project implements a Retrieval-Augmented Generation (RAG) pipeline powered by Azure OpenAI, LangChain, and ChromaDB.
It allows you to index local documents (PDF, DOCX, TXT) into a vector database and then query them through an intelligent question-answering system that retrieves relevant chunks before generating accurate, context-aware answers.

🧩 Project Overview

This system is divided into two main components:

indexer.py – Handles document ingestion, chunking, and embedding creation using Azure OpenAI embeddings.
It builds and persists a Chroma vector database from a specified data directory.

retriever.py – Loads the Chroma vector store, performs similarity search, and uses Azure OpenAI chat models to generate factual answers based on retrieved content.


#### Steps to run this project
1. Run the indexer.py from your project root directory.This will read the hdfc_financial_statement_2024.pdf, chunk it, create embeddings and save them in the vector database<br>Hint: If indexer.py is not in project root, modify the path of the pdf file in indexer.py (line 57)
2. Make sure chroma_db file is created in your project roort directory. It contains the embeddings from the HDFC financial report.
3. Run retriever.py to test it.
4. Run streamlit_app_rag_pdf_qanda.py by issuing following command:
```bash
streamlit run streamlit_rag_app.py
```
5. You can ask any question about the document, such as "who will be a good fit for accountant role?"

### Azure: Create embeddings + chat/LLM deployments
1. Create an Azure OpenAI resource
 -->> In the Azure portal → Create a resource → search “Azure OpenAI”.
 -->> Provision the resource in a region that supports OpenAI.

2. Create an embedding model

3. Create LLM model
API Key (create or copy a key from Keys & Endpoint,model)
