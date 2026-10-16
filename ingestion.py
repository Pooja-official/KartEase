import os
import shutil
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

load_dotenv(override=True)

api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
embed_model = os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-001")

PERSIST_DIR = "chroma_db"
DATA_DIR = "data"

def get_vector_store():
    embeddings = GoogleGenerativeAIEmbeddings(
        model=embed_model,
        google_api_key=api_key
    )

    # Check if a non-empty database already exists on disk
    if os.path.exists(PERSIST_DIR) and len(os.listdir(PERSIST_DIR)) > 0:
        try:
            vs = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
            # Ensure it actually has documents
            if vs._collection.count() > 0:
                print(f"--> Found existing Chroma collection at '{PERSIST_DIR}' with {vs._collection.count()} chunks.")
                print("--> Skipping re-embedding to save API quota.")
                return vs
        except Exception:
            pass

    print("--> Loading policy documents from 'data/' folder...")
    loader = DirectoryLoader(DATA_DIR, glob="*.md", loader_cls=TextLoader)
    documents = loader.load()
    print(f"--> Loaded {len(documents)} policy files.")

    # Split into chunks of 500 characters with 50 character overlap
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = text_splitter.split_documents(documents)
    print(f"--> Split documents into {len(chunks)} chunks.")

    # Preserve file name in chunk metadata
    for chunk in chunks:
        chunk.metadata["source"] = os.path.basename(chunk.metadata.get("source", "unknown"))

    print(f"--> Generating embeddings using '{embed_model}' and storing in '{PERSIST_DIR}'...")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIR
    )
    print("--> Vectors successfully written to disk!")
    return vector_store

if __name__ == "__main__":
    vs = get_vector_store()

    # Milestone 2 verification test query
    test_query = "What is the return window for phones?"
    print("\n" + "=" * 55)
    print(f"Milestone 2 Verification Query: '{test_query}'")
    print("=" * 55)

    results = vs.similarity_search_with_score(test_query, k=3)
    for i, (doc, score) in enumerate(results, 1):
        source_name = doc.metadata.get("source", "Unknown")
        print(f"\n[Result {i}] Source: {source_name} | Distance: {score:.4f}")
        preview = doc.page_content.strip().replace("\n", " ")
        print(f"Content: {preview[:150]}...")