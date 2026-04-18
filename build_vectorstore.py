"""
build_vectorstore.py
--------------------
Loads healthcare guideline documents from guidelines/ directory,
chunks them, embeds using sentence-transformers, and stores in ChromaDB.
"""

import os
import glob
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GUIDELINES_DIR = os.path.join(BASE_DIR, "guidelines")
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")


def build():
    # Load all markdown files from guidelines/
    docs = []
    for fpath in sorted(glob.glob(os.path.join(GUIDELINES_DIR, "*.md"))):
        loader = TextLoader(fpath, encoding="utf-8")
        docs.extend(loader.load())
        print(f"  Loaded: {os.path.basename(fpath)}")

    if not docs:
        print("No guideline documents found in guidelines/")
        return

    # Chunk documents
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=80,
        separators=["\n## ", "\n### ", "\n- ", "\n\n", "\n", " "],
    )
    chunks = splitter.split_documents(docs)
    print(f"  Created {len(chunks)} chunks from {len(docs)} documents")

    # Embed and store in ChromaDB
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )

    # Remove existing DB if present
    if os.path.exists(CHROMA_DIR):
        import shutil
        shutil.rmtree(CHROMA_DIR)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
    )
    print(f"  ChromaDB saved to {CHROMA_DIR} ({len(chunks)} vectors)")
    print("Done!")


if __name__ == "__main__":
    build()
