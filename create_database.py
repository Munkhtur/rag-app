import os
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Qdrant
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from llama_parse import LlamaParse
import asyncio
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
import numpy as np
from langchain_qdrant import QdrantVectorStore
from uuid import uuid4

def create_database(pdf_path: str, db_path: str, collection_name: str):
    # Set up the parsing instruction
    instruction = """"The provided document is about bacteria and its shape, behaviour, and life.
    Answer questions in Mongolian and try to be precise."""

    # Initialize the parser
    parser = LlamaParse(
        api_key="llx-aY00BReMMoA9seMDZ8cmo0zBmyiLWneI0WwSu84WuObzMqOB",
        result_type="markdown",
        parsing_instruction=instruction,
        max_timeout=5000,
        language="mn"
    )

    # Parse the document

    
    llama_parse_documents = parser.load_data(pdf_path)
    print(len(llama_parse_documents), "----------")
    parsed_doc = llama_parse_documents[0]

    # Save parsed document as Markdown
    document_path = Path("data/parsed_bacteria.md")
    document_path.parent.mkdir(parents=True, exist_ok=True)
    with document_path.open("w", encoding="utf-8") as f:
        f.write(parsed_doc.text)

    # Load the Markdown file
    loader = UnstructuredMarkdownLoader(document_path)
    loaded_docs = loader.load()

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2048, chunk_overlap=128)
    docs = text_splitter.split_documents(loaded_docs)
    client = QdrantClient(url="http://localhost:6333") 
    # Embed the documents and store them in Qdrant
    embeddings = FastEmbedEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    # qdrant = Qdrant.from_documents(
    #     docs, embeddings,  client=client, collection_name=collection_name
    # )
    embedded_docs = np.array([embeddings.embed_documents(doc.page_content) for doc in docs])

    client.create_collection( 
 
                              collection_name="bacteria_embeddings",
                     
                               vectors_config=VectorParams(size=100, distance=Distance.COSINE))
    # client.add(documents=embedded_docs, collection_name="bacteria_embeddings")

    # client.create_collection(collection_name="bacteria_embeddings", )

    vector_store = QdrantVectorStore(
                client=client,
                collection_name="bacteria_embeddings",
                embedding=embeddings
            )
    
    uuids = [str(uuid4()) for _ in range(len(docs))]
    vector_store.add_documents(documents=docs, ids=uuids,)

    print(f"Qdrant database created and stored at {db_path} under collection '{collection_name}'")

# Example usage
if __name__ == "__main__":
    create_database(pdf_path="data/bacteria.pdf", db_path="./db", collection_name="bacteria_embeddings")
