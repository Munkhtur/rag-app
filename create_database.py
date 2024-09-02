import os
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.vectorstores import Qdrant
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from llama_parse import LlamaParse
import asyncio
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
import numpy as np
from langchain_qdrant import QdrantVectorStore
from uuid import uuid4
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, losses
from langchain.embeddings.base import Embeddings


load_dotenv()

def create_database(pdf_path: str, db_path: str, collection_name: str):
    print("-------------begin --------------------")
    # Set up the parsing instruction
    instruction = """
    The provided document is an ancient historical text known as "Монголын нууц товчоо" 
    (The Secret History of the Mongols). Document contains text and poems. Your task is to analyze this document and extract 
    key information such as historical events, important figures, cultural practices, and 
    notable quotes. Focus on understanding the context of Mongolian history, politics, and 
    social structures presented in the text. Answer questions in Mongolian and provide detailed, 
    accurate responses. Ensure that your responses maintain the historical and cultural integrity of the document.
    Organize your findings in a clear and structured format, summarizing the core elements of each chapter or section. 
    Emphasize precision and relevance while staying true to the original language and meaning of the text.
    """

    # Initialize the parser
    # parser = LlamaParse(
    #     api_key=os.getenv("PARSER_KEY"),
    #     result_type="markdown",
    #     parsing_instruction=instruction,
    #     max_timeout=5000,
    #     language="mn"
    # )

    # Parse the document

    
    # llama_parse_documents = parser.load_data(pdf_path)
    # print(len(llama_parse_documents), "----------")
    # parsed_doc = llama_parse_documents[0]

    # Save parsed document as Markdown
    document_path = Path("data/secret_history.md")
    document_path.parent.mkdir(parents=True, exist_ok=True)
    # with document_path.open("w", encoding="utf-8") as f:
    #     f.write(parsed_doc.text)

    # Load the Markdown file
    loader = UnstructuredMarkdownLoader(document_path)
    print("------------loading-----------------")
    loaded_docs = loader.load()

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2048, chunk_overlap=128)
    print("------------split-----------------")

    docs = text_splitter.split_documents(loaded_docs)
    print("------------client-----------------")

    client = QdrantClient(url="http://localhost:6333") 
    # Embed the documents and store them in Qdrant
    # print(client.collection_exists(collection_name="bacteria_embeddings"), "exists")
    # embeddings = FastEmbedEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    # embeddings = FastEmbedEmbeddings(model_name="./fine-tuned-model")
    class CustomSentenceTransformerEmbeddings(Embeddings):
        def __init__(self, model_path: str):
            self.model = SentenceTransformer(model_path)

        def embed_documents(self, texts):
            # This function should return the embeddings for a list of documents
            return self.model.encode(texts)

        def embed_query(self, text):
            # This function should return the embedding for a single query (optional, depending on usage)
            return self.model.encode([text])[0]


# Initialize the embedding model
    embedding_model = CustomSentenceTransformerEmbeddings('./history-fine-tuned')
    # qdrant = Qdrant.from_documents(
    #     docs, embeddings,  client=client, collection_name=collection_name
    # )
    print("------------embed-----------------")
    # embedded_docs = np.array([embeddings.embed_documents(doc.page_content) for doc in docs])

    if(client.collection_exists(collection_name=collection_name) == False):
        client.create_collection( 
    
                                collection_name=collection_name,
                        
                                vectors_config=VectorParams(size=384, distance=Distance.COSINE))
    # client.add(documents=embedded_docs, collection_name="bacteria_embeddings")

    # client.create_collection(collection_name="bacteria_embeddings", )
    print("------------vector-----------------")

    vector_store = QdrantVectorStore(
                client=client,
                collection_name=collection_name,
                embedding=embedding_model
            )
    print(client.get_collection(collection_name=collection_name))
    uuids = [str(uuid4()) for _ in range(len(docs))]
    print("------------adddoc-----------------")

    vector_store.add_documents(documents=docs, ids=uuids,)

    print(f"Qdrant database created and stored at {db_path} under collection '{collection_name}'")

# Example usage
if __name__ == "__main__":
    create_database(pdf_path="data/secret_history.pdf", db_path="./db", collection_name="secret_history")
