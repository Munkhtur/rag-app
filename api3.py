from flask import Flask, request, jsonify
from langchain.vectorstores import Qdrant
from get_embedding_function import get_embedding_function
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import textwrap
import os
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from qdrant_client import QdrantClient
from dotenv import load_dotenv
from langchain.embeddings.base import Embeddings
from sentence_transformers import SentenceTransformer
from langchain_fireworks import Fireworks


load_dotenv()
app = Flask(__name__)

# Initialize your components here, so they are ready to be used by the API endpoints
os.environ["FIREWORKS_API_KEY"] = "fw_3ZX4yv1JWAqk86NGtaZz2v7G"
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
client = QdrantClient(url="http://localhost:6333") 
# embeddings = FastEmbedEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
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
qdrant = Qdrant(
    client=client,
    collection_name="secret_history",
    embeddings=embedding_model
)
retriever = qdrant.as_retriever(search_kwargs={"k": 5})
compressor = FlashrankRerank(model="ms-marco-MiniLM-L-12-v2")
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=retriever
)
llm = Fireworks( temperature=0,
    model="accounts/gmunkhtur-df7f59/models/llama-history", max_tokens=200
)

prompt_template="""
Use the following pieces of information to answer the user's question.
If you don't know the answer, just say you don't know, don't try to make up an answer.
Answer the questions in Mongolian.

Context: {context}
Question: {question}

Answer the question and provide additional helpful information in Mongolian, based on the pieces of information, if applicable. Be succinct.

Responses should be properly formatted to be easily read.
"""

prompt = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=compression_retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt, "verbose":True}
)

def format_response(response):
    response_txt = response["result"]
    formatted_response = []
    for chunk in response_txt.split("\n"):
        if not chunk:
            formatted_response.append("")
            continue
        formatted_response.append("\n".join(textwrap.wrap(chunk, 100, break_long_words=False)))
    return "\n".join(formatted_response)

@app.route("/query", methods=["POST"])
def query():
    data = request.json
    query_text = data.get("query_text")
    if not query_text:
        return jsonify({"error": "query_text is required"}), 400
    
    response = qa.invoke(query_text)
    print(response, "response")
    formatted_response = format_response(response)
    
    return jsonify({"response": formatted_response})

if __name__ == "__main__":
    app.run(debug=True)
