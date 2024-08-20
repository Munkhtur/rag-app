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

load_dotenv()
app = Flask(__name__)

# Initialize your components here, so they are ready to be used by the API endpoints
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
client = QdrantClient(url="http://localhost:6333") 
embeddings = FastEmbedEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
qdrant = Qdrant(
    client=client,
    collection_name="bacteria_embeddings",
    embeddings=embeddings
)
retriever = qdrant.as_retriever(search_kwargs={"k": 5})
compressor = FlashrankRerank(model="ms-marco-MiniLM-L-12-v2")
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=retriever
)
llm = ChatGroq(temperature=0, model_name="llama3-70b-8192")

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
    formatted_response = format_response(response)
    
    return jsonify({"response": formatted_response})

if __name__ == "__main__":
    app.run(debug=True)
