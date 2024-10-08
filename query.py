
from langchain.vectorstores import Qdrant
from get_embedding_function import get_embedding_function
import argparse
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

def main():
    os.environ["GROQ_API_KEY"] = "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"
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    client = QdrantClient(path="./db") 
    embeddings = FastEmbedEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    qdrant = Qdrant(
        client=client,
        collection_name="bacteria_embeddings",
        # embedding_function=embeddings
        embeddings=embeddings
    )
    args = parser.parse_args()
    query_text = args.query_text
    retriever = qdrant.as_retriever(search_kwargs={"k": 5})
    compressor = FlashrankRerank(model="ms-marco-MiniLM-L-12-v2")
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=retriever
        )
    llm = ChatGroq(temperature=0, model_name="llama3-70b-8192")
    prompt_template="""
    Use the following pieces of information to answer the user's question.
    If you don't know the asnwer, just say you don't know, don't try to make up an answer.
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
        retriever= compression_retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt, "verbose":True}
        )
    response = qa.invoke(query_text)
    print_reponse(response)

def print_reponse(response):
  response_txt = response["result"]
  for chunk in response_txt.split("\n"):
    if not chunk:
      print()
      continue
    print("\n".join(textwrap.wrap(chunk, 100, break_long_words=False)))

if __name__ == "__main__":
   main()