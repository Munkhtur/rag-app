import logging
logging.basicConfig(level=logging.INFO)
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
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from llama_index.llms.huggingface import HuggingFaceLLM

# import torch
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from IPython.display import Markdown, display
# base_model = AutoModelForCausalLM.from_pretrained("unsloth/Meta-Llama-3.1-8B")

# # # Load the tokenizer
# tokenizer = AutoTokenizer.from_pretrained("unsloth/Meta-Llama-3.1-8B")

# fine_tuned_model = PeftModel.from_pretrained(base_model, "history_lora_model")
llm = HuggingFaceLLM(
    context_window=4096,
    max_new_tokens=256,
    generate_kwargs={"temperature": 0.7, "do_sample": False},  
    tokenizer_name="gmunkhtur/llama3-secret-history",
    model_name="gmunkhtur/llama3-secret-history",
    device_map="auto",
    stopping_ids=[50278, 50279, 50277, 1, 0],
    tokenizer_kwargs={"max_length": 4096},
    # model_kwargs={"torch_dtype": torch.float16},
)

# class CustomLLM:
#     def __init__(self, model, tokenizer):
#         self.model = model
#         self.tokenizer = tokenizer

#     def __call__(self, prompt):
#         # Tokenize the input prompt
#         inputs = self.tokenizer(prompt, return_tensors="pt")

#         # Generate response
#         outputs = self.model.generate(**inputs, max_length=512, eos_token_id=self.tokenizer.eos_token_id)

#         # Decode the generated response
#         response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
#         return response

#     def generate(self, prompt):
#         return self.__call__(prompt)


load_dotenv()
app = Flask(__name__)

# Initialize your components here, so they are ready to be used by the API endpoints
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
Settings.llm = llm  # Set your LLM here, e.g., OpenAI, LLaMA, etc.
Settings.chunk_size = 1024
Settings.embed_model = embedding_model
# Settings.transformations = [SentenceSplitter(chunk_size=1024)]
index = VectorStoreIndex.from_vector_store(
    client=client,
    collection_name="llama_index_collection",
    embedding_model=embedding_model
)
# retriever = qdrant.as_retriever(search_kwargs={"k": 5})
compressor = FlashrankRerank(model="ms-marco-MiniLM-L-12-v2")
# compression_retriever = ContextualCompressionRetriever(
#     base_compressor=compressor, base_retriever=retriever
# )
# llm = CustomLLM(model=fine_tuned_model, tokenizer=tokenizer)


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
# Define the query string
query_str = "What causes Alstrom syndrome?"

# Set up the query engine
query_engine = index.as_query_engine()

# Run the query and display the result


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
    
    response = query_engine.query(query_str)
    display(Markdown(f"<b>{response}</b>"))
    formatted_response = format_response(response)
    
    return jsonify({"response": formatted_response})

if __name__ == "__main__":
    app.run(debug=True)
