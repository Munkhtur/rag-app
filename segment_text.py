from langchain.text_splitter import CharacterTextSplitter
import re
from pathlib import Path
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

def segment_text():
    document_path = Path("data/secret_history.md")
    document_path.parent.mkdir(parents=True, exist_ok=True)


    loader = UnstructuredMarkdownLoader(document_path)
    print("------------loading-----------------")
    loaded_docs = loader.load()
    # Custom splitter for segmenting by poetry lines and conversations
    def custom_text_splitter(text):
        # First, split by paragraphs (or chapters if necessary)
        paragraphs = text.split("\n\n")
        
        segments = []
        for para in paragraphs:
            # Split by poetry lines (assuming new lines indicate poetry)
            lines = para.split("\n")
            
            for line in lines:
                # Split by sentence if it's a narrative or conversational
                sentences = re.split(r'(?<=[\.\!\?\…])\s+', line)
                segments.extend(sentences)
        
        return segments

    # Example usage
    segmented_texts = [custom_text_splitter(doc.page_content) for doc in loaded_docs]


    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2048, chunk_overlap=128)
    chunked_docs = []

# Iterate over each document, segment, then chunk
    for doc in loaded_docs:
        segmented_text = custom_text_splitter(doc.page_content)  # Custom segmentation
        joined_segments = " ".join(segmented_text)  # Join segments into one text block

        # Apply the RecursiveCharacterTextSplitter to get chunks
        chunked_texts = text_splitter.split_text(joined_segments)

        # Convert each chunk into a Document object
        for chunk in chunked_texts:
            chunked_docs.append(Document(page_content=chunk, metadata=doc.metadata))
    print("---------------checked docs ----------------------")
    print(chunked_docs[0])
    return chunked_docs


    # ""
    # here is part of a text to be used rag app.

    # create data augmentation to fine tune model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2') 
    # in the following csv format
    # sentence1,sentence2,label
    # "Шижуудай Доголдай хоёр бүлгээ.","Шижуудай, Доголдай хоёр бүлэгтэй.",1. 
    # create as many examples as possible