from langchain_groq import ChatGroq
import os

# os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
llm = ChatGroq(temperature=0, model_name="llama3-70b-8192")

def generate_paraphrases_with_chatgroq(texts):
    paraphrases = []
    for text in texts:
        messages = [
            (
                "system",
                """You are a helpful assistant that paraphrases given text. 
                Text is in mongolian and response must in Mongolian. 
                Paraphrase the user text in multiple ways including paraphrasing, Synonym Replacement, Sentence Shuffling, 
                Contextual Replacement, Noise Injection.
                response should be in csv format 'original_text,new_text,label' where label is 0 or 1 for closenes of meaning of the texts. 1 is similar 0 is not same.
                """,
            ),
            ("human", text),
        ]
        paraphrased_text = llm.invoke(messages)  # Generate the paraphrase with ChatGroq
        paraphrases.append((paraphrased_text.content))
    return paraphrases

# Example usage
texts = ["Мөн үхэр жил, Чингис хаан зарлиг болж Сүбээдэйд төмөр тэрэг хийлгэж өгөөд Тогтоагийн хөвүүд Худу, Гал, Чулуун тэргүүтнийг нэхүүлэхээр илгээхдээ Сүбээдэйд зарлиг болгосон нь"]
paraphrasing_data = generate_paraphrases_with_chatgroq(texts)
print(paraphrasing_data)