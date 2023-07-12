
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader
from langchain.embeddings import HuggingFaceEmbeddings
from transformers import pipeline, AutoModelForQuestionAnswering, AutoTokenizer
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader
def chatsection(prompt):
#If we have a pdf, we can convert pdf to text
# # %pip install PyPDF2
# from PyPDF2 import PdfReader
# reader = PdfReader("./swiss_constitution.pdf")
# number_of_pages = len(reader.pages)
# for i in range(number_of_pages):
#     page = reader.pages[i]
#     text = page.extract_text()
#     with open("./swiss_constitution_text.txt", "a") as f:
#         f.write(text)

    raw_documents = TextLoader('test.txt',encoding='utf8').load()
    text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=0)
    documents = text_splitter.split_documents(raw_documents)

    embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    db = Chroma.from_documents(documents, embedding_function, persist_directory="./vectordb")

    query = prompt
    #"What was the US' economy like in 1850?"
    print("Conducting Sim Search")
    docs = db.similarity_search(query)
    print(docs[0].page_content)

    model_name = "deepset/roberta-base-squad2"


    nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)
    QA_input = {
        'question': query,
        'context': docs[0].page_content
    }
    res = nlp(QA_input)

    # b) Load model & tokenizer
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # print(res)
    return res