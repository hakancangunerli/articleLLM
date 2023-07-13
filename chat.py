
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
import torch
from transformers import pipeline
from langchain import PromptTemplate, LLMChain
from langchain.llms import HuggingFacePipeline


# use dolly v2-3b

import os
os.environ['CURL_CA_BUNDLE'] = '' # per https://stackoverflow.com/a/75746105


def chatsection(prompt):
    

    raw_documents = TextLoader('test.txt',encoding='utf8').load()
    text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=0)
    documents = text_splitter.split_documents(raw_documents)

    print("calling sentence transformer")
    embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    print("calling database search")
    db = Chroma.from_documents(documents, embedding_function, persist_directory="./vectordb")

    query = prompt
    #"What was the US' economy like in 1850?"
    print("Conducting similarity search")
    docs = db.similarity_search(query)
    # print(docs[0].page_content)
    # template for an instruction with input
    prompt_with_context = PromptTemplate(
    input_variables=["instruction", "context"],
    template="{instruction}\n\nInput:\n{context}")
    
    generate_text = pipeline(model="databricks/dolly-v2-3b", torch_dtype=torch.bfloat16,
                         trust_remote_code=True, device_map="auto", return_full_text=True)
    hf_pipeline = HuggingFacePipeline(pipeline=generate_text)
    llm_context_chain = LLMChain(llm=hf_pipeline, prompt=prompt_with_context)
    
    

    # b) Load model & tokenizer

    answer = llm_context_chain.predict(instruction=query, context=docs[0].page_content).lstrip()
    
    print(answer)
    return answer