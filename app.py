import numpy as np
import streamlit as st
from langchain import PromptTemplate, HuggingFaceHub, LLMChain
from langchain.llms import HuggingFacePipeline
import torch
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import CharacterTextSplitter
from langchain import HuggingFaceHub,LLMChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
import os
import textwrap
from pathlib import Path
from langchain.vectorstores import Chroma
from pathlib import Path
from langchain.chains import RetrievalQA
from langchain.document_loaders import PDFMinerLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM,pipeline

def preprocess(file_path,question):
  pdf_folder_path = file_path
  os.listdir(pdf_folder_path)
  loader = [PDFMinerLoader(os.path.join(pdf_folder_path, fn)) for fn in os.listdir(pdf_folder_path)]
  embeddings = HuggingFaceEmbeddings()
  index = VectorstoreIndexCreator(
    embedding=HuggingFaceEmbeddings(),
    text_splitter=CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)).from_loaders(loader)
  tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
  model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")
  pipe = pipeline(
    "text2text-generation",
    model=model, 
    tokenizer=tokenizer, 
    max_length=512)
  local_llm = HuggingFacePipeline(pipeline=pipe)
  chain = RetrievalQA.from_chain_type(llm=local_llm, 
                                    chain_type="stuff", 
                                    retriever=index.vectorstore.as_retriever(), 
                                    input_key="question")
  return chain.run(question)
def main():
  st.title("Bank Management")
  st.selectbox("Data Types" , ['Pdf', 'Images'])
  question=st.text_area("Enter your questions here...")
  file_path = st.file_uploader(
    "Upload pdf files", type=["pdf"], accept_multiple_files=True)
  if st.button("Get the answer"):
        result = preprocess(file_path,question)
        st.text(result)

        
        
if __name__ == '__main__':
    main()
  



  # return docs[0]
