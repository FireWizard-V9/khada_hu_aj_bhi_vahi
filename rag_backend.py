import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import BedrockEmbeddings
from langchain.vectorstores import FAISS
from langchain.indexes import VectorstoreIndexCreator
from langchain.llms.bedrock import Bedrock

def hr_index():
    # Define the data source and load data with PyPDFLoader
    data_load = PyPDFLoader("https://www.infosys.com/investors/reports-filings/Documents/COO-executive-employment-agreement2018.pdf")

    # Split text into characters, tokens, etc.
    data_split = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],
        chunk_size=1000,  # Adjusted chunk size to a more practical value
        chunk_overlap=200  # Adjusted chunk overlap to a more practical value
    )

    # Create embeddings -- client connection
    data_embeddings = BedrockEmbeddings(
        credentials_profile_name='default',
        model_id='amazon.titan-embed-text-v1'  # Ensure this is the correct parameter name
    )

    # Create vector db, store embeddings, and index for search
    data_index = VectorstoreIndexCreator(
        text_splitter=data_split,
        embedding=data_embeddings,
        vectorstore_cls=FAISS
    )

    # Create index
    db_index = data_index.from_loaders([data_load])
    return db_index

# Function to connect to Bedrock foundational model
def hr_llm():
    llm = Bedrock(
        credentials_profile_name='default',
        model_id='anthropic.claude-v2:1',  # Ensure this is the correct parameter name
        model_kwargs={
            'temperature': 0.5,
            'max_tokens_to_sample': 350,
            'top_p': 0.9
        }
    )
    return llm

# Function which searches the user prompt, best match from vector db, and sends both to llm
def hr_rag_response(index, question):
    rag_llm = hr_llm()
    hr_rag_query = index.query(question, llm=rag_llm)
    return hr_rag_query
