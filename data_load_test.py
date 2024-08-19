#import OS,Document Loader,Text Splitter,Bedrock embeddings,vector DB,VectorStoreIndex,Bedrock-LLM
import os
from langchain.document_loaders import PyPDFLoader

#Define the data source and load data with pypdfLoader
data_load=PyPDFLoader("https://www.upl-ltd.com/corporate_governance_pdfs/Q7Rxtq4J16gJBNuWWrA8jL41GpSvxNtgKIz21mHQ/Global-Business-Information-Protection-Policy.pdf")
data_test=data_load.load_and_split()
print(len(data_test))
print(data_test[2])