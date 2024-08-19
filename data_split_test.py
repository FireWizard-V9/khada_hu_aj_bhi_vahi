#import OS,Document Loader,Text Splitter,Bedrock embeddings,vector DB,VectorStoreIndex,Bedrock-LLM
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

#Define the data source and load data with pypdfLoader
data_load=PyPDFLoader("https://www.upl-ltd.com/corporate_governance_pdfs/Q7Rxtq4J16gJBNuWWrA8jL41GpSvxNtgKIz21mHQ/Global-Business-Information-Protection-Policy.pdf")

#Split text into characters,tokens..
data_split=RecursiveCharacterTextSplitter(separators=["\n\n","\n"," ",""],chunk_size=100,chunk_overlap=10)
data_sample='Private business information is among the most valuable of UPL’s assets. It can give us an edge over our competitors in the market. Like any other asset, though, it must be carefully protected.If fail to preserve the confidentiality of our private business information, the reputation and competitiveness of our Company could be damaged.' 
data_split_test=data_split.split_text(data_sample)
print(data_split_test)