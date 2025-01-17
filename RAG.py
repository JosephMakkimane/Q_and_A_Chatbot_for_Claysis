# from langchain.document_loaders import CSVLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain import HuggingFaceHub

from pinecone import Pinecone
from pinecone import ServerlessSpec

import os

import warnings
warnings.filterwarnings('ignore')

#initializing the tokens required
hugging_face_api = 'HF_TOKEN'
pinecone_api = 'PC_TOKEN'

os.environ["HUGGINGFACEHUB_API_TOKEN"] = hugging_face_api
os.environ['PINECONE_API_TOKEN'] = pinecone_api

#loading pdf
loader = PyPDFLoader("/content/SpaceX_NASA_CRS-7_PressKit.pdf")
#temporarily set to accept only a single test file for speed purposes


documents = loader.load()

#chunking pdf 
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

#len(docs)

#embedding documents using hf embeddings and pinecone db
embeddings = HuggingFaceEmbeddings()

pc = Pinecone(api_key=os.environ['PINECONE_API_TOKEN'])

index_name = 'qna'

index = pc.Index(index_name)

embeds = []
for i, doc in enumerate(docs):
    embeds.append((str(i), embeddings.embed_query(str(doc))))

#creating a function to retrieve the documents that match the query the most
def retrieve_relevant_context(question, index, docs, top_k, embeddings):
    question_embedding = embeddings.embed_query(question)
    results = index.query(vector=[question_embedding], top_k=top_k, include_metadata=False)
    indices = [int(result.id) for result in results['matches']]
    contexts = [docs[i] for i in indices]
    return contexts

#defining the llm to be used
llm=HuggingFaceHub(repo_id="google/flan-t5-base", model_kwargs={"temperature":0.7, "max_length":512})

chain = load_qa_chain(llm, chain_type="stuff")

#testing the Q ans A
question = 'What is SpaceX planning to do by cooperating with NASA?'
context = retrieve_relevant_context(question, index, docs, 5, embeddings)
answer = chain.run(input_documents=context, question=question)
print(f'\nQuestion: {question}\n\nAnswer: {answer}')
