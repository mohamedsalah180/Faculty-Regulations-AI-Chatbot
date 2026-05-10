import os
from dotenv import load_dotenv
from google.colab import drive,userdata

load_dotenv()

os.environ["GROQ_API_KEY"] = userdata.get('GROQ_API_KEY')

drive.mount('/content/drive')
PDF_PATH = "/content/drive/MyDrive/FEE_Chatbot/اللائحة الجديدة.pdf"

CHROMA_DIR = "/content/drive/MyDrive/FEE_Chatbot/chroma_database_new"

if os.path.exists(PDF_PATH) :
  print("PDF_PATH exists")
else:
  print("PDF_PATH does not exist")

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

loader = PyPDFLoader(PDF_PATH)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(docs)

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embedding,
    persist_directory=CHROMA_DIR
)
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 15}
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough ,RunnableParallel
from langchain_groq import ChatGroq

def format_docs(docs):
    return "\n\n".join([f"[Page {doc.metadata.get('page', '?')}] {doc.page_content}" for doc in docs])

llm = ChatGroq(
  model="llama-3.1-8b-instant",
  temperature=0.2
)

prompt = ChatPromptTemplate.from_messages([
    ("system", """
You are a smart and friendly assistant specifically for the Faculty of Electronic Engineering at Menoufia University.

Rules:
1. Only answer questions using information from the official PDF rules and regulations of the Faculty of Electronic Engineering, Menoufia University.
2. Do NOT make up answers or provide information from any other college or source.
3. If the answer is not found in the document, politely respond:
   "This information is not available in the official regulations. Please contact the student affairs office."
4. Always answer in a clear, friendly, and polite manner.
5. Include references to page numbers when relevant, if available from the document.

CONTEXT:
{context}

"""),
    ("human", "{question}")
])


# Chain
rag_chain = (
    {
        "context": retriever | format_docs,  # docs + page numbers
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)




def ask_bot(question):
    return rag_chain.invoke(question)
