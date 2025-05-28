import os
import logging
from typing import List
from mcp.server.fastmcp import FastMCP
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.schema import HumanMessage
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("medical_assistant.log", mode='a')
    ]
)
logger = logging.getLogger(__name__)

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

mcp = FastMCP("MedicalAssistant")
generic_prompt = "You are a helpful medical assistant. Based on the following information, answer the user's question."

llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.2,
    openai_api_key=OPENAI_API_KEY
)

os.makedirs("faiss_indexes", exist_ok=True)

def prepare_retriever(file_path: str, index_path: str):
    if os.path.exists(index_path):
        logger.info(f"Loading existing FAISS index from: {index_path}")
        return FAISS.load_local(
            index_path,
            OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY),
            allow_dangerous_deserialization=True
        )

    logger.info(f"Creating FAISS index for: {file_path}")
    loader = TextLoader(file_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = splitter.split_documents(docs)
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectorstore = FAISS.from_documents(splits, embeddings)
    vectorstore.save_local(index_path)
    return vectorstore

retrievers = {
    "headaches": prepare_retriever("headaches.txt", "faiss_indexes/headaches"),
    "conjunctivitis": prepare_retriever("conjunctivitis.txt", "faiss_indexes/conjunctivitis"),
    "covid": prepare_retriever("covid.txt", "faiss_indexes/covid")
}

# def retrieve_and_respond(domain: str, query: str) -> str:
#     docs = retrievers[domain].similarity_search(query)
#     doc_id= docs.id
#     doc_page_content = docs.page_content
#     logger.info(f"[{domain.upper()}] Retrieved {len(docs)} documents for query: '{query}'")
#     for i, doc in enumerate(docs):
#         logger.debug(f"[{domain.upper()}] Document {i+1}: {doc.page_content[:]}")

#     if not docs:
#         return "No information found."
    
#     context = "\n".join([doc.page_content for doc in docs])
#     prompt = f"""{generic_prompt}. For source refer to the rextual description that you are recieving in the {context} and always return the whole descxription. return source in source segment.

# Context:
# {context}

# Question: {query}
# Answer:
# Source:ID:{doc_id}
#        Context{doc_page_content}:"""
#     response = llm.invoke([HumanMessage(content=prompt)])
#     logger.info(f"[{domain.upper()}] Response: {response.content[:]}")
#     return response.content
def retrieve_and_respond(domain: str, query: str) -> str:
    docs = retrievers[domain].similarity_search(query)

    if not docs:
        return "No information found."

    logger.info(f"[{domain.upper()}] Retrieved {len(docs)} documents for query: '{query}'")

    context_blocks = []
    source_blocks = []

    for i, doc in enumerate(docs):
        logger.debug(f"[{domain.upper()}] Document {i+1}: {doc.page_content}")
        context_blocks.append(doc.page_content)
        source_blocks.append(f"ID: {doc.metadata.get('id', f'doc-{i+1}')}\nContext: {doc.page_content}")

    context = "\n".join(context_blocks)
    sources = "\n\n".join(source_blocks)

    prompt = f"""{generic_prompt}. For source refer to the textual description that you are recieving in the {sources} and always return the whole descxription. return source in source segment.

Context:
{context}

Question: {query}

Answer:
Source:
{sources}
"""
    logger.info(f"source info is **********{sources}")
    response = llm.invoke([HumanMessage(content=prompt)])
    logger.info(f"[{domain.upper()}] Response: {response.content[:]}")
    return response.content

@mcp.tool()
def get_headache_info(query: str) -> str:
    return retrieve_and_respond("headaches", query)

@mcp.tool()
def get_conjunctivitis_info(query: str) -> str:
    return retrieve_and_respond("conjunctivitis", query)

@mcp.tool()
def get_covid_info(query: str) -> str:
    return retrieve_and_respond("covid", query)


# Run the MCP server
if __name__ == "__main__":
    mcp.run()