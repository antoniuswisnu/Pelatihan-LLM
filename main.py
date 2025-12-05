import bs4
from langgraph.prebuilt import create_react_agent
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain.chat_models import init_chat_model
from langchain.agents import AgentState
from langchain.messages import MessageLikeRepresentation
import requests
import os

from dotenv import load_dotenv
load_dotenv()

pdf_url = "https://peraturan.bpk.go.id/Download/46205/PP%20No.%2040%20Th%201996.pdf"
pdf_path = "PP_No_40_Th_1996.pdf"

if not os.path.exists(pdf_path):
    print(f"Downloading PDF from {pdf_url}...")
    response = requests.get(pdf_url)
    with open(pdf_path, 'wb') as f:
        f.write(response.content)
    print(f"PDF downloaded successfully to {pdf_path}")
else:
    print(f"PDF already exists at {pdf_path}")

loader = PyPDFLoader(pdf_path)
docs = loader.load()
print(f"Loaded {len(docs)} pages from PDF")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(docs)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vector_store = InMemoryVectorStore(embeddings)
_ = vector_store.add_documents(documents=all_splits)

@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve information to help answer a query."""
    retrieved_docs = vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs

def prompt_with_context(state: AgentState) -> list[MessageLikeRepresentation]:
    """Inject context into state messages."""
    last_query = state["messages"][-1].text
    retrieved_docs = vector_store.similarity_search(last_query)

    docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)

    system_message = (
        "Anda adalah asisten hukum dan ahli geospasial yang membantu menjawab pertanyaan berdasarkan"
        f"\n\n{docs_content}"
    )

    return [{"role": "system", "content": system_message}, *list(state["messages"])]

tools = [retrieve_context]

# prompt = (
#     "You have access to a tool that retrieves context from a blog post. "
#     "Use the tool to help answer user queries."
# )

llm = init_chat_model("gemini-2.5-flash", model_provider="google_genai")

checkpointer = MemorySaver()

agent = create_react_agent(
    model=llm,
    tools=tools,
    prompt=prompt_with_context,
    checkpointer=checkpointer
)

query = "Tanah yang dapat diberikan dengan Hak Guna Usaha adalah tanah Negara tersebut dalam pasal berapa?"

config = {"configurable": {"thread_id": "1"}}

for step in agent.stream(
    {"messages": [{"role": "user", "content": query}]},
    config=config,
    stream_mode="values",
):
    step["messages"][-1].pretty_print()