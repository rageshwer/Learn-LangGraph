"""
This is a college chatbot that will ask user about what programme they want to talk about. Then
it will use one of 3 nodes (academic RAG node, fee RAG node, general LLM node) to answer.
The chat will revolve around the programme selected by the user. There is a category field in the
State that will help the router function to decide which node to proceed with.
"""
#=====================================================================================
# Importing Libraries
#=====================================================================================
import warnings
warnings.filterwarnings('ignore')
import os
from dotenv import load_dotenv
load_dotenv()
from typing import Annotated,Literal
from pydantic import BaseModel

from langchain_groq import ChatGroq

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langgraph.graph import add_messages,START,END,StateGraph
#=====================================================================================
# Models and Retriever
#=====================================================================================
llm = ChatGroq(
    model = 'openai/gpt-oss-120b',
    temperature = 0.3,
    max_completion_tokens = 200
)

embedding_model = HuggingFaceEmbeddings(
    model_name = 'sentence-transformers/all-MiniLM-L6-v2'
)

def build_retriever(pdf_path : str):
    loader = PyPDFLoader(pdf_path)
    document = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(document)
    vector_store = FAISS.from_documents(chunks,embedding_model)
    return vector_store.as_retriever(search_kwargs={'k':4})

academic_retriever = build_retriever('conditional_project/academics_handbook.pdf')
fee_retriever = build_retriever('conditional_project/fee_structure.pdf')
#=====================================================================================
# State, Nodes and Router function
#=====================================================================================
class State(BaseModel):
    programme : str
    messages : Annotated[list,add_messages]     # The add_messages maintains the history.
    query_type : str='Unknown'
    retrieved_context : str='Unknown'

def classifier_node(state : State)->dict:
    '''This node classifies the user query into one of 3 types to further engage the 
    specific node required.'''

    last_message = state.messages[-1].content
    prompt = f"""
            Classify the following query into exactly one of the three categories : 'academic',
            'fee' or 'general'. Return only the category as response in one word.
            Query : {last_message}
            """
    response = llm.invoke(prompt).content.strip().lower()
    return {'query_type':response}

def academic(state : State)->dict:
    """Retrives relevant chunks from the academic pdf."""
    query = state.messages[-1].content
    docs = academic_retriever.invoke(query)
    context = "\n\n".join(doc.page_content for doc in docs)
    return {'retrieved_context':context}

def fee(state : State)->dict:
    """Retrives relevant chunks from the fee pdf."""
    query = state.messages[-1].content
    docs = fee_retriever.invoke(query)
    context = "\n\n".join(doc.page_content for doc in docs)
    return {'retrieved_context':context}

def general(state : State)->dict:
    """Returns that no context is needed, the LLM can answer on its own."""
    return {'retrieved_context':"NO_RETRIEVAL_NEEDED"}

def response_node(state : State)->dict:
    """Uses the context, programme and query to frame an appropriate LLM response."""
    programme = state.programme
    query = state.messages[-1].content
    context = state.retrieved_context

    if context == "NO_RETRIEVAL_NEEDED":
        prompt = f"""You are a helpful college assistant. Help a student with the query, who is
                 interested in the {programme} programme. Query : {query}
        """
    else : 
        prompt = f"""You are a helpful college assistant, assisting a {programme} student.
                 Use the context retrieved from the college handbook and frame an appropriate
                 answer. Context : {context} and Query : {query}
        """

    response = llm.invoke(prompt).content.strip()
    return {'messages':[('ai',response)]}

def router(state : State) -> Literal['academic_rag','fee_rag','general']:
    """This is the router function that decides which node to invoke. It returns the name (a
    string) of the node that need to be invoked.
    """
    if state.query_type == 'academic':
        return 'academic_rag'
    elif state.query_type == 'fee':
        return 'fee_rag'
    else:
        return 'general'
#=====================================================================================
# Graph
#=====================================================================================
builder = StateGraph(State)
builder.add_node('classifier',classifier_node)
builder.add_node('academic_rag',academic)           # Imp : The name need to be same as you
builder.add_node('fee_rag',fee)                     # put in the router.
builder.add_node('general',general)
builder.add_node('response',response_node)

builder.add_edge(START,'classifier')
builder.add_conditional_edges('classifier',router)
# The FAN IN:
builder.add_edge('academic_rag','response')
builder.add_edge('fee_rag','response')
builder.add_edge('general','response')

builder.add_edge('response',END)

graph = builder.compile()
#=====================================================================================
# The UI
#=====================================================================================
print("===========================================================")
print("Welcome to college assistant! Write 'exit' to end chat.")
print("===========================================================")
programme = input("Select which programme you are interested in : BBA, BCA, B.Com (H) : ")
print(f"You have selected {programme}! Ask what you want to know?")

while True:
    user_query = input("You : ")
    if user_query != 'exit':
        response = graph.invoke({'messages':[("human",user_query)],'programme':programme})
        print(f"AI :{response.get('messages')[-1].content}")
    else:
        break