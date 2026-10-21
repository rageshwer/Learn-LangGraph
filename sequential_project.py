'''
Content creation pipeline : 
Raw data -> editor node -> scriptwriter node -> hinglish node -> o/p
======================================================================
Editor : cleans grammar and tone
Scriptwriter : formats into engaging script
Hinglish : into hinglish format
'''
import os
import warnings
warnings.filterwarnings('ignore')
from dotenv import load_dotenv
load_dotenv()
from langchain_groq import ChatGroq
from pydantic import BaseModel

# STATE :=============================================================

class PipelineState(BaseModel):
    raw_input : str
    edited_text : str=""
    scripted_text : str=""
    hinglish_output : str=""

# NODES :=============================================================
model = ChatGroq(
    model = 'openai/gpt-oss-120b',
    temperature = 0.7,
    max_completion_tokens = 200
)

def editor_node(state : PipelineState) -> dict:
    '''Stage 1 : This node cleans grammar, removes typos and refines
       the raw text'''
    prompt = (
        "You are an expert copy editor. Clean the following text."
        "Fix grammatical errors, typos and smoothen the flow of text."
        "Keep the core message intact and return the edited text."
        f"Text : {state.raw_input}"
    )
    response = model.invoke(prompt)
    return {
        'edited_text' : response.content.strip()
    }

def scripted_node(state : PipelineState) -> dict:
    '''Stage 2 : This node converts edited text into engaging script.'''
    prompt = (
        "You are an expert story teller. Take the following text "
        "and convert it into an engaging script. Keep the original "
        " message and return script."
        f"Text : {state.edited_text}"
    )
    response = model.invoke(prompt)
    return {
        'scripted_text' : response.content.strip()
    }

def hinglish_node(state : PipelineState) -> dict:
    '''Stage 3 : This node converts english script into hinglish.'''
    prompt = (
        "You are an expert translator. Tranlate the following script into "
        "hinglish natural flow."
        f"Script : {state.scripted_text}"
    )
    response = model.invoke(prompt)
    return {
        'hinglish_output' : response.content.strip()
    }

# EDGES :===============================================================
# Now lets create the graph
from langgraph.graph import StateGraph, START, END

# Add the state and nodes
graph = StateGraph(PipelineState)
graph.add_node('Editor',editor_node)
graph.add_node('Scriptwriter',scripted_node)
graph.add_node('Translator',hinglish_node)

# Now add edges
graph.add_edge(START,'Editor')
graph.add_edge('Editor','Scriptwriter')
graph.add_edge('Scriptwriter','Translator')
graph.add_edge('Translator',END)

# Compile graph, which is a runnable
app = graph.compile()

result = app.invoke({
        'raw_input':input('Enter script:')
})

print(result.get('hinglish_output'))