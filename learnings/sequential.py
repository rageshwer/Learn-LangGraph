# The first thing you create in a graph is the state
import os

#1) Typed DICT
from typing import TypedDict
class State1(TypedDict):
    topic : str
    summary : str
    score : int

#2) Pydantic
from pydantic import BaseModel,field_validator
class State2(BaseModel):
    topic : str
    summary : str
    score : int

    @field_validator
    def score_positive(cls,score):
        if score<0 or score>10:
            raise ValueError("Score must be positive and less than 10!")


# Now lets make the nodes:
from langchain_huggingface import ChatHuggingFace,HuggingFaceEmbeddings
from dotenv import load_dotenv
load_dotenv()

model = ChatHuggingFace(llm=HuggingFaceEmbeddings(
    model = 'meta-llama/Llama-3.3-70B-Instruct',
    temperature = 0.7,
    max_tokens = 200
))

# A node is nothing but a python function

def summarizer_node(state : State2) -> dict:
    '''This node will take in state and do some job and return the type 
    state also. Docstring mandatory.'''
    prompt = (
        "You are a summarizer. Create the summary of the text."
        "Text : {state.topic}"
    )
    response = model.invoke(prompt)
    return {
        'summary':response.content.strip()
    }

def scorer_node(state : State2) -> dict:
    '''This node rates the summary out of 10.'''
    prompt = (
        "Rate this summary out of 10. Summary :{state.summary}"
    )
    response = model.invoke(prompt)
    return {
        'score':int(response.content)
    }

# Now the graph is constructed using edges, so very important.


