"""
MULTI-AGENT LINKEDIN POST GENERATOR with INBUILT LLM REVIEWER
This is a linkedIN post generator with built in reviewer, that takes in user topic and uses tavily search to search 
top 3 results on the internet. Then it uses the information to draft a post. Another agent provides feedback on the
post generated and the generator adjusts accordingly. 
"""
#===================================================================================================================
# Importing Libraries
#===================================================================================================================
import warnings
warnings.filterwarnings('ignore')
import os
from dotenv import load_dotenv
load_dotenv()
from typing import Annotated,Literal
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode
from langchain_tavily import TavilySearch
from langgraph.graph import add_messages,START,END,StateGraph

#===================================================================================================================
# LLM, tool and State 
#===================================================================================================================
writer = ChatOpenAI(model = 'gpt-4o-mini',temperature=1)
reviewer = ChatGroq(
    model = 'openai/gpt-oss-120b',
    temperature = 0.3
)

search_tool = TavilySearch(max_results = 3)
tools = [search_tool]
writer_with_tools = writer.bind_tools(tools)

class State(BaseModel):
    topic : str
    messages : Annotated[list,add_messages]
    draft : str=""
    attempts : int=0
    feedback : str='First call.'
    is_approved : bool=False

#===================================================================================================================
#  Nodes and router funtion
#===================================================================================================================
def draft_generator(state : State)->dict:
    '''This node takes the user topic and uses the tavily search to gather information on the topic to draft the
    linkedIN post.'''
    prompt = f"""
                You are an expert linkedIN post generator with creative style. You have tool to update your knowledge
                if required and also the history of previous drafts and their feedbacks. Create a draft for the post.
                Address the feedback issues and generate the draft in about 30-40 words. Good draft has hook and a clear
                takeaway. Topic : {state.topic}. History : {state.messages}. Feedback {state.feedback}
                """
    response = writer_with_tools.invoke(prompt).content.strip()
    return {
        'messages' : [('ai',response)]
    }

tool_node = ToolNode(tools)
'''
See the .bind_tools only enable the llm to choose what tool to use, it cant execute on its own.
ToolNode handles the execution and state persistance on its own.
'''

def extract_draft(state : State)->dict:
    '''This node extracts the draft after the generator node has finished executing tool call and generating.
    '''
    last_message = state.messages[-1].content
    return {
        'draft':last_message
    }

def reviewer_node(state : State)->dict:
    '''This node reviews the draft and classifies it as positive and negative. If negative, the state is updated
    with feedback and sent back to the generator node. If positive, the final draft is printed.'''
    prompt = f"""
            You are an expert linkedIN post reviewer. Rate the post attached as positive or negative only. The post
            should not contain any racist, sexist or any other derogatory remarks to any community, religion or region.
            The post should be within 30-40 words, with a hook and clear takeaway. Create a constructive feedback if the 
            rating is negative.\n
            Mandatory : Start the response with "positive"/"negative" based on the feedback.\n
            Draft : {state.draft} 
            """
    response = reviewer.invoke(prompt).content.strip()
    attempt = state.attempts + 1
    if 'positive' == response.lower().split()[0]:
        is_approved = True
    else : is_approved = False
    return {
        'is_approved':is_approved,
        'feedback':response,
        'attempts':attempt
    }

# Router functions
def should_call_tool(state : State):
    '''This checks weather writer needs tool call to search information for the draft or it has the information.'''
    last_message = state.messages[-1]
    if getattr(last_message,'tool_calls',None):
        return 'tools'
    return 'extracter'

def should_stop_looping(state : State):
    '''This router function decides weather the draft needs to be updated or if its good to print.'''
    if state.is_approved == True:
        print(state.draft)
        return END
    if state.attempts >3:
        print('End of attempts!')
        return END
    else : return 'writer' 

#===================================================================================================================
# Graph building
#===================================================================================================================
builder = StateGraph(State)
builder.add_node('writer',draft_generator)
builder.add_node('tools',tool_node)
builder.add_node('extracter',extract_draft)
builder.add_node('reviewer',reviewer_node)

# See notes for its diagram and explanation :
builder.add_edge(START,'writer')

builder.add_conditional_edges('writer',should_call_tool)
builder.add_edge('tools','writer')
builder.add_edge('extracter','reviewer')
builder.add_conditional_edges('reviewer',should_stop_looping)

graph = builder.compile()

#===================================================================================================================
# UI
#===================================================================================================================
print("===========================================================")
print("Welcome to Multi-Agent LinkedIN post generator with built in LLM reviewing.")
print("===========================================================")
topic = input("Select on which topic you want your post to be? : ")
response = graph.invoke({
    'topic':topic,'messages':[('human','Write linkedIN post on the given topic.')]
    })