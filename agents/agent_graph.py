import os
from dotenv import load_dotenv

load_dotenv()

from langgraph.graph import StateGraph, END
from .graph_state import AgentState
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain.schema import Document
from langchain_community.document_loaders import TextLoader

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash",temperature=0.1)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

loader = TextLoader("docs/context.txt")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
texts = text_splitter.split_documents(documents)
vectorstore = Chroma.from_documents(documents=texts, embedding=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k":3})

def retrieve_context(state: AgentState):
    """Retrieve relevant context from the knowledge base."""
    print("[Node: Retrieving context...]")
    
    docs = retriever.invoke(state["input"])
    context = "\n\n".join([doc.page_content for doc in docs] )
    return {"context": context}

def route_query(state: AgentState):
    """Router decides the next step based on conext and input."""

    escalation_keywords = ["emergency","critical","now","urgent","human","escalate","call me"]
    if any(keyword in state["input"].lower() for keyword in escalation_keywords):
        return {"next_step":"escalate"}
    
    if state["context"] and len(state["context"]) > 50:
        return {"next_step":"generate_response"}
    else:
        return {"next_step":"clarify"}

def generate_response(state: AgentState):
    """Generate a response using the context and history"""

    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""
        You are "SecurAssist", the AI support consultant for CloudDefense AI.
You are conversational, empathetic, fluent and professional.

Use the following pieces of context from the GuardianEDR knowledge base to answer the question at the end.
If you don't know the answer, just say that you don't know. Don't try to make up an answer.
If the question is not about CloudDefense AI, politely inform the user you are tuned to only answer those questions.

**CRITICAL: Be agentic and proactive. Ask clarifying questions to get to the root cause.**
**CRITICAL: Remember the conversation history. Do not ask for information already provided.**
**CRITICAL: Reply as if you are working with the user. For example: in the case of the user asking for an installation guide output the first step then ask the user to tell you once that is done only then output the second step (also individually). 

**KNOWLEDGE BASE CONTENT HANDLING:**
- For lengthy Q&A content: Extract the core essence and present it conversationally. Don't dump large blocks of text.
- For concise Q&A content: Expand slightly to make it sound natural in spoken conversation while keeping it accurate.
- Always prioritize the most relevant information from the knowledge base for the user's specific situation.
- If the knowledge base contains multiple relevant answers, synthesize them into a coherent response rather than listing them separately.

**SPOKEN RESPONSE OPTIMIZATION:**
Even for long contexts, answer as if you are verbally talking to the user, which means:
- Don't output a lot of information at once - only output the essentials suitable for listening
- Break complex information into digestible chunks with natural pauses
- Offer to continue with more details: "Would you like me to go into more detail about any of this?"
- For very long procedures, offer to guide step-by-step: "I can walk you through this process one step at a time"

****SUPER CRITICAL: The output that you give is going to go to a text to speech model so the model can speak it out loud. Follow these TTS optimization rules:

1. AVOID ALL SPECIAL CHARACTERS that might cause issues with TTS:
   - Never Ever use: *, ~, _, \, /, |, #, @, %, ^, &, +, =, <, >, [, ], {, }
   - Replace bullets with words: Instead of "*" use "bullet point" or "step"
   - Replace symbols with words: Instead of ">" use "then" or "next"
   - Replace slashes with "or" or "and" depending on context
   - Handle parentheses by rephrasing: Instead of "config (optional)" use "optional config"

2. WRITE FOR SPOKEN LANGUAGE:
   - Use contractions: "you're" instead of "you are", "it's" instead of "it is"
   - Avoid complex punctuation: Use commas and periods only
   - Spell out numbers: "five" instead of "5" for small numbers, "twenty-five" instead of "25"
   - Spell out abbreviations: "Central Processing Unit" instead of "CPU" (unless it's very common like "CPU")
   - Avoid markdown formatting of any kind
   - Use verbal lists: "First," "Second," "Finally" instead of numbered lists

3. KEEP IT CONVERSATIONAL:
   - Use phrases like "Let me walk you through this" instead of "The steps are as follows:"
   - Break long responses into multiple sentences with natural pauses
   - Use verbal cues: "Okay", "Alright", "Now", "Next", "Great question"
   - Add conversational fillers where appropriate: "Basically," "So," "Now," "Alright"

4. HANDLE TECHNICAL TERMS:
   - Spell out acronyms the first time: "Endpoint Detection and Response, or E-D-R"
   - For code or commands, speak them phonetically: "Run the command: C-D space into directory name"
   - For file paths: "Slash opt slash app slash config dot json" instead of "/opt/app/config.json"
   - For options/flags: "Dash dash verbose" instead of "--verbose"

5. ADJUST LENGTH DYNAMICALLY:
   - For complex topics: "There are several aspects to this. Let me start with the most important points..."
   - Offer follow-ups: "I've covered the basics. Would you like me to go into more detail about any specific part?"
   - Gauge user interest: "This is a detailed topic. Should I continue with more information?"
###Critical: Never Ever use: *, ~, _, \, /, |, #, @, %, ^, &, +, =, <, >, [, ], {, }
If you don't know based on the context, say so. Ask clarifying questions if needed."""),
        MessagesPlaceholder(variable_name="chat_history"),
        SystemMessage(content=f"Relevant Context:\n{state['context']}"),
        HumanMessage(content=state["input"])
    ])

    messages = prompt.invoke({"chat_history": state["chat_history"], "context":state["context"], "input":state["input"]})
    response = llm.invoke(messages)

    new_history = state["chat_history"] + [HumanMessage(content=state["input"]), response]

    return {"response":response.content, "chat_history":new_history, "next_step":"generate_response"}

def clarify_question(state: AgentState):
    """Ask user for clarification when context is lacking"""
    print("Asking for clarification...")

    clarification_prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""The user's question is unclear or you lack context. 
        Politely ask for clarification or more specific details about their GuardianEDR issue.
        **CRITICAL: Remember the conversation history. Do not ask for information already provided.**
        **CRITICAL: Reply as if you are working with the user. For example: in the case of the user asking for an installation guide output the first step then ask the use to tell you once that is done only then output the second step (also individually). 
        Even for long contexts answer as if you are verbally talking to the user, which means dont output alot of information only output the essentials that are suitable for the user to listen to if they were hearing you.
    
        ****SUPER CRITICAL: The output that you give is going to go to a text to speech model so the model can speak it out loud. make sure you handle special characters, that can cause the audio to sound rubbish, properly. For example `,~_\/.
        ###Critical: Never Ever use: *, ~, _, \, /, |, #, @, %, ^, &, +, =, <, >, [, ], {, }"""),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessage(content=state["input"])
    ])

    messages = clarification_prompt.invoke({"chat_history":state["chat_history"], "input":state["input"]})
    response = llm.invoke(messages)
    new_history = state["chat_history"] + [HumanMessage(content=state["input"]),response]

    return {"response": response.content, "chat_history":new_history, "next_step":"clarify"}

def escalate_to_human(state: AgentState):
    """Handle escalation to human support"""
    print("Escalating to human...")

    escalation_notice = "I'm connecting you with our senior support team immediately. Please hold while I transfer your call. They will have expert knowledge to handle your urgent issue."
    new_history = state["chat_history"] + [
        HumanMessage(content=state["input"]),
        AIMessage(content=escalation_notice)
    ]
    print("Escalation Required")
    return {"response": escalation_notice, "chat_history": new_history, "next_step": "escalate"}

def create_agent_graph():
    workflow = StateGraph(AgentState)
    workflow.add_node('retrieve',retrieve_context)
    workflow.add_node("route",route_query)
    workflow.add_node("generate",generate_response)
    workflow.add_node("clarify",clarify_question)
    workflow.add_node("escalate",escalate_to_human)

    workflow.set_entry_point("retrieve")

    workflow.add_edge("retrieve","route")

    workflow.add_conditional_edges(
        "route",
        lambda state: state["next_step"],
        {
            "generate_response":"generate",
            "clarify":"clarify",
            "escalate":"escalate"
        }
    )

    workflow.add_edge("generate",END)
    workflow.add_edge("clarify",END)
    workflow.add_edge("escalate",END)

    return workflow.compile()

agent_graph = create_agent_graph()