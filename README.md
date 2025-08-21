# SecurAssist
This is a voice enabled Customer Support Agent that answer customer queries for Cloud Defense AI. It can be communicated via voice and text. It can understand and respond to all sorts of user queries while staying in the boundaries of the given context about the organisation. Has the ability to understand and process every type of questions/queries regarding the organisations that the users might have. If the query isn't clear the agent requests for clarification until the question/query is crystal clear. In case of an escalated situation, it can alert the organisation requesting for human intervention.
Tech Stack Used:
- Langgraph
- Langchain
- FastAPI
- ChromaDB
- Python

## Models Used:
- Gemini-1.5-Flash
- Eleven Labs
- Resemble
- Speechify

## Knowladge Database:
Generated a documentation for a Cloud Security Company offering endpoint protection, etc. using ChatGPT and DeepSeek.
Used generated doccumentation for further generating 300+ frequently asked user queries and their answers.

## Agent
Created Agent using langgraph with a router that can decide on what should be the next step. Connected router with conditional edges to a generator, clarifier and a excalator. Connected Gemini-1.5-Flash to the documentation by implementing RAG for faster and accurate retrieval of data. Fine-Tuned prompts for gemini-1.5-flash to generate apt and accurate output for the text to be fed into a voice model. Agent can communicate in text as well as audio.

## Audio
Used Speech Recognition python library to understand user input and convert it to text. Connected agent to Eleven Labs API to convert generated output to conversational and fluent voice.
Well, Unfortunately I got banned on Eleven Labs for creating multiple accounts. Therefore I have introduced Speechify and Resemble AI functionalities. In the case of the api running out of credits from speechify, the program runs on resemble as a back up. Although Speechify and resemble are alternatives, none of them or any product comes close to the quality and conversational voice eleven labs is able to produce.

## Front-End
Used a very basic FastAPI and HTML template to connect the agent and its functionalities to a UI.
