# Import required libraries for Langchain and other components
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
import streamlit as st
import os
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Set Langchain environment variables for tracing and authentication
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

# Define a prompt template for the chatbot interaction
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Please respond to user queries."),
    ("user", "Question: {question}")
])

# Set up the Streamlit interface
st.title("Langchain Demo with OLLAMA ")  # Main title for the Streamlit app
st.subheader("Interact with a chatbot that uses LLAMA3 API.")  # Brief description

# Get input from the user through a text input box
input_text = st.text_input("Enter your question:", "")  # Placeholder text

# Create a Langchain with a local model (OLLAMA with LLAMA3)
llm = Ollama(model="llama3")  # Specify the model to be used
output_parser = StrOutputParser()  # Output parser to convert results to strings
chain = prompt | llm | output_parser  # Chain the components to create a workflow

# If the user enters text, process it with the chain
if input_text:
    # Generate a response from the chatbot and display it in Streamlit
    response = chain.invoke({"question": input_text})  # Get the response
    st.write("Response from the chatbot:")
    st.write(response)  # Display the response in Streamlit
