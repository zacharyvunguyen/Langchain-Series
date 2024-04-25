import requests
import streamlit as st
import json

# Function to get response from OpenAI endpoint
def get_openai_response(input_text):
    try:
        response = requests.post(
            "http://localhost:8000/essay/invoke",
            json={'input': {'topic': input_text}}
        )

        # Check if the response is successful
        if response.status_code != 200:
            raise ValueError("Failed to get response from OpenAI endpoint")

        # Attempt to parse JSON from the response
        data = response.json()  # This can raise JSONDecodeError
        return data['output']['content']  # Access the content from the parsed JSON

    except (requests.RequestException, ValueError, KeyError, json.JSONDecodeError) as e:
        return f"Error getting response from OpenAI: {str(e)}"


# Function to get response from OLLAMA endpoint
def get_ollama_response(input_text):
    try:
        response = requests.post(
            "http://localhost:8000/poem/invoke",
            json={'input': {'topic': input_text}}
        )

        if response.status_code != 200:
            raise ValueError("Failed to get response from OLLAMA endpoint")

        data = response.json()
        return data['output']  # Access the output content

    except (requests.RequestException, ValueError, KeyError, json.JSONDecodeError) as e:
        return f"Error getting response from OLLAMA: {str(e)}"


# Streamlit framework
st.title('Langchain Demo with LLAMA3 API')  # Main title for the Streamlit app

# Text input fields for different tasks
essay_topic = st.text_input("Write an essay on:")
poem_topic = st.text_input("Write a poem on:")

# Check if text is entered, then call the respective function
if essay_topic:
    st.write("Essay Response:")
    st.write(get_openai_response(essay_topic))

if poem_topic:
    st.write("Poem Response:")
    st.write(get_ollama_response(poem_topic))
