try:
    from fastapi import FastAPI
    from langchain.prompts import ChatPromptTemplate
    #from langchain.chat_models import ChatOpenAI
    from langchain_community.chat_models import ChatOpenAI  # Instead of langchain.chat_models
    from langserve import add_routes
    import uvicorn
    import os
    from langchain_community.llms import Ollama
    from dotenv import load_dotenv

    load_dotenv()

    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY is not set")

    app = FastAPI(
        title="Langchain Server",
        version="1.0",
        description="A simple API Server"  # Fixed typo in 'description'
    )

    add_routes(app, ChatOpenAI(), path="/openai")

    model = ChatOpenAI()  # Ensuring OpenAI key is set
    llm = Ollama(model="llama3")

    prompt1 = ChatPromptTemplate.from_template("Write me an essay about {topic} with 100 words")
    prompt2 = ChatPromptTemplate.from_template("Write me a poem about {topic} for a 5-year-old with 100 words")

    add_routes(app, prompt1 | model, path="/essay")
    add_routes(app, prompt2 | llm, path="/poem")

    if __name__ == "__main__":
        uvicorn.run(app, host="localhost", port=8000)

except ImportError as e:
    print("ImportError:", str(e))
except ValueError as e:
    print("ValueError:", str(e))
except Exception as e:
    print("An unexpected error occurred:", str(e))
