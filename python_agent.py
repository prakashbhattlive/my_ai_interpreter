from dotenv import load_dotenv
from langchain.embeddings import OllamaEmbeddings
from langchain.llms import Ollama
from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor
from langchain_experimental.tools import PythonAstREPLTool ## Package need to scoute & add guard_rails for safety


# Load environment variables
load_dotenv()

# Local embedding configuration
OLLAMA_URL = "http://192.168.1.6:11434" # Replace with your local Ollama server URL or localhost if running locally in a same machine
OLLAMA_MODEL = "mxbai-embed-large:latest"

# Initialize embedding model
embedding_model = OllamaEmbeddings(
    model=OLLAMA_MODEL,
    base_url=OLLAMA_URL
)

# Initialize LLM
llm = Ollama(
    model="llama3:latest",
    base_url=OLLAMA_URL
)

def main():
    print("Start...")

    instructions = """ You are an agent designed to write and execute python code to answer questions.
    You have access to a python REPL, which you can use to execute python code.
    If you get an error, debug your code and try again.
    Only use the output of your code to answer the questions.
    You might know the answer without running any code, but you should still run the code to get the answer.
    If it does not seem like you can write code to answer the question, just return "I don't know" as the answer.
    """

    base_prompt = hub.pull("langchain-ai/react-agent-template")
    prompt = base_prompt.partial(instructions=instructions)

    tools = [PythonAstREPLTool()]
    agent = create_react_agent(
        prompt=prompt,
        llm=llm,
        tools=tools,
    )


    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    agent_executor.invoke(
        input={
            "input": """generate and save in current working directory 2 QRcodes
            that point the https://github.com/prakashbhattlive/my_ai_interpreter, you have qrcode package installed already"""
        
        }
    )

if __name__ == '__main__':
    main()


