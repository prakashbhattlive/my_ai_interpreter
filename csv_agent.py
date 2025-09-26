from dotenv import load_dotenv
from langchain.embeddings import OllamaEmbeddings
from langchain.llms import Ollama
from langchain.agents import create_react_agent, AgentExecutor
from langchain_experimental.tools import PythonAstREPLTool
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain import hub

# Load environment variables
load_dotenv()

# Ollama configuration
OLLAMA_URL = "http://192.168.1.6:11434"
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
    print("Starting agents...")

    # Load ReAct prompt from LangChain hub
    react_prompt = hub.pull("hwchase17/react")  # You can customize this or use your own

    # Python REPL agent setup
    python_tool = PythonAstREPLTool()
    react_agent = create_react_agent(llm=llm, tools=[python_tool], prompt=react_prompt)
    python_agent_executor = AgentExecutor(agent=react_agent, tools=[python_tool], verbose=True)

    # CSV agent setup
    csv_agent = create_csv_agent(
        llm=llm,
        path="episode_info.csv",
        allow_dangerous_code=True,
        verbose=True,
    )

    # Invoke CSV agent
    response = csv_agent.invoke(
        input={"input": "how many columns are there in file episode_info.csv"}
    )
    print(response)

    response2 = csv_agent.invoke(
        input={"input": "which writer wrote the most episodes? how many episodes did he write?"}
    )
    print(response2)

if __name__ == "__main__":
    main()