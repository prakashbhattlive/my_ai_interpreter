from typing import Any
from dotenv import load_dotenv
from langchain.embeddings import OllamaEmbeddings
from langchain.llms import Ollama
from langchain.agents import initialize_agent, AgentType, AgentExecutor
from langchain_experimental.tools import PythonAstREPLTool
from langchain_experimental.agents.agent_toolkits import create_csv_agent

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

    # CSV agent setup - this will handle episode questions directly
    csv_agent = create_csv_agent(
        llm=llm,
        path="episode_info.csv",
        allow_dangerous_code=True,
        verbose=True,
    )

    # Python REPL agent setup for other tasks
    python_tool = PythonAstREPLTool()
    python_agent_executor = initialize_agent(
        tools=[python_tool],
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True
    )

    # Simple router logic based on keywords
    def route_query(query):
        query_lower = query.lower()
        
        # If query is about episodes, sessions, or CSV data, use CSV agent
        if any(keyword in query_lower for keyword in ['episode', 'session', 'csv', 'data', 'most', 'count']):
            print("Routing to CSV Agent...")
            try:
                result = csv_agent.invoke({"input": query})
                return result.get("output", str(result))
            except Exception as e:
                return f"Error processing CSV query: {str(e)}"
        
        # For QR code generation, provide specific instructions
        elif 'qr' in query_lower or 'qrcode' in query_lower:
            print("Routing to Python Agent for QR code generation...")
            enhanced_query = """
Generate 5 QR codes pointing to https://github.com/prakashbhattlive/my_ai_interpreter and save them in the current working directory.

Use this code approach:
```python
import qrcode
import os

# Create the URL
url = "https://github.com/prakashbhattlive/my_ai_interpreter"

# Generate 5 QR codes with different names
for i in range(1, 6):
    # Create QR code instance
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=4,
    )
    
    # Add data to QR code
    qr.add_data(url)
    qr.make(fit=True)
    
    # Create image
    img = qr.make_image(fill_color="black", back_color="white")
    
    # Save image
    filename = f"qr_code_{i}.png"
    img.save(filename)
    print(f"Saved: {filename}")

print("All 5 QR codes generated successfully!")
```

If qrcode is not installed, first install it using: pip install qrcode[pil]
"""
            try:
                result = python_agent_executor.invoke({"input": enhanced_query})
                return result.get("output", str(result))
            except Exception as e:
                return f"Error executing Python code: {str(e)}"
        
        # Otherwise use Python agent for general code generation/execution
        else:
            print("Routing to Python Agent...")
            try:
                result = python_agent_executor.invoke({"input": query})
                return result.get("output", str(result))
            except Exception as e:
                return f"Error executing Python code: {str(e)}"

    # Test the episode question
    print("\n=== Testing Episode Question ===")
    episode_result = route_query("which session has the most episodes?")
    print(f"Episode query result: {episode_result}")

    # Test the QR code generation
    print("\n=== Testing QR Code Generation ===")
    qr_result = route_query("Generate and save in current working directory 5 qrcodes that point to https://github.com/prakashbhattlive/my_ai_interpreter")
    print(f"QR code result: {qr_result}")

if __name__ == "__main__":
    main()