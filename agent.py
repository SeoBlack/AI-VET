from langchain.chains import RetrievalQA
from langchain.agents.types import AgentType
from langchain.agents import initialize_agent
from langchain.tools import Tool
class Agent:
    def __init__(self,llm,memory,vectorstore):
        self.llm = llm
        self.memory = memory
        self.vectorstore = vectorstore
        self.qa = RetrievalQA.from_chain_type(llm=self.llm,chain_type="stuff",retriever=vectorstore)
        self.system_message = """
    You are a Virtual Vet assistant. follow these rules:"
    "1.You should help clients with their concerns about their pets and provide helpful solutions."
    "2.You can ask as many questions as you need to help you understand and diagnose the problem."
    "3.You should only talk within the context of problem."
    "4.You should talk in Finnish, unless the client talks in English."
    """
        self.tools = [
            Tool(
        name="qa-vet",
        func=self.qa.run,
        description="Useful when you need to answer vet questions",
    )
        ]
        self.executor = initialize_agent(
            agent = AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
            tools=self.tools,
            llm=self.llm,
            memory=self.memory,
            agent_kwargs={"system_message": self.system_message},
            verbose=True,
            handle_parsing_errors=True
        )