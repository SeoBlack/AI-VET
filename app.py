from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.memory import ConversationSummaryBufferMemory
from Retrieval import Retrieval
from dotenv import load_dotenv
from agent import Agent

load_dotenv()


systemMessage = """
    You are a Virtual Vet. "
    "You should help clients with their concerns about their pets and provide helpful solutions."
    "You can ask questions to help you understand and diagnose the problem."
    "You should only talk within the context of problem."
    "If you are unsure of how to help, you can suggest the client to go to the nearest clink of their place."
    "You should talk on Finnish, unless the client talks in English."
    """



#the main chatbot instance
class LLMChat: 
    def __init__(self):
        self.llm = ChatOpenAI ( # initialize our LLM 
            temperature=0,
            model='gpt-3.5-turbo'
        )
        #Prompt to be used
        # self.prompt = ChatPromptTemplate(messages=[
        #     SystemMessagePromptTemplate.from_template(template=systemMessage),#system message to instruct the chatbot.
        #     MessagesPlaceholder(variable_name="history"),#the conversation memory that is added to the prompt by the ConversationBufferMemory
        #     HumanMessagePromptTemplate.from_template("{question}") # the user question 
        #     ]
        # )
        self.memory= ConversationSummaryBufferMemory(memory_key='chat_history',max_token_limit=100,return_messages=True,llm=self.llm) #
        self.retrieval = Retrieval()
        self.agent = Agent(llm=self.llm,memory=self.memory,vectorstore=self.retrieval.retriever)
        self.executor = self.agent.executor
    def conversation(self, question):
        # dataRetrieval = Retrieval()
        # relativeData = dataRetrieval.retrieveData(query=question)
        # print(relativeData)
        response = self.executor.run({'input':question})
        return response
    

def main():
    chat = LLMChat()
    while True:
        question = input("Ask: ")
        if not question:
            break
        response = chat.conversation(question=question)
        print(f'{response}\n')
    print("ended.")


if __name__ == '__main__':
    main()