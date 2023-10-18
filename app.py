from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv

load_dotenv()


systemMessage = "You are a helpful Vet Assistant who knows different kinds of pet problems and can diagnose them using provided symptyms."



#the main chatbot instance
class LLMChat: 
    def __init__(self):
        self.llm = ChatOpenAI ( # initialize our LLM 
            temperature=0,
            model='gpt-3.5-turbo'
        )
        #Prompt to be used
        self.prompt = ChatPromptTemplate (messages=[
            SystemMessagePromptTemplate.from_template(systemMessage),#system message to instruct the chatbot.
            MessagesPlaceholder(variable_name="history"),#the conversation memory that is added to the prompt by the ConversationBufferMemory
            HumanMessagePromptTemplate.from_template("{question}") # the user question 
            ]
        )
        self.memory= ConversationBufferMemory(memory_key='history',return_messages=True) #
        self.conversationChain = LLMChain(llm=self.llm,verbose=True,memory=self.memory,prompt=self.prompt)

    def conversation(self, question):
        response = self.conversationChain({'question':question})
        return response['text']
    

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