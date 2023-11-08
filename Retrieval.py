from langchain.document_loaders  import DirectoryLoader
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from dotenv import load_dotenv

load_dotenv()

class Retrieval:
    def __init__(self,path='./knowledge/'):
        self.loader = DirectoryLoader(path,loader_cls=CSVLoader)
        self.embedding = OpenAIEmbeddings()
        try:
            self.database = FAISS.load_local('vector_db',self.embedding)
            self.retriever = self.database.as_retriever()
        except:
            self.database = None
        
    def embedAndSaveData(self,overWrite=False):
        def saveDatabase():
            data = self.loader.load()
            self.database = FAISS.from_documents(data,self.embedding)
            self.database.save_local('vector_db')
        if overWrite:
            saveDatabase()
            print("Current vector database overwritten with new data")
        elif self.database:
            pass
        else:
            saveDatabase()
            print('Data embedded and saved into the vector store database.')
    def retrieveData(self,query:str):
        self.embedAndSaveData(False)
        similarData = self.database.similarity_search(query=query, k=3)
        results = [doc.page_content for doc in similarData]

        return results
