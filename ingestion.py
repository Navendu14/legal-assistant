from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import yaml
import os


class Ingestion:
    def __init__(self,address:str):
        with open(address, "r") as f:  # accessing .yaml file
            config = yaml.safe_load(f)  #list of variables in yaml file
        try:
            self.data_dir = config["data_dir"]
            self.chunk_size = config["chunk_size"]
            self.chunk_overlap = config["chunk_overlap"]
            self.batch_size = config["batch_size"]
            self.embedding_model = config["embedding_model"]
            self.persist_directory = config["persist_directory"]
            self.collection_name = config["collection_name"]
            self.chunk_batch_size = config["chunk_batch_size"]
        except KeyError as e:
            raise KeyError(f"Missing required config key: {e}") from e

        if not os.path.isdir(self.data_dir):
            raise FileNotFoundError(f"data_dir not found: {self.data_dir}")

        print("object initialized")

    def document_loader(self):
        loader = PyPDFDirectoryLoader(  # use to create a object (loader) from PyPDFDirectoryLoader class
            self.data_dir
        )
        docs = loader.lazy_load()  # calling lazy_load() method of loader object
        print("document loaded")
        return docs

    def text_splitter(self,document):
        splitter=RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            add_start_index=True
        )

        batch = []
        for doc in document:
            batch.append(doc)
            if (len(batch) < self.batch_size):
                continue

            splits = splitter.split_documents(batch)  # giving list of documents as input to text splitter

            for split in splits:  # generator function for splitter
                yield split

            batch.clear()

        if (len(batch)):
            splits = splitter.split_documents(batch)

            for split in splits:
                yield split

            batch.clear()

    def vector_database(self):
        document=self.document_loader()

        embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model)

        vector_store = Chroma(
            collection_name=self.collection_name,       #I can store different types of data in the same Chroma database directory, example collection_name="legal_docs", collection_name="medical_docs", collection_name="chat_history"
            embedding_function=embeddings,
            persist_directory=self.persist_directory,  # save data locally
        )

        splits=self.text_splitter(document)

        chunk_batch=[]
        for chunk in splits:
            chunk_batch.append(chunk)
            if(len(chunk_batch)<self.chunk_batch_size):
                continue
            print(f"adding {self.chunk_batch_size} chunk in database")
            vector_store.add_documents(documents=chunk_batch)
            '''vector_store.add_documents() adds new documents on top of the existing ones'''
            print(f"{self.chunk_batch_size} chunks are added")
            chunk_batch.clear()

        if(len(chunk_batch)):
            vector_store.add_documents(documents=chunk_batch)

            chunk_batch.clear()

if __name__=="__main__":
    ingestion=Ingestion("config.yaml")      #config.yaml file address is given as input to constructor
    ingestion.vector_database()     #vector database is created