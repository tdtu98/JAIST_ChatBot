import os
import glob

from operator import itemgetter

from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import MessagesPlaceholder, ChatPromptTemplate, PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
from langchain_core.output_parsers import StrOutputParser


import streamlit as st
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

def load_data(data_path):
        docs = []
        for i in glob.glob(data_path):
            print(i)
            loader = PyPDFLoader(i)
            docs.extend(loader.load())
        return docs

class Chatbot():

    def __init__(self):
        data_path = "./data/"
        db_path = "./db/"

        # embedding
        embedding = OpenAIEmbeddings()

        if len(os.listdir(db_path)) == 0:
            docs = load_data(data_path + "*")

            # text splitter
            text_splitter = RecursiveCharacterTextSplitter(["\n", "\n\n"], keep_separator = False)
            documents = text_splitter.split_documents(docs)

            db = Chroma().from_documents(collection_name = "JAIST",
                                        persist_directory = db_path,
                                        documents = documents,
                                        embedding = embedding)
        else:
            db = Chroma(persist_directory=db_path, embedding_function=embedding, collection_name = "JAIST")
        
            
        # retriever
        retriever = db.as_retriever(search_kwargs={"k":4}, search_type = "mmr")

        # chat model
        llm = ChatOpenAI(model_name = "gpt-3.5-turbo", temperature = 0)

        # we create a sub-chain that aims to rewrite a new question from input question and chat history
        # For example, if in history we mention about reporting missuse of funding in JAIST
        # and the next question is "what about I am outside of JAIST?", the model should know we want
        # to report missuse of fund in case we are outside of JAIST.

        system_message = """
        Given a chat history and the latest user question \
        which might reference context in the chat history, formulate a standalone question \
        which can be understood without the chat history. Do NOT answer the question, \
        just reformulate it if needed and otherwise return it as is.
        """
        sub_chat_prompt = ChatPromptTemplate.from_messages(
        [
            ("system",  system_message),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}")
        ]
        )
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            input_key = "question",
            return_messages=True
        )
        self.memory.load_memory_variables({})

        sub_chain = (
            RunnableParallel(
            chat_history = RunnableLambda(self.memory.load_memory_variables) | itemgetter("chat_history"),
            question = RunnablePassthrough()
            ) |
            sub_chat_prompt | llm | StrOutputParser()
        )

    # Our main chain that answers question based on information from retrieved documents (context)
    # chat history and rewrited question.

        template = """
        Please only use the context behind to answer question!
        -----------------------
        Context: {context}
        -----------------------
        History: {chat_history}
        =======================
        Human: {question}
        Chatbot:
        """

        prompt = PromptTemplate.from_template(template= template)

        self.chain = (
            RunnableParallel(
            question = sub_chain,
            context = retriever,
            chat_history = RunnableLambda(self.memory.load_memory_variables) | itemgetter("chat_history")
            )
            | prompt
            | llm
            | StrOutputParser()
        )

    def generate_response(self, user_input):
        response = self.chain.invoke(user_input)
        self.memory.chat_memory.add_user_message(user_input)
        self.memory.chat_memory.add_ai_message(response)
        return response
