import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pathlib import Path
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.utilities import SQLDatabase
from sqlalchemy import create_engine
import pandas as pd
from langchain_community.agent_toolkits import create_sql_agent



class DocumentAnalysis:
    def __init__(self, file_path, model_name="gpt-4", temperature=0.3):
        load_dotenv()  
        self.configure_environment()
        self.file_path = file_path
        self.model_name = model_name
        self.temperature = temperature
        self.prompt_template = """
        Answer the question as detailed as possible from the provided context and make conclusions in a non technical manner, make sure to provide all the details, if the answer is not in
        provided context just say, 'answer is not available in the context', don't provide the wrong answer\n\n
        Context:\n {context}?\n
        Question: \n{question}\n
        Answer:
        """

    def configure_environment(self):
        print("Environment configured for LangChain and OpenAI.")

    def load_document(self):
        loader = PyPDFLoader(self.file_path)
        pages = loader.load()
        return pages  

    def split_text(self, pages, chunk_size=1000, chunk_overlap=150):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        splits = text_splitter.split_documents(pages)
        return splits 

    def create_vector_store(self, splits):
        vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
        retriever = vectorstore.as_retriever()
        return retriever  

    def setup_prompt_and_model(self):
        prompt = ChatPromptTemplate.from_template(self.prompt_template)
        model = ChatOpenAI(model_name=self.model_name, temperature=self.temperature)

        return prompt, model   

    def answer_query(self, retriever, prompt, model, question):
        chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt | model | StrOutputParser()
        )
        answer = chain.invoke(question)
        return answer 
    








class RiskProfileAnalysis:
    def __init__(self, high_risk_file, low_risk_file, medium_risk_file, db_path="sqlite:///risk.db"):
        self.high_risk_file = high_risk_file
        self.low_risk_file = low_risk_file
        self.medium_risk_file = medium_risk_file
        self.db_path = db_path

    def load_data(self):
        df_high = pd.read_csv(self.high_risk_file)
        df_low = pd.read_csv(self.low_risk_file)
        df_medium = pd.read_csv(self.medium_risk_file)
        return {
            "df_high": df_high,
            "df_low": df_low,
            "df_medium": df_medium,
        }

    def setup_database(self, df_high, df_low, df_medium):
        engine = create_engine(self.db_path)
        df_high.to_sql("high_risk", engine, index=False, if_exists='replace')
        df_low.to_sql("low_risk", engine, index=False, if_exists='replace')
        df_medium.to_sql("medium_risk", engine, index=False, if_exists='replace')
        db = SQLDatabase(engine=engine)
        return db

    def query_database(self, db, query):
        result = db.run(query)
        return result

    def setup_chat_agent(self, db, model_name="gpt-4"):
        agent_executor = create_sql_agent(ChatOpenAI(model=model_name), db=db, agent_type="openai-tools", verbose=True)
        return agent_executor

    def ask_question(self, agent_executor, question):
        response = agent_executor.invoke({"input": question})
        output_parser = StrOutputParser()
        response = output_parser.parse(response)
        return response['output']











class InsuranceRiskProfileAnalyzer:
    def __init__(self, model_name="gpt-4", temperature=0.3):
        self.model_name = model_name
        self.temperature = temperature
        self.prompt_template = """
        Given the following details of a potential customer's auto insurance profile:\n\n
        Context:\n {context}?\n
        Analyze and provide a concise summary of the insights that can be derived from this data regarding the customer's risk profile and potential impact on insurance strategy. Consider aspects such as premium pricing, risk mitigation, and strategic considerations for insurance policy adjustments based on the provided data.
        Make sure you keep it concise and clear.
        Answer:
        """

    def get_prompt(self):
        return ChatPromptTemplate.from_template(self.prompt_template)

    def get_model(self):
        return ChatOpenAI(model_name=self.model_name, temperature=self.temperature)

    def get_chain(self, prompt, model):
        return prompt | model | StrOutputParser()

    def analyze_risk_profile(self, context):
        """
        Takes a context dictionary describing a potential customer's auto insurance profile and returns a concise summary.
        """
        prompt = self.get_prompt()
        model = self.get_model()
        chain = self.get_chain(prompt, model)
        formatted_context = self.format_context(context)
        response = chain.invoke({"context": formatted_context})
        return response

    def format_context(self, context):
        """
        Format the context dictionary into a string that fits the prompt template.
        """
        context_str = ", ".join(f"{key}: {value}" for key, value in context.items())
        return context_str