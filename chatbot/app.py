from langchain.agents import create_sql_agent 
from langchain.agents.agent_toolkits import SQLDatabaseToolkit 
from langchain.sql_database import SQLDatabase 
from langchain.llms.openai import OpenAI 
from langchain.agents import AgentExecutor 
from langchain.agents.agent_types import AgentType
from langchain.chat_models import ChatOpenAI
import sys
sys.path.append('../')
from credentials import OPENAI_KEY, USER, HOST, PORT, DB, PASSWORD
import chainlit as cl


@cl.on_message
async def main(message:str):
    pg_uri=f'postgresql+psycopg2://{USER}:{PASSWORD}@{HOST}:{PORT}/{DB}'
    db=SQLDatabase.from_uri(pg_uri)

    llm = OpenAI(temperature=0, openai_api_key=OPENAI_KEY, model_name='gpt-3.5-turbo')

    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    agent_executor = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    )
    
    await cl.Message(
        content=agent_executor.run(message)
    ).send()
