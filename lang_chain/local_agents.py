#%%
from langchain_ollama.llms import OllamaLLM
import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
import ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


from langchain_core.tools import Tool
from langchain.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
#%%
wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=3000))
#%%

wikipedia('gojo')

#%%
wikipedia_tool = Tool(name = 'wikipedia', 
                      description='''Use this tool when the user asks 
                      for factual information, historical context, definitions, or 
                      detailed explanations on general knowledge topics that are 
                      likely to be well-documented on Wikipedia. 
                      Avoid using this tool for subjective questions, 
                      highly specific or niche topics, 
                      or when the user requests personal opinions or creative inputs.''',
                      func=wikipedia)

#%%

from langchain.agents import AgentExecutor, create_react_agent
from langchain import hub
#%%

llm = OllamaLLM(model="llama3.2")
prompt = hub.pull('hwchase17/react')
tools = [wikipedia_tool]
#%%
agent = create_react_agent(llm, tools, prompt)

#%%
agent_executor = AgentExecutor.from_agent_and_tools(agent, tools, 
                                                    verbose=True,
                                                    handling_parsing_erros=True)
#%%
resp = agent_executor.invoke({'input':'gojo is alive?'})