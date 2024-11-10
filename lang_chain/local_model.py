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
#%%
llm = OllamaLLM(model="llama3.2")


#%%
system_prompt =  "Faça o que é pedido"
user_prompt = "Diga em qual capitulo o personagem  {input} de jujutsu kaisen morreu"


prompt = ChatPromptTemplate.from_messages([('system', system_prompt),
                                          ('user', user_prompt)])
#%%

chain = prompt | llm


#%%
chain.invoke({'input': 'gojo satoru'})

#%%

loader = WebBaseLoader(web_paths=["https://www.yahoo.com/entertainment/jujutsu-kaisen-did-gojo-satoru-053816403.html?guccounter=1&guce_referrer=aHR0cHM6Ly93d3cuZ29vZ2xlLmNvbS8&guce_referrer_sig=AQAAACT0Ba7KyRzNBSDAsMnrlj3vFhYVkhCkC6JhqEyoG0qJB4P-jyMhsQd_qiXGcPU5JwQgCCWNiaET7kJkaSqBkR-SnKPxAqdRH_uiCmP6rzYK0ai943Edf0bosfKh1aIZ2MNPJnNzip_tohp2p_aGBpfVHgn7BJyMIL_wApmGJ2Zo"])
 
#%%
docs = loader.load()
#%%
from langchain_text_splitters import RecursiveCharacterTextSplitter

#%%
text_spliter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)

#%%%
splits = text_spliter.split_documents(docs)


#%%
embedding = ollama_emb = OllamaEmbeddings( 
  model='mxbai-embed-large'
)

#%%
vector_store = Chroma.from_documents(documents=splits, embedding=embedding)

#%%
#k: quantidade de elementos recuperados
retriver = vector_store.as_retriever(search_type = 'similarity',
                                     search_kwargs={"k": 6})


#%%
system_after_rag = "Voce é um assistente q ira ajudar respondar as seguintes questoes"
user_rag_prompt = """Responda a questao baseado somente no seguinte contexto {context}
Questao: {question}"""

after_rag_prompt = ChatPromptTemplate([system_after_rag, user_rag_prompt])

#%%
after_rag_prompt


#%%
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)
#%%
chain_rag = ({'context': retriver | format_docs,
              'question': RunnablePassthrough()}
              |after_rag_prompt
              | llm
              |StrOutputParser())
#%%
chain_rag.invoke("quando gojo satoru morreu?")

#%%
chain.invoke("gojo satoru")