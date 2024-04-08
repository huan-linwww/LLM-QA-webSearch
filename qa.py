from langchain_openai import ChatOpenAI
from langchain import hub
from retrievor import q_searching
from config import Config
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableLambda
cfg = Config()

llm = ChatOpenAI(model="gpt-4-32k",openai_api_base="",
                 openai_api_key="")
prompt = hub.pull("rlm/rag-prompt")


if __name__=="__main__":

    #query = '东南大学的现任校长是谁？'
    #query = '叔本华信仰什么宗教？'
    #query = '戊戌变法中创建了什么报纸？'
    query ="华山派的开山鼻祖是谁？"
    # query = '家喻户晓的角色印第安纳·琼斯是由哪个演员扮演的？'
    # query = '有什么和"凿壁偷光"意思相近的词语？'
    rag_chain = (
            {"context": RunnablePassthrough() | RunnableLambda(q_searching), "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )

    for chunk in rag_chain.stream(query):
        print(chunk, end="", flush=True)



