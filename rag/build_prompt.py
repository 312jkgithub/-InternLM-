

from langchain.retrievers import BM25Retriever
from  embed import Embedd
import os
from db_vector import splite_batch_md,get_vectordb,query_top_K
from diskcache_client import diskcache_client

MAX_HISTORY_SESSION_LENGTH = 2
import configparser
import os
# 获取当前文件的绝对路径
current_file_path = os.path.abspath(__file__)
# 获取当前文件的目录路径（上一级）
parent_dir_path = os.path.dirname(current_file_path)
# 获取上两级目录的路径
grandparent_dir_path = os.path.dirname(parent_dir_path)

config = configparser.ConfigParser()
conf_path=grandparent_dir_path+'/config.ini'
print(conf_path)
config.read(conf_path)
EMBED_PATH = config['paths']['embedding_path']

# EMBED_PATH="/root/models/bce-embedding-base_v1"
# db_vector="/root/wenlv/rag/db_vector"
# Duration in seconds before a session expires
SESSION_EXPIRE_TIME = 1800
def build_wenlv_prompt(question: str):
    assert EMBED_PATH != None
    embed = Embedd(EMBED_PATH)
    embeddings = embed.embeddings
    # 从缓存中取出wenlv数据信息
    data = diskcache_client.get("wenlu")
    # print(data)
    if  data == None :
        print("数据空吗")
        folder_path = "/root/wenlv/data"
        files = [os.path.abspath(os.path.join(folder_path, file)) for file in os.listdir(folder_path)]
        # print(files)
        docs = splite_batch_md(files)
        diskcache_client.append_to_list("wenlu",
                                        docs,
                                        ttl=SESSION_EXPIRE_TIME,
                                        max_length=MAX_HISTORY_SESSION_LENGTH)
        data = diskcache_client.get_list("wenlu")[::-1]
    # print(data_list)
    retriever = BM25Retriever.from_documents(data[0])
    ans_docs = retriever.get_relevant_documents(question)
    # db_path =create_db_vector(embeddings,docs)
    db = get_vectordb(embeddings)
    print("*" * 100)
    docs = query_top_K(question=question, top_k=3, db=db,embeddings=embeddings)
    unique_docs = set()
    deduplicated_docs = []
    for doc in docs:
        if doc.page_content not in unique_docs:
            unique_docs.add(doc.page_content)
            deduplicated_docs.append(doc)
    # for doc in ans_docs:
    #     if doc.page_content not in unique_docs:
    #         unique_docs.add(doc.page_content)
    #         deduplicated_docs.append(doc)
    context = "\n".join([doc.page_content for doc in deduplicated_docs])
    PROMPT_TEMPLATE = """身份设定：你是海南文旅小助手琼琼，专门负责依据用户需求提供精准的旅游信息。您精通依托于`文档内容`与`对话历史`精准匹配用户所需的旅游咨询信息。
    1. **未主动询问旅游情况**：当用户对话中未直接提及旅游需求，您的回应为：“您好，我是海南文旅小助手琼琼。如果您正筹备旅行计划，不妨分享您向往的目的地，我将很乐意为您分享相关信息。”
    2. **仅表达旅游意向**：面对那些只透露出游意愿而未明确目的地的用户，应该进一步探究：“听起来您有出游的打算，那您心中是否有心仪的城市或景点呢？。”
    3. **若询问美食信息** ：根据文档内容进行回答，若文档内容无法回答，则回复：“很抱歉，该美食我不清楚。”
    4. 如用户提出的问题与`文档内容`无直接关联，则灵活应变，以帮助用户为核心，不拘泥于文档，提供友好、有价值的建议或引导至相关资源。
    文档内容：```
    {context}```
    user: ```{question}```
    
    """
    prompt = PROMPT_TEMPLATE.replace("{question}", question).replace("{context}", context)
    return prompt
