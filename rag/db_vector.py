from langchain.document_loaders.markdown import UnstructuredMarkdownLoader
from langchain.document_loaders import (
    TextLoader,
    UnstructuredFileLoader)
from typing import List, Tuple, Dict
from langchain.docstore.document import Document
import numpy as np
from utils import torch_gc
from langchain.vectorstores.faiss import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain.text_splitter import MarkdownHeaderTextSplitter
from  embed import Embedd
from langchain_community.embeddings import HuggingFaceEmbeddings
from LLM import InternLM2Chat
from langchain.retrievers import BM25Retriever
import os
def splite_batch_md(files_path :List[str]):
    docs =[]
    for file  in  files_path:
         mds=splite_md(file)
         docs.extend(mds)
    return docs


def splite_md(file_path :str):
    # 按照列表顺序进行切分
    headers_to_split_on = [
        ('#', 'Header 1'),
        ('##', 'Header 2'),
        ('###', 'Header 3'),
    ]
    with open(file_path, 'r', encoding='utf-8') as markdown_file:  # 读入文件测试
        markdown_content = markdown_file.read()
    # print(markdown_content)   #解析出的所有pdf文档

    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    docs = markdown_splitter.split_text(markdown_content)
    return docs

def create_db_vector(embed :HuggingFaceEmbeddings,docs):
    # 将文档进行向量化处理
    db = FAISS.from_documents(docs, embed)
    db_path="./db_vector"
    db.save_local(folder_path=db_path, index_name='wenlv')
    print("保存成功")
    return db_path

def get_vectordb(embed: HuggingFaceEmbeddings,db_path:str='./db_vector'):
    db=FAISS.load_local(folder_path= db_path, index_name='wenlv', embeddings=embed,
                        allow_dangerous_deserialization=True,
                distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT)
    return db


# 函数接受一个整数列表，返回一个列表的列表，其中每个子列表都包含连续的整数
def seperate_list(ls: List[int]) -> List[List[int]]:
    lists = []
    ls1 = [ls[0]]
    for i in range(1, len(ls)):
        if ls[i - 1] + 1 == ls[i]:
            ls1.append(ls[i])
        else:
            lists.append(ls1)
            ls1 = [ls[i]]
    lists.append(ls1)
    return lists

def query_top_K(question,top_k,db,embeddings):
    # 知识库匹配相关度阈值，取值范围在0-1之间，SCORE越小，相关度越高，取到1相当于不筛选，建议设置在0.5左右
    scores, indices = db.index.search(np.array([embeddings.embed_documents([question])[0]], dtype=np.float32), top_k)
    print(scores, indices)
    # 用于存储找到的文档
    docs = []
    # 用于存储找到的文档的 id
    id_set = set()
    # 查看文档存储的描述id
    # print("这个是index_to_docstore_id: {0}".format(db.index_to_docstore_id))
    # 根据索引来查出文档，这个db是FAISS的信息
    store_len = len(db.index_to_docstore_id)  # 记录向量库中的向量数量
    # print("存储索引的信息的长度为:  {0}".format(store_len))
    # indices[0] -----》 [ 1 13  3  6 15  2  7  5 22 12 29 18  8 10 14  9 20 27 19 23]
    # # 遍历搜索结果的索引和得分
    for j, i in enumerate(indices[0]):
        # print("这个i的数值为: {0}".format(i))
        # print("这个j的数值为: {0}".format(j))
        # print("这个score分数的阈值是： ")
        #  知识检索内容相关度 Score, 数值范围约为0-1100，如果为0，则不生效，经测试设置为小于500时，匹配结果更精准
        # 后台默认设置的
        # print(db.score_threshold)
        # # 如果索引无效或者得分低于阈值，则忽略该结果，i为docstore_id
        if i == -1 or 0 < 1 < scores[0][j]:
            # This happens when not enough docs are returned.
            # 当没有返回足够的文档时，就会发生这种情况。跳出去
            continue
        # i是向量库中的索引值
        # j是按照顺序的数值
        # db是Fassi
        _id = db.index_to_docstore_id[i]  # 根据索引获取文档的 id
        doc = db.docstore.search(_id)  # 根据 id 在文档库中查找文档

        # 向不重复的集合中存放索引
        id_set.add(i)  # 记录文档的索引
        docs_len = len(doc.page_content)  # 记录文档的长度
        # 对找到的文档进行处理，寻找相邻的文档，尽可能将多个文档的内容组合在一起，直到达到设定的最大长度
        for k in range(1, max(i, store_len - i)):
            break_flag = False
            for l in [i + k, i - k]:
                if 0 <= l < len(db.index_to_docstore_id):
                    # 拿到索引
                    _id0 = db.index_to_docstore_id[l]
                    # 取出文本
                    doc0 = db.docstore.search(_id0)
                    # 判断当前文本的长度+第一段文本的长度大于匹配后单段上下文长度
                    if docs_len + len(doc0.page_content) > 250:
                        break_flag = True
                        # 停止循环
                        break
                    elif doc0.metadata["source"] == doc.metadata["source"]:
                        # 如果数据的来源一致，则添加文档的长度
                        docs_len += len(doc0.page_content)
                        # 在集合里边增加索引号
                        id_set.add(l)
            # 如果超出最大长度以后也是暂停
            if break_flag:
                break
    # 如果不需要对文档内容进行分块，直接返回找到的文档
    # if not db.chunk_conent:
    #     return docs
    #  如果没有找到满足条件的文档，返回空列表
    if len(id_set) == 0 :
        return []
        # print("信息为空")
    id_list = sorted(list(id_set))  # 将找到的文档的 id 排序
    # print("这个是id的集合信息{0}".format(id_list))
    id_lists = seperate_list(id_list)  # 将 id 列表分块
    # print("这个是分隔后id的集合信息{0}".format(id_lists))
    # 遍历分块后的 id 列表，将同一块中的文档内容组合在一起
    for id_seq in id_lists:
        # print("这是一些id_seq的信息: {0}".format(id_seq))
        for id in id_seq:
            # 如果id是第一个则进行搜索，拿到文档片段信息
            if id == id_seq[0]:
                _id = db.index_to_docstore_id[id]
                doc = db.docstore.search(_id)
            else:
                # 如果不是第一个文档片段信息，则需要加上首个文档片段信息
                _id0 = db.index_to_docstore_id[id]
                doc0 = db.docstore.search(_id0)
                doc.page_content += " " + doc0.page_content
        ## 检查组合后的文档是否有效
        if not isinstance(doc, Document):
            raise ValueError(f"Could not find document for id {_id}, got {doc}")
        # 找到一个最小的分数设置为doc_score
        # 计算组合后的文档的得分
        # print([scores[0][id] for id in [indices[0].tolist().index(i) for i in id_seq if i in indices[0]]])
        doc_score = min([scores[0][id] for id in [indices[0].tolist().index(i) for i in id_seq if i in indices[0]]])
        doc.metadata["score"] = int(doc_score)  # 将得分记录到文档的元数据中
        # print("输出文档的元数据信息")
        # print(doc.metadata)
        docs.append(doc)  # 将组合后的文档添加到结果列表中
    torch_gc()  # 清空缓存
    return docs


# if __name__ == '__main__':
#     embed= Embedd(r"D:\Learning_materials\Model_library\DIY_model\bce-embedding-base_v1")
#     embeddings=embed.embeddings
    # print(embeddings.embed_documents(["你好"]))
    # docs= splite_md("D:\Learning_materials\Model_library\DIY_model\文旅清洗爬虫\data\difangtechan.md")
    # print(docs)


    # from diskcache_client import diskcache_client
    #
    # MAX_HISTORY_SESSION_LENGTH = 2
    #
    # # Duration in seconds before a session expires
    # SESSION_EXPIRE_TIME = 1800
    #
    # # data_list=diskcache_client.get_list("wenlu")
    # data=diskcache_client.get("wenlu")
    # print(data)
    # print("数据不为空")
    # if len(data) == 0:
    #     print("数据空吗")
    #     folder_path = "D:\Learning_materials\Model_library\DIY_model\文旅清洗爬虫\data"
    #     files = [os.path.abspath(os.path.join(folder_path, file)) for file in os.listdir(folder_path)]
    #     # print(files)
    #     docs = splite_batch_md(files)
    #     diskcache_client.append_to_list("wenlu",
    #                                     docs,
    #                                     ttl=SESSION_EXPIRE_TIME,
    #                                     max_length=MAX_HISTORY_SESSION_LENGTH)
    #     data = diskcache_client.get_list("wenl")[::-1]
    #
    # # print(data_list)
    # retriever=BM25Retriever.from_documents(data[0])
    # ans_docs=retriever.get_relevant_documents("海南的美食")
    # # print(ans_docs)
    #
    # # db_path =create_db_vector(embeddings,docs)
    # db=get_vectordb(embeddings)
    # print("*"*100)
    # docs=query_top_K(question="海南景色",top_k=3,db=db)
    # unique_docs = set()
    # deduplicated_docs = []
    # for doc in docs:
    #     if doc.page_content not in unique_docs:
    #         unique_docs.add(doc.page_content)
    #         deduplicated_docs.append(doc)
    # for doc in ans_docs:
    #     if doc.page_content not in unique_docs:
    #         unique_docs.add(doc.page_content)
    #         deduplicated_docs.append(doc)
    # print(deduplicated_docs)
    # print(len(deduplicated_docs))
    # context = "\n".join([doc.page_content for doc in docs])
    # print(len(docs))
    # print(context)
    # from LLM import InternLM2Chat
    # path=r'D:\Learning_materials\Model_library\DIY_model\agent\ReAct\model'
    # # model= InternLM2Chat(path)
    # PROMPT_TEMPLATE = """已知信息：
    # {context}
    # 根据上述已知信息,使用已知信息里边的内容来回答用户的问题，不能修改或者添加内容。如果无法从中得到答案，请说 “根据已知信息无法回答该问题” 或 “没有提供足够的相关信息”，不允许在答案中添加编造成分，答案请使用中文。 问题是：{question}"""
    # prompt = PROMPT_TEMPLATE.replace("{question}", "海南的特色").replace("{context}", context)
    # model.chat()
    # md_page = docs[0]
    # print(f"每一个元素的类型：{type(md_page)}.",
    #       f"该文档的描述性数据：{md_page.metadata}",
    #       f"查看该文档的内容:\n{md_page.page_content[0:][:200]}",
    #       sep="\n------\n")