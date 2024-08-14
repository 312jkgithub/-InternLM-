from langchain_community.embeddings import HuggingFaceEmbeddings


class Embedd:
    def __init__(self,embedding_model_path,batch_size: int=1,normalize_embeddings :bool =True):
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model_path,
            model_kwargs={'device': 'cuda'},
            encode_kwargs={
                'batch_size': batch_size,
                'normalize_embeddings': normalize_embeddings
            })
        self.embeddings.client = self.embeddings.client.half()
