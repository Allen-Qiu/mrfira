"""
A Milvus vector database
"""
from pymilvus import MilvusClient
import numpy as np

class VectorDB:
    def __init__(self, vdb, dim=1024, newdb=False):   # newdb, 创建新的数据库，否则读入已有数据库
        self.client = MilvusClient(vdb)
        # self.client = MilvusClient(
        #                             uri=vdb,
        #                             grpc_keepalive_time=120000, # 增加到 120 秒
        #                             grpc_keepalive_timeout=20000,
        #                             grpc_keepalive_permit_without_calls=True
        #                         )
        self.dim = dim

        # 创建collection
        if newdb:
            self.client.drop_collection(collection_name="my_collection")
            self.client.create_collection(
                collection_name="my_collection",
                dimension=self.dim,
                metric_type="IP"
            )
    def close(self):
        self.client.close()

    def size(self):
        res = self.client.query(
            collection_name="my_collection",
            output_fields=["count(*)"]
        )
        return res[0]["count(*)"]

    # 将数据添加到数据库中
    # data = [{"bid": i, "vector": vectors[i]}]
    def insert(self, data):
         res = self.client.insert(collection_name="my_collection", data=data)

    def search(self, query_vectors, topk=20):
        res = self.client.search(
            collection_name="my_collection",  # target collection
            data=query_vectors,
            limit=topk,  # number of returned entities
            output_fields=["bid"],  # specifies fields to be returned
            metric_type="IP"
        )
        return res


if __name__ == "__main__":
    dbname = "reports/000100.db"
    vdb = VectorDB(dbname)
    print(vdb.size())
