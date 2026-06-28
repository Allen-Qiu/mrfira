"""
encoding text to embeddings
"""
from FlagEmbedding import BGEM3FlagModel

import numpy as np

class TextEncoder:
    def __init__(self, checkpoint):
        self.model = BGEM3FlagModel(checkpoint, use_fp16=False)

    def get_embedding(self, text, type=0):
        # 统一处理输入：如果是单条字符串，转为列表以触发 Fast Tokenizer 性能
        if isinstance(text, str):
            text = [text]

        if type == 0:  # Dense embedding (最常用的存储类型)
            output = self.model.encode(
                text,
                batch_size=12,
                max_length=1000,
                return_dense=True,
                return_sparse=False,
                return_colbert_vecs=False
            )
            return np.array(output['dense_vecs'])

        elif type == 1:  # Sparse embedding (用于传统搜索或混合检索)
            output = self.model.encode(text, return_dense=False, return_sparse=True)
            return output['lexical_weights']

        elif type == 2:  # Multi-vectors (ColBERT, 用于重排序)
            output = self.model.encode(text, return_colbert_vecs=True)
            return np.array(output['colbert_vecs'])
        else:
            raise ValueError("Unknown type. Use 0 for dense, 1 for sparse, 2 for multi-vectors.")

if __name__ == '__main__':
    checkpoint = encoder = '/root/autodl-tmp/bge-m3'
    encoder = TextEncoder(checkpoint)
    sentences = ["What is BGE M3?", "Defination of BM25"]
    embeddings = encoder.get_embedding(sentences)
    print(embeddings.shape)
    print(embeddings)