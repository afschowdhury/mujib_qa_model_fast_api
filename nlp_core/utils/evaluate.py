from sentence_transformers import SentenceTransformer
import torch
from sentence_transformers.util import semantic_search


retriever_model = SentenceTransformer('afschowdhury/st_model')


def get_embeddings(sentences):
    embeddings = retriever_model.encode(sentences, convert_to_tensor=True)
    return embeddings


def evaluate(ques, index, base_embeddings, base_index, qs):

    model_idx = []
    for q in ques:
        q_emb = retriever_model.encode(q)
        query_embeddings = torch.FloatTensor(q_emb)
        hits = semantic_search(query_embeddings, base_embeddings, top_k=1)

        model_idx.append(hits[0][0]['corpus_id'])

    f_d = []
    for i in range(len(model_idx)):
        f = model_idx[i]

        f_d.append(base_index[model_idx[i]])

    count = 0
    for i in range(len(f_d)):
        if f_d[i] == index[i]:
            count += 1

    print("="*15+f" {qs} Evaluation "+"="*15)
    print()
    print(f"\tSuccessfully indexed: {count} questions out of {len(index)}")

    model_accuracy = count/len(index)
    print("\tModel's Accuracy : ", model_accuracy)
