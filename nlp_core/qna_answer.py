from sentence_transformers import SentenceTransformer
import numpy as np
import torch
from sentence_transformers.util import semantic_search
import pickle
from nlp_core.utils import evaluate
from transformers import  pipeline

qa_model = pipeline("question-answering",model = "afschowdhury/afs-qa",tokenizer="afschowdhury/afs-qa")


retriever_model = SentenceTransformer('afschowdhury/st_model')


ques_embds = np.load('nlp_core/data/ques_embeddings.npy')

with open('nlp_core/data/questions.bin', 'rb') as f:
    ques = pickle.load(f)

with open('nlp_core/data/qna.bin', 'rb') as f:
    qnas = pickle.load(f)

context_embeddings = np.load('context_embeddings.npy')

with open('contexts.bin','rb') as f:
    contexts = pickle.load(f)


def reload_data():
    global ques_embds, ques, qnas
    ques_embds = np.load('nlp_core/data/ques_embeddings.npy')

    with open('nlp_core/data/questions.bin', 'rb') as f:
        ques = pickle.load(f)

    with open('nlp_core/data/qna.bin', 'rb') as f:
        qnas = pickle.load(f)


def context_query_only_ans(query):
  q_emb = evaluate.get_embeddings(query)
  respns = []

  query_embeddings = torch.FloatTensor(q_emb)
  hits = semantic_search(query_embeddings, context_embeddings, top_k = 4)
  # print(f"User's Question: {query}")
  for i in range(len(hits[0])):
    context = contexts[hits[0][i]["corpus_id"]]
    qa_input = {'question': query,'context': context}
    
    res = qa_model(qa_input)
    respns.append(res)
    
  max_score = max([ans['score'] for ans in respns])
  for res in respns:
    if max_score == res['score']:
      # print(f"{'*'*15} QA models best answer:  {'*'*15}")
      # print(res['answer'])
      return res['answer'],max_score,True

def find_answer_qna(query):

    q_emb = retriever_model.encode(query)
    query_embeddings = torch.FloatTensor(q_emb)
    hits = semantic_search(query_embeddings, ques_embds, top_k=1)

    found_question = ques[hits[0][0]["corpus_id"]]
    confidence_score = hits[0][0]["score"]

    if confidence_score > 0.7:

        for qna in qnas:

            if qna['question'] == found_question:

                return qna['answer'], confidence_score, True
    else:
        return "Sorry, I did not understand your question.", confidence_score, False

    return "Invalid Input. Try with another question !"
