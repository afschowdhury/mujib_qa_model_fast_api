from fastapi import FastAPI, Request
from nlp_core import qna_answer as qna
import uvicorn
from nlp_core.utils import utils
from nlp_core import retrain
from fastapi.responses import JSONResponse
from threading import Thread
from langdetect import detect
import avro

app = FastAPI()





def check_language(text):
    try:
        language = detect(text)
        if language == 'bn':
            return 'Bengali'
        elif language == 'en':
            return 'English'
        else:
            return 'Other'
    except:
        return 'Error'

def banglish_to_bangla(data):
    
    return avro.parse(data)



@app.post("/ask_question")
async def read_root(request: Request):
    data = await request.json()
  
    print(data)
    print(list(data.keys()))
    if 'question' in data:
        user_input = data['question']
        
        if check_language(user_input) != 'Bengali':
            user_input = banglish_to_bangla(user_input)
            
        user_input = utils.add_question_mark(user_input)
        query = utils.change_to_mujib([user_input])[0]
        answer, confidence_score, succesfully_answered = qna.find_answer_qna(
            query)
        response = {"question": user_input,
                    "changedQuestion": query,
                    "answer": answer,
                    "score": confidence_score,
                    "isSuccessful": succesfully_answered}

        return JSONResponse(content=response, status_code=200)

    else:
        response = {"message": "Please provide a question"}
        return JSONResponse(content=response, status_code=400)


@app.post("/ask_question_qa")
async def read_root(request: Request):
    data = await request.json()
  
    print(data)
    print(list(data.keys()))
    if 'question' in data:
        user_input = data['question']
        
        if check_language(user_input) != 'Bengali':
            user_input = banglish_to_bangla(user_input)
            
        user_input = utils.add_question_mark(user_input)
        query = utils.change_to_mujib([user_input])[0]
        answer, confidence_score, succesfully_answered = qna.context_query_only_ans(
            query)
        response = {"question": user_input,
                    "changedQuestion": query,
                    "answer": answer,
                    "score": confidence_score,
                    "isSuccessful": succesfully_answered}

        return JSONResponse(content=response, status_code=200)

    else:
        response = {"message": "Please provide a question"}
        return JSONResponse(content=response, status_code=400)

# -------------------------- data add --------------------

def check_data(list2, elem1, elem2):
    if elem1 in list2 and elem2 in list2:
        return True
    else:
        return False


@app.post("/add_question")
async def read_root(request: Request):
    data = await request.json()
    print(data)
    data_fields = list(data.keys())

    data_dict = {}

    if not check_data(data_fields, "question1", "answer"):
        response = {
            "message": "Data not added! Please check the data format and try again"
        }

        return JSONResponse(content=response, status_code=400)

    data_dict["question1"] = data["question1"]
    if 'question2' in data_fields:
        data_dict["question2"] = data["question2"]
    if 'q3' in data_fields:
        data_dict["question3"] = data["question3"]
    if 'q4' in data_fields:
        data_dict["question4"] = data["question4"]
    data_dict["answer"] = data["answer"]

    response = {"data": data_dict}
    
    t1 = Thread(target=retrain.retrain, args=(data_dict,))
    t1.start()
    
    # retrain.retrain(data_dict)
    return JSONResponse(content=response, status_code=200)


if __name__ == "__main__":
    uvicorn.run("main:app", host='0.0.0.0', port=8080, reload=True)
