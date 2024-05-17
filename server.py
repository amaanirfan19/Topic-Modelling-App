import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import json
from utils import generate_prompts, generate_topics

app = FastAPI()

class ArrayJSON(BaseModel):
    array_json: str

class DataFrameJSON(BaseModel):
    df_json: str

@app.post("/gen-prompts")
async def gen_prompts(data: DataFrameJSON):
    try:
        topic_info_df = pd.read_json(data.df_json, orient='split')
        prompts = generate_prompts(topic_info_df)

        return {"data": prompts}
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/gen-topics")
async def gen_topics(data: ArrayJSON):
    try:
        prompts = json.loads(data.array_json)
        topics = generate_topics(prompts)

        return {"data": topics}
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
