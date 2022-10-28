from fastapi import FastAPI
import uvicorn
import os
from search_query import search

port = os.environ["PORT"]

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "Semantic Search"}

@app.get("/test")
def read_test():
    return {"Hello": "Search"}


@app.get('/search-query')
def get_news(query: str):
    result = search(query)
    answer = result
    return {'message': f', {answer}'}


if __name__ == "__main__":
    uvicorn.run("app:app", host = "0.0.0.0",port = int(port),reload=True)