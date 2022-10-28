# Semantic Search

**Semantic search** is a data searching technique in a which a search query aims to not only find keywords, but to determine the intent and contextual meaning of the the words a person is using for search.


# Folder Structure

```

├── data
│   ├── embeddings
│   │   ├── testembeddings.npy
│   │   └── trainembeddings.npy
│   ├── test
│   ├── train[:12000]
│   └── training_index.faiss
├── Dockerfile
├── docs
│   └── README.md
├── LICENSE
├── README.md
├── requirements.txt
└── scripts
    ├── app.py
    ├── embed_data.py
    ├── fetch_load_data.py
    ├── load_model.py
    └── search_query.py
```


## Project Goal and Method used:

> Given a query, find the top 5 relevant new article based on the semantic similarity. 
>> Steps:
>>> Data Collection
>>> Model embedding over data
>>> Indexing the data to get faster retrieval.
>>> Pass the query and embed the query using the same model .
>>> Use the nearest neighbor algorithm to get the relevant news articles.
 
>  The dataset used is **AG's corpus of news articles** constructed by assembling titles and description fields of articles from the 4 largest classes (“World”, “Sports”, “Business”, “Sci/Tech”) of AG’s Corpus.

> For embedding the data, I am using **multi-qa-MiniLM-L6-cos-v1** as the model. 

>  For indexing the data, I used FAISS (Faiss is a library for efficient similarity search and clustering of dense vectors).

## Files

- fetch_load_data.py : To download the training and the testing data
- embed.py : To embed the training data and generate the indexing the data.
- load_model.py : To download the model and train in case.
- search_query.py : get the query as input to get indexes of the news paper articles and return that as output.
- app.py : API wrapper around search_query.py to generate docs to test using Fast API.

# Deployment 
- requirements.txt : all the requirements needed to install
- Dockerfile : Using Python 3.7 and ubuntu 20.01 as the base image to run the project locally.
- docker_image.yml : Containerize the model to generate image to use it.
- main.yml : Connect to heroku container registery to push the repo code, build the model image and release it on heroku platform
- Link to access the api : https://semantic-search-application.herokuapp.com/docs

# Output

![Search-query.png](https://github.com/therealadarsh/rbs-ml-e2e-challenge/blob/develop/docs/Search-query.png)

