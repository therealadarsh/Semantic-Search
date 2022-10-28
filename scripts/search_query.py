from sentence_transformers import SentenceTransformer, util
import datasets
import faiss
import argparse
import os
import time

def search(query):
    embedder_name = 'multi-qa-MiniLM-L6-cos-v1'
    if os.path.isdir('models/semantic-model'):
        embedder_name = 'models/semantic-model'
        
    embedder = SentenceTransformer(embedder_name)
    t=time.time()
    query_vector = embedder.encode([query])
    dataset = datasets.load_from_disk("data/train[:12000]")
    dataset.load_faiss_index('embeddings', 'data/training_index.faiss')
    scores, retrieved_examples = dataset.get_nearest_examples('embeddings', query_vector, k=5)
    print(scores)
    output = {}
    
    for retrieved_example in retrieved_examples['text']:
        print(retrieved_example)
        print("--------------------------------------------------")
    print('totaltime: {}'.format(time.time()-t))
    return retrieved_examples['text']

# def main():
#     parser = argparse.ArgumentParser(description = 'Search Query')
#     parser.add_argument('--query', help='quert name, enter it')
#     args = parser.parse_args()
#     print(args.query)
#     search(args.query)

# if __name__ == "__main__": 
#     main()
