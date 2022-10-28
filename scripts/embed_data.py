import numpy as np
import datasets
import argparse
from sentence_transformers import SentenceTransformer, util


def embed_data(datapath, split):
    dataset = datasets.load_from_disk(datapath)
    embedder_name = 'multi-qa-MiniLM-L6-cos-v1'
    embedder = SentenceTransformer(embedder_name)
    ds_with_embeddings = dataset.map(lambda example: {'embeddings': embedder.encode(example['text'], convert_to_tensor=True).numpy()})
    ds_with_embeddings.save_to_disk(datapath)
    ds_with_embeddings.add_faiss_index(column='embeddings')
    ds_with_embeddings.save_faiss_index('embeddings', 'data/training_index.faiss')
    
    return

def main():
    parser = argparse.ArgumentParser(description = 'Embed dataset')
    parser.add_argument('--dataset', help='dataset name, enter it')
    parser.add_argument('--split', help='split of dataset')
    args = parser.parse_args()
    embed_data(args.dataset,args.split,)

if __name__ == "__main__": 
    main()
