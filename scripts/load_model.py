# from sentence_transformers import SentenceTransformer, util

# def get_embedder(model_name = ""):
#     embedder_name = 'multi-qa-MiniLM-L6-cos-v1'
#     embedder = SentenceTransformer(embedder_name)
#     return embedder

# def main():
#     parser = argparse.ArgumentParser(description = 'Embed dataset')
#     parser.add_argument('--model_name', help='model name, enter it')
#     args = parser.parse_args()
#     load_embedder(args.model_name)

# if __name__ == "__main__": 
#     main()