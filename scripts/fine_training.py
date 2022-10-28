import os
import numpy as np
from datasets import load_from_disk
from transformers import T5Tokenizer, T5ForConditionalGeneration
from sentence_transformers import SentenceTransformer, InputExample, losses, models, datasets
import torch
import argparse
from tqdm import tqdm

# Parameters for generation
batch_size = 64 #Batch size
num_queries = 2 #Number of queries to generate for every paragraph
max_length_paragraph = 128 #Max length for paragraph
max_length_query = 32   #Max length for output query

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_query_model():
    
    tokenizer = T5Tokenizer.from_pretrained('BeIR/query-gen-msmarco-t5-base-v1')
    model = T5ForConditionalGeneration.from_pretrained('BeIR/query-gen-msmarco-t5-base-v1')
    model.eval()
    
    model.to(device)

    return tokenizer, model

def generate_synthesis_query():

    tokenizer, model = get_query_model()
    dataset = load_from_disk("data/train[:12000]")
    encoded_dataset = dataset.map(lambda examples: {'labels': examples['label']}, batched=True)
    encoded_dataset = encoded_dataset.map(lambda e: tokenizer(e['text'], truncation=True, padding='max_length'), batched=True)
    encoded_dataset.set_format(type='torch', columns=['input_ids' , 'attention_mask'])
    dataloader = torch.utils.data.DataLoader(encoded_dataset, batch_size=batch_size)
    with open('data/generated_queries.tsv', 'w') as fOut:
        for i, batch in enumerate(tqdm(dataloader)):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model.generate(
                **batch,
                max_length=max_length_query,
                do_sample=True,
                top_p=0.95,
                num_return_sequences=num_queries)

            print("\nGenerated Queries:")
            for idx, out in enumerate(outputs):
                
                query = tokenizer.decode(out, skip_special_tokens=True)
                para = dataset[int(idx/num_queries)+ batch_size*i]['text']
                fOut.write("{}\t{}\n".format(query.replace("\t", " ").strip(), para.replace("\t", " ").strip()))
    return 


def fine_tune():
    train_examples = []
    with open('data/generated_queries.tsv') as fIn:
        for line in fIn:
            query, paragraph = line.strip().split('\t', maxsplit=1)
            train_examples.append(InputExample(texts=[query, paragraph]))


    train_dataloader = datasets.NoDuplicatesDataLoader(train_examples, batch_size=64)

    embedder_name = 'sentence-transformers/multi-qa-MiniLM-L6-cos-v1'
    if os.path.isdir('models/semantic-model'):
        embedder_name = 'models/semantic-model'
    word_emb = models.Transformer(embedder_name)
    pooling = models.Pooling(word_emb.get_word_embedding_dimension())
    model = SentenceTransformer(modules=[word_emb, pooling])

    train_loss = losses.MultipleNegativesRankingLoss(model)

    #Tune the model
    num_epochs = 3
    warmup_steps = int(len(train_dataloader) * num_epochs * 0.1)
    model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=num_epochs, warmup_steps=warmup_steps, show_progress_bar=True)
    os.makedirs('models', exist_ok=True)
    model.save('models/semantic-model')


def main():
    parser = argparse.ArgumentParser(description = 'Search Query')
    parser.add_argument('--generate', help='Pass Yes or No to generate query')
    args = parser.parse_args()
    if args.generate.lower() == "yes":
        generate_synthesis_query()
        fine_tune()
    else:
        fine_tune()

if __name__ == "__main__": 
    main()