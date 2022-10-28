from datasets import load_dataset
import numpy as np
import os

def load_data(data_dir = "",database_name = "", split = "train"):
    if os.path.exists(data_dir + split):
        print("newspaper data already existed ...")
        return
    print("Downloading newspaper data...")
    dataset = load_dataset(database_name, split = split)
    dataset.save_to_disk(data_dir + split)
    return

def get_news_data():
    
    data_base = 'ag_news'
    data_dir = "data/"

    load_data(data_dir, database_name = data_base, split = "train[:12000]")
    #load_data(data_dir, database_name = data_base, split = "test")
    return

def main():
    news_texts = get_news_data()

if __name__ == "__main__":
    main()