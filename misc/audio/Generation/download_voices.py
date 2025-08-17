import os
from datasets import load_dataset

save_directory = "./data/voices-jsalt"
os.makedirs(save_directory, exist_ok=True)
dataset = load_dataset("sdialog/voices-jsalt")
dataset.save_to_disk(save_directory)
