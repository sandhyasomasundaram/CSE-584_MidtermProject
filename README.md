﻿# CSE-584_MidtermProject

# From BERT to Custom Embeddings: A Comparative Study on LLM Output Classification

### Team Members:
Lakshmi Chandrika Yarlagadda
Sandhya Somasundaram

## To run the repo

## Data Generation

Run generate_xi.py script to generate the truncated sentences from Open_Subtitles corpus. This generates a truncated_data.csv in the data folder. 
Run generate_xj.py script to generate sentence completions using various LLMs. Update groq and OpenAPI keys in the file before running. Change the models name for various LLMs you wish to be generated.
Ensure the data is split to ensure a balanced labels in the dataset. Such a dataset is saved in the data folder and is named CSE_584_Final_Dataset.csv

## Run Models
The sentence transformer is combined with a neural network to create a classifier. This can be found in the Experiments&Final
