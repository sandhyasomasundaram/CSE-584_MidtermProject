# CSE-584_MidtermProject

# From BERT to Custom Embeddings: A Comparative Study on LLM Output Classification

### Team Members:
Lakshmi Chandrika Yarlagadda \
Sandhya Somasundaram

## To run the repo

## Data Generation

Run generate_xi.py script to generate the truncated sentences from Open_Subtitles corpus. This generates a truncated_data.csv in the data folder. 
Run generate_xj.py script to generate sentence completions using various LLMs. Update groq and OpenAPI keys in the file before running. Change the models name for various LLMs you wish to be generated.
Ensure the data is split to ensure a balanced labels in the dataset. Such a dataset is saved in the data folder and is named CSE_584_Final_Dataset.csv

## Run Models
The sentence transformer is combined with a neural network to create a classifier. This can be found in the models/Sentence_transformer_Stella.ipynb. 
Run this on google colab, it requires GPUs. Run it cell by cell to reproduce the results. There are also various experiments conducted on hyperparameter tuning which can be found in the notebook.

The BERT model was also fine tuned to classify the LLMs. This can be found in models/Bert_Model.ipynb. This has to be run same as above in google colab or in a system with GPU. Ensure that the data file is saved in the same location as that of the notebooks. The experiments run on Bert model is stored in two different notebooks - model/Experiment_1_Bert_Model_CSE_584.ipynb and model/Experiment_2_Bert_Model_CSE_584.ipynb. Follow the above steps to run these notebooks on Google Colab. 
