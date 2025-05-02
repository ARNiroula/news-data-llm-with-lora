# news-data-llm-with-lora
Parameter Efficient Finetuning of LLMs with Low-Rank Adaption (LoRA) for News Data

This project is to build a text classification model fine-tuned using Low-Rank Adaptation (LoRA) to categorize news articles into one of four categories: World, Sports, Business, or Sci/Tech.
Dataset used
  - AG News Dataset
Constraints
  - Use the roBERT model as the base model
  - The model should have less than 1M Trainable parameters


Note: we didn't include the checkpoint of the fine-tuned GPT-2 LLM model for sample generation because it's too large. It can be accessed using this link: https://drive.google.com/drive/folders/1gKe8IZ2fhkJ2VGdLkbtRCqOiH64gtDqc?usp=sharing


## To Run the Notebook
### If in local machine, initialize the virtual environment
- Run the following commands
```
python -m venv ./venv

# If in Windows machine
source venv/Script/activate

# If in Mac or Linux Machine
source venv/bin/activate
```

### Install the required packages
```
pip install -r requirements.txt
```

### Check the Logs of all the runs in the Tensorboard
- Run the following command
```
tensorboard --logdir=<tensorboard-log-path>
```
- In this notebook, the `tensorboard` logs are inside <experiment-name>/logs/

## Repo Structure
This repo has 2 types of directories
- experiments: This was used as scratch environment to test parameters and Experiments with different techniques
- {accuracy}_privateScore_files: This directory contains the notebook, checkpoint, output, logs that was done to get the accuracy

### Note => Out best model is in the directory /87628_privateScore_files/best-model/
