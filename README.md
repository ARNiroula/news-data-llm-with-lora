# news-data-llm-with-lora
Parameter Efficient Finetuning of LLMs with Low-Rank Adaption (LoRA) for News Data

This project is to build a text classification model fine-tuned using Low-Rank Adaptation (LoRA) to categorize news articles into one of four categories: World, Sports, Business, or Sci/Tech.
Dataset used
  - AG News Dataset
Constraints
  - Use the roBERT model as the base model
  - The model should have less than 1M Trainable parameters


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

### Check the Logs of all the runs in the Tensorboard
- Run the following command
```
tensorboard --logdir=runs
```
