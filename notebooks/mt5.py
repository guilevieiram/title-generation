# %%
# !git clone https://github.com/guilevieiram/title-generation.git
# !pip install datasets
# !pip install git+https://github.com/guilevieiram/title-generation.git
# !pip install transformers[torch]
# !pip install evaluate
# !pip install trl
# !pip install peft

# %%
from datasets import load_dataset

import pandas as pd
import numpy as np

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
import evaluate

from peft import LoraConfig, get_peft_model

import nltk
from nltk.tokenize import sent_tokenize

nltk.download("punkt")
nltk.download('stopwords')

# %%
device = 'cuda' if torch.cuda.is_available() else 'cpu'
finetune_att=True
finetune_lin=True
crop=0.3
batch_size = 1
num_train_epochs = 10
max_input_length = 1024
max_target_length = 128
r=32
model_checkpoint = "google/mt5-large"
model_name = f"{model_checkpoint}-finetune"
out_dir = f"/Data/{model_name}"
rouge_score = evaluate.load("rouge")

# %%
dataset = load_dataset('csv', data_files={'train': '../data/train.csv', 'validation': '../data/validation.csv'})
train_sample_size = int(crop * len(dataset['train']))
validation_sample_size = int(crop * len(dataset['validation']))
dataset['train'] = dataset['train'].shuffle().select(range(train_sample_size))
dataset['validation'] = dataset['validation'].shuffle().select(range(validation_sample_size))


# %%
def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


# %%
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
print_trainable_parameters(model)
model.to(device)

# %%
def preprocess_function(examples):
    text = [f'summarize: {t}' for t in examples['text']]
    model_inputs = tokenizer(
        text,
        max_length=max_input_length,
        truncation=True,
    )
    labels = tokenizer(
        examples["titles"], max_length=max_target_length, truncation=True
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs
tokenized_datasets = dataset.map(preprocess_function, batched=True)

# %%
att_target_modules = ["q_proj", "k_proj", "v_proj", "out_proj"]
import re
pattern = r'\((\w+)\): Linear'
linear_layers = re.findall(pattern, str(model.modules))
lin_target_modules = list(set(linear_layers))

target_modules = []
if finetune_att: target_modules.extend(att_target_modules)
if finetune_lin: target_modules.extend(lin_target_modules)

lora_config = LoraConfig(
    r=r,
    target_modules = target_modules,
    lora_alpha=8,
    lora_dropout=0.05,
    bias="none",
    task_type="SEQ_2_SEQ_LM"
)

model = get_peft_model(model, lora_config)
print_trainable_parameters(model)

# %%
model_name = model_checkpoint.split("/")[-1]


# %%
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    decoded_preds = ["\n".join(sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(sent_tokenize(label.strip())) for label in decoded_labels]
    result = rouge_score.compute(
        predictions=decoded_preds, references=decoded_labels, use_stemmer=True
    )
    return {k: round(v, 4) for k, v in result.items()}

args = Seq2SeqTrainingArguments(
    output_dir=out_dir,
    evaluation_strategy="epoch",
    learning_rate=5.6e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=num_train_epochs,
    predict_with_generate=True,
)
trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# %%
trainer.evaluate()

# %%
trainer.train()

# %%
def predict(text, model):
    text=f"summarize: {text} \n\n "
    tokens = tokenizer(text, return_tensors='pt').to(device)
    output_tokens=model.generate(**tokens, max_new_tokens=max_target_length)
    return tokenizer.decode(output_tokens[0], skip_special_tokens=True)

def print_summary(idx, model):
    review = dataset["validation"][idx]["text"]
    title = dataset["validation"][idx]["titles"]
    text = dataset["validation"][idx]['text']
    summary = predict(text, model)
    print(f"'>>> Text: {review}'")
    print(f"\n'>>> Title: {title}'")
    print(f"\n'>>> predicted: {summary}'")

# %%
print_summary(1, model)

# %%
trainer.save_model(out_dir)

# %% [markdown]
# # making prediction

# %%
model = AutoModelForSeq2SeqLM.from_pretrained(out_dir)
tokenizer = AutoTokenizer.from_pretrained(out_dir)
model.to(device)

# %%
csv_file_path = '../data/test_text.csv'
dataset = load_dataset('csv', data_files={"data":csv_file_path})

def preprocess(examples):
    return tokenizer(
        [f"summarize: {t}" for t in examples["text"]],
        truncation=True,
        padding='max_length',
        max_length=max_input_length
      )

tokenized_datasets = dataset.map(preprocess, batched=True)

# %%
batch_size = 8 
data_loader = DataLoader(tokenized_datasets['data'], batch_size=batch_size)

preds = []
model.eval()
with torch.no_grad():
    for bidx, batch in enumerate(data_loader):
        print(f"batch {bidx+1}/{len(data_loader)}")
        input_ids = torch.stack(batch['input_ids']).T.to(device)
        attention_mask = torch.stack(batch['attention_mask']).T.to(device)

        output_tokens = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=max_target_length)

        for idx, tokens in enumerate(output_tokens):
            out = tokenizer.decode(tokens, skip_special_tokens=True)
            preds.append({'ID': batch['ID'][idx].item(), "titles": out})

# %%
test_df=pd.DataFrame(preds)
test_df.to_csv(f"./submission-{model_name}.csv", columns=['ID', 'titles'], index=False)
