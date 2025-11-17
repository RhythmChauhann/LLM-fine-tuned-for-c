import os
import math
import json
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)


MODEL_NAME = "Salesforce/codet5-base"
DATA_PATH = "final.jsonl"
SAVE_DIR = "./codet5_chunked/"
CHUNK_SIZE = 500
NUM_EPOCHS = 2
BATCH_SIZE = 4
LEARNING_RATE = 5e-5
USE_FP16 = True 


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


checkpoints = sorted(
    [d for d in os.listdir(SAVE_DIR) if d.startswith("chunk_")],
    key=lambda x: int(x.split("_")[1]) if "_" in x else 0
)
last_checkpoint = checkpoints[-1] if checkpoints else None

if last_checkpoint:
    print(f"Resuming from checkpoint: {last_checkpoint}")
    model = AutoModelForSeq2SeqLM.from_pretrained(os.path.join(SAVE_DIR, last_checkpoint))
    start_chunk = int(last_checkpoint.split("_")[1]) + 1
else:
    print("Starting fresh training...")
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    start_chunk = 0

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

import re

with open(DATA_PATH, "r", encoding="utf-8") as f:
    content = f.read().strip()

    if content.startswith("["):
        print("Detected JSON array format")
        lines = json.loads(content)

    else:
        print(" Detected JSONL or mixed format")

        
        json_objects = re.findall(r'\{.*?\}(?=\s*\{|\s*$)', content, flags=re.DOTALL)

        lines = []
        for obj in json_objects:
            try:
                lines.append(json.loads(obj))
            except json.JSONDecodeError as e:
                print(f"Skipping malformed JSON: {e}")


total_chunks = math.ceil(len(lines) / CHUNK_SIZE)
print(f"Total samples: {len(lines)} | Chunks: {total_chunks}")


for chunk_idx in range(start_chunk, total_chunks):
    start = chunk_idx * CHUNK_SIZE
    end = min((chunk_idx + 1) * CHUNK_SIZE, len(lines))
    chunk_data = lines[start:end]

    print(f"\nTraining on chunk {chunk_idx+1}/{total_chunks} ({start}-{end})")

   
    dataset = Dataset.from_list(chunk_data)

  
    def preprocess(example):
        input_enc = tokenizer(example["input"], padding="max_length", truncation=True, max_length=512)
        target_enc = tokenizer(example["target"], padding="max_length", truncation=True, max_length=128)
        input_enc["labels"] = target_enc["input_ids"]
        return input_enc

    tokenized_dataset = dataset.map(preprocess, remove_columns=dataset.column_names)

   
    split = tokenized_dataset.train_test_split(test_size=0.1)

    training_args = Seq2SeqTrainingArguments(
        output_dir=os.path.join(SAVE_DIR, f"chunk_{chunk_idx}"),
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        logging_dir="./logs",
        logging_steps=50,
        eval_strategy="epoch",
        save_total_limit=1,
        fp16=USE_FP16,
        do_train=True,
        do_eval=True,
        report_to="none"
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(os.path.join(SAVE_DIR, f"chunk_{chunk_idx}"))
    tokenizer.save_pretrained(os.path.join(SAVE_DIR, f"chunk_{chunk_idx}"))

    print(f"Chunk {chunk_idx} training done. Saved to chunk_{chunk_idx}.\n")

print("\nAll chunks completed successfully!")
