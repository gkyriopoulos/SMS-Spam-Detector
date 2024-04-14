from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
import evaluate
import numpy as np
import wandb
import os

HF_DATASETS_CACHE="/dataset_cache"

class ModelTrainer:
    def __init__(self, model_name, model_path = "my_model", wandb_path = "my_path"):
        self.model_path = model_path
        self.model_name = model_name
        self.wandb_path = wandb_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        self.id2label = {0: "NON-SPAM", 1: "SPAM"}
        self.label2id = {"NON-SPAM": 0, "SPAM": 1}
        

    def preprocess_function(self, examples):
        return self.tokenizer(examples["text"], truncation=True)

    def compute_metrics(self, eval_pred):

        metric1 = evaluate.load("precision")
        metric2 = evaluate.load("recall")
        metric3 = evaluate.load("f1")
        metric4 = evaluate.load("accuracy")

        predictions, labels = eval_pred

        predictions = np.argmax(predictions, axis=1)    
        precision = metric1.compute(predictions=predictions, references=labels, average="micro")["precision"]
        recall = metric2.compute(predictions=predictions, references=labels, average="micro")["recall"]
        f1 = metric3.compute(predictions=predictions, references=labels, average="micro")["f1"]
        accuracy = metric4.compute(predictions=predictions, references=labels)["accuracy"]

        return {"precision": precision,
                "recall": recall,
                "f1": f1,
                "accuracy": accuracy
        }

    def train(self, msgs, learning_rate = 2e-5, per_device_train_batch_size = 16, per_device_eval_batch_size = 16, num_train_epochs = 3, weight_decay = 0.01, eval_steps = 20, save_steps = 800):
        tokenized_tweets = msgs.map(self.preprocess_function, batched=True)

        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=2, id2label=self.id2label, label2id=self.label2id
        )
        
        name = self.model_name.replace('/', '-')
        os.environ["WANDB_PROJECT"]=self.wandb_path

        # save your trained model checkpoint to wandb
        os.environ["WANDB_LOG_MODEL"]="true"

        # turn off watch to log faster
        os.environ["WANDB_WATCH"]="false"

        training_args = TrainingArguments(
            output_dir=self.model_path,
            report_to="wandb",
            learning_rate=learning_rate,
            per_device_train_batch_size = per_device_train_batch_size,
            per_device_eval_batch_size = per_device_eval_batch_size,
            num_train_epochs = num_train_epochs,
            weight_decay = weight_decay,
            evaluation_strategy = "steps",
            save_strategy = "steps",
            eval_steps= eval_steps,
            save_steps = save_steps,
            load_best_model_at_end=True,
            push_to_hub=False,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_tweets["train"],
            eval_dataset=tokenized_tweets["test"],
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            compute_metrics=self.compute_metrics,
        )

        trainer.train()
        wandb.finish()