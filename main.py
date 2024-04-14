from datasets import Dataset
from trainer_module import ModelTrainer
from data_module import DataLoader
from transformers import pipeline

print("Loading the dataset...")
#Loading the dataset.
loader = DataLoader("spam.csv")
#Cleaning the dataset.
print("Cleaning the dataset...")
loader.clean_data()
#Saving the cleaned dataset.
loader.save_data("cleaned_data.csv")

#Loading the dataframe.
df = loader.get_data()

#Converting the dataframe to dataset.
msgs = Dataset.from_pandas(df)
#Splitting the dataset into training and testing.
msgs = msgs.train_test_split(test_size=0.2)

#Models to be trained.
models = ["FacebookAI/roberta-base", "distilbert/distilbert-base-uncased", "albert/albert-base-v2", "google-bert/bert-base-uncased"]
#Model paths to save the trained models.
model_paths = ["roberta-base", "distilbert-base", "albert-base", "bert-base-uncased"]

print("Training the models...")
#Training the models.
#If you have already trained the models, you can skip this step.
for i in range(len(models)):
    model = ModelTrainer(models[i], wandb_path="autonomous-agents", model_path=model_paths[i])
    model.train(msgs)

print("A simple test of the trained models...")

custom_spam_messages = [
    "Congratulations! You've won a free vacation. Reply 'CLAIM' to redeem your prize.",
    "Urgent: Your account has been compromised. Click the link to reset your password and secure your funds.",
    "Get a loan with no credit check! Reply 'LOAN' for instant approval.",
    "Earn $1000/week working from home. Reply 'INFO' for more details."]

#Testing inference on custom spam messages.
for model in models:
    print("-----------------------------------")
    print("Model:", model)
    detector = pipeline("sentiment-analysis", model=model + "/checkpoint-800")
    for text in custom_spam_messages:
       print(detector(text))

