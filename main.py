from datasets import Dataset
from trainer_module import ModelTrainer
from data_module import DataLoader
from transformers import pipeline

loader = DataLoader("spam.csv")
loader.clean_data()
loader.save_data("cleaned_data.csv")

df = loader.get_data()

msgs = Dataset.from_pandas(df)
msgs = msgs.train_test_split(test_size=0.2)

# classifier_1 = ModelTrainer("FacebookAI/roberta-base", wandb_path="autonomous-agents", model_path="roberta-base")
# classifier_1.train(msgs)

# classifier_2 = ModelTrainer("distilbert/distilbert-base-uncased", wandb_path="autonomous-agents", model_path="distilbert-base")
# classifier_2.train(msgs)

# classifier_3 = ModelTrainer("google/electra-small-discriminator", wandb_path="autonomous-agents", model_path="electra-small-discriminator")
# classifier_3.train(msgs)

# classifier_4 = ModelTrainer("albert/albert-base-v2", wandb_path="autonomous-agents", model_path="albert-base")
# classifier_4.train(msgs)

classifier_5 = ModelTrainer("google-bert/bert-base-uncased", wandb_path="autonomous-agents", model_path="bert-base-uncased")
classifier_5.train(msgs)


# classifier_5 = pipeline("sentiment-analysis", model="electra-/checkpoint-700")

# custom_spam_messages = [
#     "Congratulations! You've won a free vacation. Reply 'CLAIM' to redeem your prize.",
#     "Urgent: Your account has been compromised. Click the link to reset your password and secure your funds.",
#     "Get a loan with no credit check! Reply 'LOAN' for instant approval.",
#     "Earn $1000/week working from home. Reply 'INFO' for more details."]

# for text in custom_spam_messages:
#     print(classifier(text))