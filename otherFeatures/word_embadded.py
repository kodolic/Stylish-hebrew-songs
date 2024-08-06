import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel

# Load the main dataset
file_path = 'new_csv.csv'  # Ensure this file is in the correct directory
df = pd.read_csv(file_path)

# Load pre-trained HeBERT model and tokenizer
tokenizer_hebert = AutoTokenizer.from_pretrained('avichr/heBERT')
model_hebert = AutoModel.from_pretrained('avichr/heBERT')

# Load pre-trained multilingual BERT model and tokenizer
tokenizer_mbert = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')
model_mbert = AutoModel.from_pretrained('bert-base-multilingual-cased')

# Load pre-trained XLM-RoBERTa model and tokenizer
tokenizer_xlm = AutoTokenizer.from_pretrained('xlm-roberta-base')
model_xlm = AutoModel.from_pretrained('xlm-roberta-base')


def get_word_embeddings(sentence, tokenizer, model):
    # Tokenize input
    inputs = tokenizer(sentence, return_tensors='pt', truncation=True, padding=True, max_length=512)

    # Get the hidden states from the model
    with torch.no_grad():
        outputs = model(**inputs)

    # Get the embeddings for each word (excluding special tokens)
    last_hidden_state = outputs[0]  # outputs is a tuple and we need the first element
    word_embeddings = last_hidden_state.squeeze(0)[1:-1].numpy()  # Remove [CLS] and [SEP] tokens

    return word_embeddings


# Add columns for the embeddings of the Hebrew words from each model
df['hebert_embeddings_hebrew'] = df['words'].apply(
    lambda x: get_word_embeddings(x, tokenizer_hebert, model_hebert).tolist())
df['mbert_embeddings_hebrew'] = df['words'].apply(
    lambda x: get_word_embeddings(x, tokenizer_mbert, model_mbert).tolist())
df['xlm_embeddings_hebrew'] = df['words'].apply(lambda x: get_word_embeddings(x, tokenizer_xlm, model_xlm).tolist())

# Save the updated dataset with the new columns
updated_csv_path = 'updated_with_embeddings_from_all_models.csv'
df.to_csv(updated_csv_path, index=False)

# Print the path to the updated dataset
print(f"Updated dataset saved to: {updated_csv_path}")
