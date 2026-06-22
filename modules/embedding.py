import torch
import numpy as np
from transformers import DistilBertTokenizer, DistilBertModel

MODEL_NAME = 'cahya/distilbert-base-indonesian'
MAX_LENGTH = 128
BATCH_SIZE = 32

_tokenizer = None
_model = None
_device = None

def get_model():
    global _tokenizer, _model, _device
    if _model is None:
        _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        _tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
        _model = DistilBertModel.from_pretrained(MODEL_NAME)
        _model = _model.to(_device)
        _model.eval()
    return _tokenizer, _model, _device

def extract_features(text_list: list, progress_callback=None) -> np.ndarray:
    tokenizer, model, device = get_model()
    all_embeddings = []
    total_batches = (len(text_list) + BATCH_SIZE - 1) // BATCH_SIZE

    for i, start in enumerate(range(0, len(text_list), BATCH_SIZE)):
        batch_texts = text_list[start: start + BATCH_SIZE]

        encoded = tokenizer(
            batch_texts,
            padding='max_length',
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors='pt'
        )

        input_ids = encoded['input_ids'].to(device)
        attention_mask = encoded['attention_mask'].to(device)

        with torch.no_grad():
            output = model(input_ids=input_ids, attention_mask=attention_mask)

        all_embeddings.append(output.last_hidden_state.cpu().numpy())

        if progress_callback:
            progress_callback(i + 1, total_batches, f"Batch {i+1}/{total_batches} selesai diproses...")

    return np.concatenate(all_embeddings, axis=0)