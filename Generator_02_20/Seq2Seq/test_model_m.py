import os
import json
import torch
from mgu_seq2seq_model import Encoder, Decoder, Seq2Seq, predict, decode_prediction
from mgu_data_processor import MGUDataProcessor

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define parameters
data_dir = "mgu_data"           # Your data directory
max_length = 200                # Maximum sequence length
model_file = "models/mgu_seq2seq_finetuned_v1.pt"  # Path to your trained model checkpoint
embedding_dim = 256
hidden_dim = 512
n_layers = 2
dropout = 0.1

# Initialize the data processor
processor = MGUDataProcessor(data_dir=data_dir, max_length=max_length)
# Load the vocabulary from a saved tokenizer if available
tokenizer_path = os.path.join("models", "tokenizer.json")
if os.path.exists(tokenizer_path):
    with open(tokenizer_path, "r") as f:
         tokenizer_data = json.load(f)
    if "token2idx" in tokenizer_data:
         processor.token2idx = tokenizer_data["token2idx"]
         processor.idx2token = {int(k): v for k, v in tokenizer_data["idx2token"].items()}
    else:
         processor.token2idx = tokenizer_data
         processor.idx2token = {int(k): v for k, v in tokenizer_data.items()}
    processor.vocab_built = True
else:
    raise ValueError("Tokenizer file not found. Please build your vocabulary first.")

# Manually define an input example
input_example = "Clause1: Q(x) ∨ P(x)\nClause2: R(a) ∨ ¬P(a)"
print("Input Example:")
print(input_example)

# Tokenize and encode the input using the processor's methods
tokens = processor._tokenize(input_example)
input_indices = processor._encode(tokens)

# Pad the sequence to max_length using the <PAD> token (assumed index 0)
if len(input_indices) < max_length:
    pad_token = processor.token2idx.get("<PAD>", 0)
    input_indices = input_indices + [pad_token] * (max_length - len(input_indices))
else:
    input_indices = input_indices[:max_length]

# Convert to tensor and add batch dimension
src_tensor = torch.tensor(input_indices, dtype=torch.long).unsqueeze(0).to(device)

# Instantiate the model architecture
input_size = len(processor.token2idx)  # Both input and output vocab sizes are the same
encoder = Encoder(input_size, embedding_dim, hidden_dim, n_layers, dropout)
decoder = Decoder(input_size, embedding_dim, hidden_dim, n_layers, dropout)
model = Seq2Seq(encoder, decoder, device).to(device)

# Load the trained model weights
print(f"Loading model weights from {model_file}")
model.load_state_dict(torch.load(model_file, map_location=device))
model.eval()

# Define special token indices (if not already defined, use defaults: <SOS>=1, <EOS>=2)
sos_idx = processor.token2idx.get("<SOS>", 1)
eos_idx = processor.token2idx.get("<EOS>", 2)

# Use the predict function to generate output sequence
predicted_indices = predict(device, model, src_tensor, max_length, sos_idx, eos_idx)
predicted_output = decode_prediction(predicted_indices, processor.idx2token)

# Print the predicted output
print("\nPredicted MGU:")
print(predicted_output)
