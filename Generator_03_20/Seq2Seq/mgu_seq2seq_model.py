import os
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from mgu_data_processor import MGUDataProcessor, create_dataloaders

class Encoder(nn.Module):
    def __init__(self, input_size, embedding_dim, hidden_dim, n_layers=1, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        self.embedding = nn.Embedding(input_size, embedding_dim)
        self.rnn = nn.GRU(
            embedding_dim, 
            hidden_dim, 
            n_layers, 
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src):
        # src: [batch_size, seq_len]
        
        embedded = self.dropout(self.embedding(src))
        # embedded: [batch_size, seq_len, embedding_dim]
        
        outputs, hidden = self.rnn(embedded)
        # outputs: [batch_size, seq_len, hidden_dim]
        # hidden: [n_layers, batch_size, hidden_dim]
        
        return outputs, hidden


class Decoder(nn.Module):
    def __init__(self, output_size, embedding_dim, hidden_dim, n_layers=1, dropout=0.1):
        super().__init__()
        self.output_size = output_size
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        self.embedding = nn.Embedding(output_size, embedding_dim)
        self.rnn = nn.GRU(
            embedding_dim, 
            hidden_dim,
            n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0
        )
        self.fc_out = nn.Linear(hidden_dim, output_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden):
        # input: [batch_size, 1]
        # hidden: [n_layers, batch_size, hidden_dim]
        
        embedded = self.dropout(self.embedding(input))
        # embedded: [batch_size, 1, embedding_dim]
        
        output, hidden = self.rnn(embedded, hidden)
        # output: [batch_size, 1, hidden_dim]
        # hidden: [n_layers, batch_size, hidden_dim]
        
        prediction = self.fc_out(output.squeeze(1))
        # prediction: [batch_size, output_size]
        
        return prediction, hidden


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src: [batch_size, src_len]
        # trg: [batch_size, trg_len]
        
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.output_size
        
        # Tensor to store decoder outputs
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        
        # Get encoder outputs and hidden state
        encoder_outputs, hidden = self.encoder(src)
        
        # First input to the decoder is the <SOS> token
        input = trg[:, 0].unsqueeze(1)
        
        for t in range(1, trg_len):
            # Pass input, hidden state through decoder
            output, hidden = self.decoder(input, hidden)
            
            # Place predictions in the outputs tensor
            outputs[:, t, :] = output
            
            # Teacher forcing: use actual next token as next input
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            
            # Get the highest predicted token
            top1 = output.argmax(1)
            
            # If teacher forcing, use actual token, otherwise use predicted token
            input = trg[:, t].unsqueeze(1) if teacher_force else top1.unsqueeze(1)
        
        return outputs


def train(model, iterator, optimizer, criterion, clip, device):
    model.train()
    epoch_loss = 0
    
    for i, batch in enumerate(iterator):
        src = batch['input'].to(device)
        trg = batch['output'].to(device)
        
        optimizer.zero_grad()
        
        output = model(src, trg)
        
        # trg: [batch_size, trg_len]
        # output: [batch_size, trg_len, output_size]
        
        # Reshape output and target for loss calculation
        output_dim = output.shape[-1]
        output = output[:, 1:].contiguous().view(-1, output_dim)
        trg = trg[:, 1:].contiguous().view(-1)
        
        # Calculate loss
        loss = criterion(output, trg)
        
        # Backpropagate loss
        loss.backward()
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        # Update parameters
        optimizer.step()
        
        epoch_loss += loss.item()
        
        if (i + 1) % 50 == 0:
            print(f'Batch {i+1}/{len(iterator)}, Loss: {loss.item():.4f}')
    
    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion, device):
    model.eval()
    epoch_loss = 0
    
    with torch.no_grad():
        for batch in iterator:
            src = batch['input'].to(device)
            trg = batch['output'].to(device)
            
            output = model(src, trg, 0) # turn off teacher forcing
            
            # Reshape output and target for loss calculation
            output_dim = output.shape[-1]
            output = output[:, 1:].contiguous().view(-1, output_dim)
            trg = trg[:, 1:].contiguous().view(-1)
            
            # Calculate loss
            loss = criterion(output, trg)
            
            epoch_loss += loss.item()
    
    return epoch_loss / len(iterator)


def predict(device, model, src_tensor, max_len, sos_idx, eos_idx, ):
    model.eval()
    
    with torch.no_grad():
        # Get encoder outputs and hidden state
        encoder_outputs, hidden = model.encoder(src_tensor)
        
        # First input to the decoder is the <SOS> token
        input = torch.tensor([[sos_idx]]).to(device)
        
        # List to store predicted tokens
        predictions = [sos_idx]
        
        for i in range(max_len):
            # Pass input, hidden state through decoder
            output, hidden = model.decoder(input, hidden)
            
            # Get the highest predicted token
            pred_token = output.argmax(1).item()
            
            # Add token to predictions
            predictions.append(pred_token)
            
            # If EOS token, stop generating
            if pred_token == eos_idx:
                break
            
            # Use predicted token as next input
            input = torch.tensor([[pred_token]]).to(device)
    
    return predictions


def decode_prediction(predictions, idx2token):
    """Convert predicted indices to tokens and join them into a string."""
    tokens = [idx2token[idx] for idx in predictions]
    
    # Remove special tokens
    if '<SOS>' in tokens:
        tokens.remove('<SOS>')
    if '<EOS>' in tokens:
        tokens = tokens[:tokens.index('<EOS>')]
    
    # Join tokens with space
    return ' '.join(tokens)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Train MGU Seq2Seq model')
    parser.add_argument('--data-dir', type=str, default='mgu_data', help='Directory containing the processed data files')
    parser.add_argument('--max-length', type=int, default=200, help='Maximum sequence length')
    parser.add_argument('--embedding-dim', type=int, default=256, help='Embedding dimension')
    parser.add_argument('--hidden-dim', type=int, default=512, help='Hidden dimension')
    parser.add_argument('--n-layers', type=int, default=2, help='Number of layers')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--n-epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--clip', type=float, default=1.0, help='Gradient clipping')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--save-dir', type=str, default='models', help='Directory to save model')
    parser.add_argument('--curriculum', action='store_true', help='Use curriculum learning')
    
    args = parser.parse_args()
    
    # Set device
    global device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    if torch.cuda.is_available():
        print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️ Using CPU (GPU not available or CUDA not enabled)")
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Process data
    processor = MGUDataProcessor(
        data_dir=args.data_dir,
        max_length=args.max_length
    )
    
    if args.curriculum:
        print("Using curriculum learning...")
        level_datasets = processor.prepare_curriculum_dataset()
        
        # Create model
        input_size = len(processor.token2idx)
        output_size = len(processor.token2idx)
        
        encoder = Encoder(
            input_size=input_size,
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            n_layers=args.n_layers,
            dropout=args.dropout
        )
        
        decoder = Decoder(
            output_size=output_size,
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            n_layers=args.n_layers,
            dropout=args.dropout
        )
        
        model = Seq2Seq(encoder, decoder, device).to(device)
        
        # Initialize optimizer and criterion
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding tokens
        
        # Train on each level
        for i, dataset in enumerate(level_datasets):
            print(f"\nTraining on curriculum level {i+1}")
            
            # Create dataloader
            dataloader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=True
            )
            
            # Train for a few epochs
            epochs = 3 if i < len(level_datasets) - 1 else args.n_epochs
            for epoch in range(epochs):
                print(f'Epoch {epoch+1}/{epochs}')
                start_time = time.time()
                
                train_loss = train(model, dataloader, optimizer, criterion, args.clip)
                
                end_time = time.time()
                epoch_mins, epoch_secs = divmod(end_time - start_time, 60)
                
                print(f'Epoch {epoch+1}/{epochs} | Time: {epoch_mins}m {epoch_secs:.2f}s')
                print(f'Train Loss: {train_loss:.4f}')
            
            # Save model for this level
            torch.save(model.state_dict(), os.path.join(args.save_dir, f'mgu_seq2seq_level_{i+1}.pt'))
    else:
        print("Processing standard train/val/test datasets...")
        train_dataset, val_dataset, test_dataset = processor.prepare_dataset()
        
        # Create dataloaders
        train_loader, val_loader, test_loader = create_dataloaders(
            train_dataset, val_dataset, test_dataset,
            batch_size=args.batch_size
        )
        
        # Create model
        input_size = len(processor.token2idx)
        output_size = len(processor.token2idx)
        
        encoder = Encoder(
            input_size=input_size,
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            n_layers=args.n_layers,
            dropout=args.dropout
        )
        
        decoder = Decoder(
            output_size=output_size,
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            n_layers=args.n_layers,
            dropout=args.dropout
        )
        
        model = Seq2Seq(encoder, decoder, device).to(device)
        
        # Print model summary
        print(f"Model Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
        
        # Initialize optimizer and criterion
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding tokens
        
        # Train model
        best_valid_loss = float('inf')
        
        for epoch in range(args.n_epochs):
            print(f'Epoch {epoch+1}/{args.n_epochs}')
            start_time = time.time()
            
            train_loss = train(model, train_loader, optimizer, criterion, args.clip)
            valid_loss = evaluate(model, val_loader, criterion)
            
            end_time = time.time()
            epoch_mins, epoch_secs = divmod(end_time - start_time, 60)
            
            # Save model if validation loss improved
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(model.state_dict(), os.path.join(args.save_dir, 'mgu_seq2seq_best.pt'))
            
            print(f'Epoch {epoch+1}/{args.n_epochs} | Time: {epoch_mins}m {epoch_secs:.2f}s')
            print(f'Train Loss: {train_loss:.4f} | Val. Loss: {valid_loss:.4f}')
        
        # Load best model for evaluation
        model.load_state_dict(torch.load(os.path.join(args.save_dir, 'mgu_seq2seq_best.pt')))
        
        # Evaluate on test set
        test_loss = evaluate(model, test_loader, criterion)
        print(f'Test Loss: {test_loss:.4f}')
        
        # Save final model
        torch.save(model.state_dict(), os.path.join(args.save_dir, 'mgu_seq2seq_final.pt'))
        
        # Save tokenizer
        with open(os.path.join(args.save_dir, 'tokenizer.json'), 'w') as f:
            json.dump({
                'token2idx': processor.token2idx,
                'idx2token': {str(k): v for k, v in processor.idx2token.items()}
            }, f)
        
        # Test on some examples
        print("\nTesting model on some examples:")
        for i in range(5):
            sample = test_dataset[i]
            src_tensor = sample['input'].unsqueeze(0).to(device)
            predictions = predict(model, src_tensor, args.max_length, 
                                processor.token2idx['<SOS>'], processor.token2idx['<EOS>'])
            
            print(f"\nExample {i+1}:")
            print(f"Input: {sample['input_string']}")
            print(f"True MGU: {sample['output_string']}")
            print(f"Predicted MGU: {decode_prediction(predictions, processor.idx2token)}")
    
    print("Done!")


if __name__ == "__main__":
    main()



## --- ##
# Add these methods to your existing MGUSeq2Seq class in mgu_seq2seq_model.py

def generate(self, input_tensor, max_length=200, sos_token_id=1, eos_token_id=2):
    """
    Generate output sequence using greedy decoding.
    
    Args:
        input_tensor: Input tensor of shape [batch_size, input_seq_len]
        max_length: Maximum output sequence length
        sos_token_id: Start of sequence token ID
        eos_token_id: End of sequence token ID
        
    Returns:
        Output tensor of shape [batch_size, output_seq_len]
    """
    batch_size = input_tensor.size(0)
    
    # Initialize encoder hidden state
    encoder_hidden = self.encoder.init_hidden(batch_size, self.device)
    
    # Encoder forward pass
    encoder_outputs, encoder_hidden = self.encoder(input_tensor, encoder_hidden)
    
    # Initialize decoder input with SOS token
    decoder_input = torch.tensor([[sos_token_id] * batch_size], device=self.device).transpose(0, 1)
    
    # Use encoder's final hidden state to initialize decoder hidden state
    decoder_hidden = encoder_hidden
    
    # Initialize output tensor
    outputs = torch.zeros(batch_size, max_length, device=self.device, dtype=torch.long)
    outputs[:, 0] = sos_token_id
    
    # Generate tokens sequentially
    for t in range(1, max_length):
        # Decoder forward pass
        decoder_output, decoder_hidden, _ = self.decoder(
            decoder_input, decoder_hidden, encoder_outputs
        )
        
        # Get most likely token
        _, topi = decoder_output.topk(1)
        topi = topi.squeeze(-1).detach()  # detach from history as input
        
        # Add predicted token to outputs
        outputs[:, t] = topi
        
        # If all sequences in batch have ended, stop generation
        if (topi == eos_token_id).all():
            break
        
        # Next input is current prediction
        decoder_input = topi.unsqueeze(1)
    
    return outputs


def load_model(model_path, token2idx, device='cpu'):
    """
    Load a trained MGU model from a checkpoint.
    
    Args:
        model_path: Path to the model checkpoint
        token2idx: Dictionary mapping tokens to indices
        device: Device to load the model on ('cpu' or 'cuda')
        
    Returns:
        Loaded model
    """
    # Get model hyperparameters from checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Create model with same hyperparameters
    model = Seq2Seq(
        input_size=len(token2idx),
        hidden_size=checkpoint['hidden_size'],
        output_size=len(token2idx),
        n_layers=checkpoint['n_layers'],
        dropout=checkpoint['dropout'],
        device=device
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Move model to device
    model = model.to(device)
    
    return model