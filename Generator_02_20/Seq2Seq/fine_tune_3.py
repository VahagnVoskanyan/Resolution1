import argparse
import os
import re
import time
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

# Import modules from your project
from dataset_generator_saves import MGUDatasetCreator
from mgu_data_processor import MGUDataProcessor, create_dataloaders
from mgu_seq2seq_model import Encoder, Decoder, Seq2Seq, train, evaluate, predict, decode_prediction

def get_latest_checkpoint(save_dir, prefix="mgu_seq2seq_finetuned_v", ext=".pt"):
    """
    Scan the save directory for existing fine-tuned checkpoints and return the one with the highest version.
    """
    pattern = re.compile(rf"{re.escape(prefix)}(\d+){re.escape(ext)}")
    versions = []
    for filename in os.listdir(save_dir):
        match = pattern.match(filename)
        if match:
            versions.append(int(match.group(1)))
    if versions:
        latest_version = max(versions)
        latest_checkpoint = os.path.join(save_dir, f"{prefix}{latest_version}{ext}")
        return latest_checkpoint, latest_version
    else:
        return None, 0

def main():
    parser = argparse.ArgumentParser(description="Incremental Fine Tuning for MGU Seq2Seq Model")
    # Data and file parameters (paths are relative to your 'Seq2Seq' folder)
    parser.add_argument('--data-dir', type=str, default='mgu_data', help='Folder to save the generated dataset')
    parser.add_argument('--base-checkpoint', type=str, default='models/mgu_seq2seq_finetuned2.pt',
                        help='Path to the base model checkpoint if no previous fine tuned version exists')
    parser.add_argument('--save-dir', type=str, default='models', help='Folder to save fine tuned models')
    
    # Model architecture and preprocessing hyperparameters
    parser.add_argument('--max-length', type=int, default=200, help='Maximum sequence length for padding')
    parser.add_argument('--embedding-dim', type=int, default=256, help='Embedding dimension')
    parser.add_argument('--hidden-dim', type=int, default=512, help='Hidden dimension')
    parser.add_argument('--n-layers', type=int, default=2, help='Number of layers')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    
    # Training hyperparameters for fine tuning
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--n-epochs', type=int, default=5, help='Number of fine tuning epochs')
    parser.add_argument('--clip', type=float, default=1.0, help='Gradient clipping')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')

    args = parser.parse_args()

    # Ensure that the data and models directories exist
    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)

    # 1. Generate new dataset and save in the 'mgu_data' folder
    print("Generating new dataset...")
    creator = MGUDatasetCreator(output_dir=args.data_dir)
    creator.generate_dataset()  # Creates train.jsonl, val.jsonl, and test.jsonl in args.data_dir

    # 2. Process the new dataset using your data processor
    print("Processing the new dataset...")
    processor = MGUDataProcessor(data_dir=args.data_dir, max_length=args.max_length)
    train_dataset, val_dataset, test_dataset = processor.prepare_dataset()
    train_loader, val_loader, _ = create_dataloaders(
        train_dataset, val_dataset, test_dataset, batch_size=args.batch_size
    )

    # 3. Instantiate the model architecture
    print("Instantiating model...")
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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Seq2Seq(encoder, decoder, device).to(device)

    # 4. Load the latest fine tuned checkpoint from 'models', if available; otherwise, load the base checkpoint
    latest_checkpoint, latest_version = get_latest_checkpoint(args.save_dir)
    if latest_checkpoint:
        print(f"Loading latest fine tuned checkpoint: {latest_checkpoint}")
        model.load_state_dict(torch.load(latest_checkpoint, map_location=device))
        current_version = latest_version + 1
    else:
        print(f"No fine tuned checkpoint found. Loading base checkpoint: {args.base_checkpoint}")
        model.load_state_dict(torch.load(args.base_checkpoint, map_location=device))
        current_version = 1

    # 5. Set up optimizer and loss criterion
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding tokens

    # 6. Fine tuning training loop
    best_valid_loss = float('inf')
    for epoch in range(args.n_epochs):
        print(f'\nEpoch {epoch+1}/{args.n_epochs}')
        start_time = time.time()
        
        train_loss = train(model, train_loader, optimizer, criterion, args.clip, device)
        valid_loss = evaluate(model, val_loader, criterion, device)
        
        end_time = time.time()
        epoch_mins, epoch_secs = divmod(end_time - start_time, 60)
        print(f"Epoch Time: {epoch_mins}m {epoch_secs:.2f}s")
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {valid_loss:.4f}")
        
        # Save the model if validation loss improves
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            save_path = os.path.join(args.save_dir, f'mgu_seq2seq_finetuned_v{current_version}.pt')
            torch.save(model.state_dict(), save_path)
            print(f"Checkpoint saved at {save_path}")

    # 7. Evaluate the fine tuned model on the test set and display sample predictions
    print("\nEvaluating on test set...")
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    test_loss = evaluate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}")

    print("\nTesting model on some examples:")
    # Use up to 5 examples from the test set
    for i in range(min(5, len(test_dataset))):
        sample = test_dataset[i]
        src_tensor = sample['input'].unsqueeze(0).to(device)
        predictions = predict(model, src_tensor, args.max_length,
                              processor.token2idx['<SOS>'], processor.token2idx['<EOS>'])
        print(f"\nExample {i+1}:")
        print(f"Input: {sample['input_string']}")
        print(f"True MGU: {sample['output_string']}")
        print(f"Predicted MGU: {decode_prediction(predictions, processor.idx2token)}")

    print("Fine tuning complete.")

if __name__ == '__main__':
    main()
