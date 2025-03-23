import os
import time
import torch
import torch.optim as optim
import argparse

from dataset_generator_saves import MGUDatasetCreator
from mgu_data_processor import MGUDataProcessor, create_dataloaders
from mgu_seq2seq_model import (
    Encoder,
    Decoder,
    Seq2Seq,
    train,
    evaluate,
    predict,
    decode_prediction
)

def main():
    parser = argparse.ArgumentParser(description="Fine tune and test MGU Seq2Seq model")
    parser.add_argument("--train-size", type=int, default=8000, help="Training set size")
    parser.add_argument("--val-size", type=int, default=1000, help="Validation set size")
    parser.add_argument("--test-size", type=int, default=1000, help="Test set size")
    parser.add_argument("--data-dir", type=str, default="new_mgu_data", help="Directory for new dataset")
    parser.add_argument("--save-dir", type=str, default="models", help="Directory to save models")
    parser.add_argument("--lr", type=float, default=0.0005, help="Learning rate for fine tuning")
    parser.add_argument("--n-epochs", type=int, default=5, help="Number of epochs for fine tuning")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--max-length", type=int, default=200, help="Maximum sequence length")
    args = parser.parse_args()
    
    # Step 1: Generate new dataset
    os.makedirs(args.data_dir, exist_ok=True)
    print("Generating new dataset...")
    creator = MGUDatasetCreator(output_dir=args.data_dir, random_seed=42)
    creator.generate_dataset(train_size=args.train_size, 
                             val_size=args.val_size, 
                             test_size=args.test_size)
    
    # Step 2: Preprocess the new dataset
    print("Preprocessing new dataset...")
    processor = MGUDataProcessor(data_dir=args.data_dir, max_length=args.max_length, vocab_size=5000)
    train_dataset, val_dataset, test_dataset = processor.prepare_dataset()
    train_loader, val_loader, test_loader = create_dataloaders(
        train_dataset, val_dataset, test_dataset, batch_size=args.batch_size
    )
    
    # Step 3: Load the existing model for fine tuning
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    existing_model_path = os.path.join(args.save_dir, "mgu_seq2seq_best.pt")
    if not os.path.exists(existing_model_path):
        print(f"Error: Existing model file not found at {existing_model_path}")
        return

    input_size = len(processor.token2idx)
    output_size = len(processor.token2idx)
    embedding_dim = 256
    hidden_dim = 512
    n_layers = 2
    dropout = 0.1

    encoder = Encoder(
        input_size=input_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        dropout=dropout
    )
    decoder = Decoder(
        output_size=output_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        dropout=dropout
    )
    model = Seq2Seq(encoder, decoder, device).to(device)

    print(f"Loading existing model from {existing_model_path} ...")
    model.load_state_dict(torch.load(existing_model_path, map_location=device))
    
    # Step 4: Fine tune the model
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0)  # assuming 0 is <PAD>
    clip = 1.0
    best_valid_loss = float('inf')
    
    for epoch in range(args.n_epochs):
        print(f"\nEpoch {epoch+1}/{args.n_epochs}")
        start_time = time.time()
        
        train_loss = train(model, train_loader, optimizer, criterion, clip)
        valid_loss = evaluate(model, val_loader, criterion)
        
        end_time = time.time()
        epoch_mins, epoch_secs = divmod(end_time - start_time, 60)
        
        print(f"Epoch Time: {epoch_mins}m {epoch_secs:.2f}s")
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {valid_loss:.4f}")
        
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            upgraded_model_path = os.path.join(args.save_dir, "mgu_seq2seq_upgraded.pt")
            torch.save(model.state_dict(), upgraded_model_path)
            print(f"New best model saved at {upgraded_model_path}")
    
    # Step 5: Evaluate on test set
    print("\nEvaluating on test set...")
    test_loss = evaluate(model, test_loader, criterion)
    print(f"Test Loss: {test_loss:.4f}")
    
    # Step 6: Test on some examples
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
    
    print("Fine tuning and testing complete!")

if __name__ == "__main__":
    main()
