import argparse
import os
import torch
from torch.utils.data import DataLoader

# Import modules from your project
from mgu_data_processor import MGUDataProcessor, MGUDataset
from mgu_seq2seq_model import Encoder, Decoder, Seq2Seq, evaluate, predict, decode_prediction

def main():
    parser = argparse.ArgumentParser(description="Test MGU Seq2Seq Model using test.jsonl")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Path to the folder containing train.jsonl, val.jsonl, and test.jsonl (e.g., mgu_data)")
    parser.add_argument("--model-file", type=str, required=True,
                        help="Path to the model checkpoint file to test (e.g., models/mgu_seq2seq_final.pt)")
    parser.add_argument("--max-length", type=int, default=200, help="Maximum sequence length for padding")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for evaluation")
    parser.add_argument("--embedding-dim", type=int, default=256, help="Embedding dimension")
    parser.add_argument("--hidden-dim", type=int, default=512, help="Hidden dimension")
    parser.add_argument("--n-layers", type=int, default=2, help="Number of layers")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    args = parser.parse_args()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # File paths for the dataset splits
    train_file = os.path.join(args.data_dir, "train.jsonl")
    val_file = os.path.join(args.data_dir, "val.jsonl")
    test_file = os.path.join(args.data_dir, "test.jsonl")

    # Create the data processor and build vocabulary using the training data
    processor = MGUDataProcessor(data_dir=args.data_dir, max_length=args.max_length)
    processor.build_vocabulary([train_file])
    
    # Process the test data
    test_data = processor.preprocess_data([test_file])
    test_dataset = MGUDataset(test_data, args.max_length)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Determine vocabulary sizes
    input_size = len(processor.token2idx)
    output_size = len(processor.token2idx)

    # Instantiate the model architecture
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

    # Load the model weights from the given model file
    print(f"Loading model weights from {args.model_file}")
    model.load_state_dict(torch.load(args.model_file, map_location=device))
    model.eval()

    # Define the loss criterion (same as used during training)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0)

    # Evaluate the model on the test set
    #test_loss = evaluate(model, test_loader, criterion, device)
    #print(f"\nFinal Test Loss: {test_loss:.4f}")

    # Run a few sample predictions from the test set
    print("\nTesting model on some examples:")
    for i in range(min(5, len(test_dataset))):
        sample = test_dataset[i]
        src_tensor = sample['input'].unsqueeze(0).to(device)
        predictions = predict(device, model, src_tensor, args.max_length,
                              processor.token2idx['<SOS>'], processor.token2idx['<EOS>'])
        print(f"\nExample {i+1}:")
        print(f"Input: {sample['input_string']}")
        print(f"True MGU: {sample['output_string']}")
        print(f"Predicted MGU: {decode_prediction(predictions, processor.idx2token)}")

if __name__ == "__main__":
    main()

#uncomment test_loss
#python test_model.py --data-dir mgu_data --model-file models/mgu_seq2seq_finetuned2.pt