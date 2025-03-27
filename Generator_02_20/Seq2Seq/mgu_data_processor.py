import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter

class MGUDataProcessor:
    def __init__(self, data_dir="mgu_data", max_length=200, vocab_size=5000):
        """
        Initialize the data processor.
        
        Args:
            data_dir: Directory containing the generated data files
            max_length: Maximum sequence length for padding
            vocab_size: Maximum vocabulary size
        """
        self.data_dir = data_dir
        self.max_length = max_length
        self.vocab_size = vocab_size
        self.token2idx = {}
        self.idx2token = {}
        self.vocab_built = False
    
    def load_data(self, file_path):
        """Load data from a JSONL file."""
        data = []
        with open(file_path, 'r') as f:
            for line in f:
                data.append(json.loads(line))
        return data
    
    def build_vocabulary(self, data_files):
        """Build vocabulary from the training data."""
        print("Building vocabulary...")
        counter = Counter()
        
        for file_path in data_files:
            if not os.path.exists(file_path):
                print(f"Warning: File not found: {file_path}")
                continue
                
            print(f"  Processing {file_path}")
            data = self.load_data(file_path)
            
            for item in data:
                # Add tokens from input
                tokens = self._tokenize(item["input"])
                counter.update(tokens)
                
                # Add tokens from output (MGU)
                tokens = self._tokenize(item["mgu"])
                counter.update(tokens)
        
        # Special tokens
        special_tokens = ["<PAD>", "<SOS>", "<EOS>", "<UNK>"]
        
        # Create vocabularies
        self.token2idx = {token: idx for idx, token in enumerate(special_tokens)}
        
        # Add most common tokens from the counter
        idx = len(special_tokens)
        for token, _ in counter.most_common(self.vocab_size - len(special_tokens)):
            self.token2idx[token] = idx
            idx += 1
        
        # Create reverse mapping
        self.idx2token = {idx: token for token, idx in self.token2idx.items()}
        
        print(f"Vocabulary built with {len(self.token2idx)} tokens")
        self.vocab_built = True
        
        # Save vocabulary
        vocab_file = os.path.join(self.data_dir, "vocab.json")
        with open(vocab_file, 'w') as f:
            json.dump(self.token2idx, f, indent=2)
        print(f"Vocabulary saved to {vocab_file}")
    
    def _tokenize(self, text):
        """Simple tokenization by splitting on whitespace and separating special characters."""
        # Replace common symbols with space-padded versions for proper tokenization
        for symbol in ["(", ")", "←", "∨", "¬", "{", "}", ",", ":", "."]:
            text = text.replace(symbol, f" {symbol} ")
        return text.split()
    
    def _encode(self, tokens):
        """Convert tokens to indices."""
        if not self.vocab_built:
            raise ValueError("Vocabulary not built yet. Call build_vocabulary first.")
        
        # Convert tokens to indices, using <UNK> for unknown tokens
        return [self.token2idx.get(token, self.token2idx["<UNK>"]) for token in tokens]
    
    def _decode(self, indices):
        """Convert indices back to tokens."""
        if not self.vocab_built:
            raise ValueError("Vocabulary not built yet. Call build_vocabulary first.")
        
        # Convert indices back to tokens
        return [self.idx2token[idx] for idx in indices if idx in self.idx2token]
    
    def preprocess_data(self, data_files):
        """Preprocess data for training."""
        if not self.vocab_built:
            self.build_vocabulary(data_files)
        
        processed_data = []
        
        for file_path in data_files:
            if not os.path.exists(file_path):
                print(f"Warning: File not found: {file_path}")
                continue
                
            print(f"Processing {file_path}")
            data = self.load_data(file_path)
            
            for item in data:
                # Encode input
                input_tokens = self._tokenize(item["input"])
                input_indices = self._encode(input_tokens)
                
                # Encode output (MGU)
                output_tokens = self._tokenize(item["mgu"])
                output_indices = self._encode(output_tokens)
                
                # Add to processed data
                processed_item = {
                    "input_tokens": input_tokens,
                    "input_indices": input_indices,
                    "output_tokens": output_tokens,
                    "output_indices": output_indices,
                    "input_string": item["input"],
                    "output_string": item["mgu"],
                    "clause1": item["clause1"],
                    "clause2": item["clause2"]
                }
                processed_data.append(processed_item)
        
        return processed_data
    
    def prepare_dataset(self):
        """Prepare the dataset for training, validation, and testing."""
        train_file = os.path.join(self.data_dir, "train.jsonl")
        val_file = os.path.join(self.data_dir, "val.jsonl")
        test_file = os.path.join(self.data_dir, "test.jsonl")
        
        # Build vocabulary from training data
        self.build_vocabulary([train_file])
        
        # Process datasets
        train_data = self.preprocess_data([train_file])
        val_data = self.preprocess_data([val_file])
        test_data = self.preprocess_data([test_file])
        
        # Create PyTorch datasets
        train_dataset = MGUDataset(train_data, self.max_length)
        val_dataset = MGUDataset(val_data, self.max_length)
        test_dataset = MGUDataset(test_data, self.max_length)
        
        return train_dataset, val_dataset, test_dataset
    
    def prepare_curriculum_dataset(self):
        """Prepare datasets for curriculum learning."""
        # Find all level files
        level_files = []
        i = 1
        while True:
            level_file = os.path.join(self.data_dir, f"level_{i}.jsonl")
            if os.path.exists(level_file):
                level_files.append(level_file)
                i += 1
            else:
                break
        
        if not level_files:
            raise ValueError("No curriculum learning files found.")
        
        # Build vocabulary from all levels
        self.build_vocabulary(level_files)
        
        # Process each level
        level_datasets = []
        for level_file in level_files:
            data = self.preprocess_data([level_file])
            dataset = MGUDataset(data, self.max_length)
            level_datasets.append(dataset)
        
        return level_datasets
    
    def save_processed_data(self, processed_data, output_file):
        """Save processed data to file."""
        # Convert numpy arrays to lists for JSON serialization
        serializable_data = []
        for item in processed_data:
            serializable_item = {
                "input_tokens": item["input_tokens"],
                "input_indices": item["input_indices"],
                "output_tokens": item["output_tokens"],
                "output_indices": item["output_indices"],
                "input_string": item["input_string"],
                "output_string": item["output_string"],
                "clause1": item["clause1"],
                "clause2": item["clause2"]
            }
            serializable_data.append(serializable_item)
        
        with open(output_file, 'w') as f:
            json.dump(serializable_data, f)
        
        print(f"Processed data saved to {output_file}")


class MGUDataset(Dataset):
    def __init__(self, data, max_length):
        """
        Initialize the dataset.
        
        Args:
            data: List of preprocessed data items
            max_length: Maximum sequence length for padding
        """
        self.data = data
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Prepare input sequence
        input_seq = item["input_indices"]
        if len(input_seq) > self.max_length:
            input_seq = input_seq[:self.max_length]
        else:
            # Pad with zeros (PAD token)
            input_seq = input_seq + [0] * (self.max_length - len(input_seq))
        
        # Prepare output sequence with start and end tokens
        output_seq = [1] + item["output_indices"]  # 1 is <SOS>
        if len(output_seq) > self.max_length - 1:
            output_seq = output_seq[:self.max_length-1]
            output_seq.append(2)  # 2 is <EOS>
        else:
            output_seq = output_seq + [2]  # 2 is <EOS>
            output_seq = output_seq + [0] * (self.max_length - len(output_seq))
        
        # Convert to tensors
        input_tensor = torch.tensor(input_seq, dtype=torch.long)
        output_tensor = torch.tensor(output_seq, dtype=torch.long)
        
        return {
            "input": input_tensor,
            "output": output_tensor,
            "input_string": item["input_string"],
            "output_string": item["output_string"]
        }


def create_dataloaders(train_dataset, val_dataset, test_dataset, batch_size=32, num_workers=0):
    """Create DataLoaders for training, validation, and testing."""
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    return train_loader, val_loader, test_loader


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Process MGU dataset for sequence models')
    parser.add_argument('--data-dir', type=str, default='mgu_data', help='Directory containing the generated data files')
    parser.add_argument('--max-length', type=int, default=200, help='Maximum sequence length for padding')
    parser.add_argument('--vocab-size', type=int, default=5000, help='Maximum vocabulary size')
    parser.add_argument('--curriculum', action='store_true', help='Process curriculum learning datasets')
    parser.add_argument('--save-processed', action='store_true', help='Save processed data to file')
    
    args = parser.parse_args()
    
    processor = MGUDataProcessor(
        data_dir=args.data_dir,
        max_length=args.max_length,
        vocab_size=args.vocab_size
    )
    
    if args.curriculum:
        print("Processing curriculum learning datasets...")
        level_datasets = processor.prepare_curriculum_dataset()
        print(f"Processed {len(level_datasets)} curriculum levels")
        
        if args.save_processed:
            for i, dataset in enumerate(level_datasets):
                output_file = os.path.join(args.data_dir, f"processed_level_{i+1}.json")
                processor.save_processed_data([dataset[i].__dict__ for i in range(len(dataset))], output_file)
    else:
        print("Processing standard train/val/test datasets...")
        train_dataset, val_dataset, test_dataset = processor.prepare_dataset()
        print(f"Processed datasets: Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
        
        if args.save_processed:
            processor.save_processed_data(
                [train_dataset[i].__dict__ for i in range(len(train_dataset))],
                os.path.join(args.data_dir, "processed_train.json")
            )
            processor.save_processed_data(
                [val_dataset[i].__dict__ for i in range(len(val_dataset))],
                os.path.join(args.data_dir, "processed_val.json")
            )
            processor.save_processed_data(
                [test_dataset[i].__dict__ for i in range(len(test_dataset))],
                os.path.join(args.data_dir, "processed_test.json")
            )
    
    print("Done!")


if __name__ == "__main__":
    main()
