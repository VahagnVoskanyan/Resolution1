import os
import json
import torch
import torch.nn as nn
import torch.optim as optim

# 1) Generate new dataset (dataset_generator_saves.py)
from dataset_generator_saves import MGUDatasetCreator

# 2) Preprocess new dataset (mgu_data_processor.py)
from mgu_data_processor import MGUDataProcessor, create_dataloaders

# 3) Import model pieces from mgu_seq2seq_model.py
from mgu_seq2seq_model import (
    Encoder, Decoder, Seq2Seq,
    train, evaluate, predict, decode_prediction
)

def fine_tune_once(
    data_folder="mgu_data",
    models_folder="models",
    existing_checkpoint="mgu_seq2seq_best.pt",
    new_checkpoint="mgu_seq2seq_finetuned.pt",
    train_size=8000,
    val_size=1000,
    test_size=1000,
    batch_size=32,
    lr=1e-4,
    n_epochs=5,
    clip=1.0,
    embedding_dim=256,
    hidden_dim=512,
    n_layers=2,
    dropout=0.1,
    max_length=200
):
    """
    Generates new data in `mgu_data/`, loads old model from `models/`,
    fine-tunes on the newly generated data, evaluates, and saves a new model.
    """
    # --------------------------
    # Device Setup
    # --------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Ensure directories exist
    os.makedirs(data_folder, exist_ok=True)
    os.makedirs(models_folder, exist_ok=True)

    # --------------------------
    # Step 1: Generate new dataset
    # --------------------------
    print("\n=== Generating fresh train/val/test data ===")
    creator = MGUDatasetCreator(output_dir=data_folder, random_seed=42)
    creator.generate_dataset(
        train_size=train_size,
        val_size=val_size,
        test_size=test_size
    )
    # Now you should have train.jsonl, val.jsonl, test.jsonl in mgu_data/

    # --------------------------
    # Step 2: Preprocess
    # --------------------------
    print("\n=== Preprocessing data ===")
    processor = MGUDataProcessor(
        data_dir=data_folder,
        max_length=max_length,
        vocab_size=5000  # can adjust
    )
    train_dataset, val_dataset, test_dataset = processor.prepare_dataset()
    train_loader, val_loader, test_loader = create_dataloaders(
        train_dataset, val_dataset, test_dataset,
        batch_size=batch_size
    )

    print(f"Train set size: {len(train_dataset)}")
    print(f"Val set size:   {len(val_dataset)}")
    print(f"Test set size:  {len(test_dataset)}")

    # --------------------------
    # Step 3: Load old model checkpoint
    # --------------------------
    print(f"\n=== Loading previous model checkpoint: {existing_checkpoint} ===")
    existing_ckpt_path = os.path.join(models_folder, existing_checkpoint)
    if not os.path.isfile(existing_ckpt_path):
        # Let the user specify a full path if needed:
        if os.path.isfile(existing_checkpoint):
            existing_ckpt_path = existing_checkpoint
        else:
            raise FileNotFoundError(f"Cannot find checkpoint: {existing_ckpt_path}")

    input_size = len(processor.token2idx)
    output_size = len(processor.token2idx)

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

    # Load weights
    model.load_state_dict(torch.load(existing_ckpt_path, map_location=device))
    print("Loaded model parameters successfully.")

    # --------------------------
    # Step 4: Fine-tune
    # --------------------------
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    print("\n=== Fine-tuning the model ===")
    best_val_loss = float("inf")
    new_ckpt_path = os.path.join(models_folder, new_checkpoint)

    for epoch in range(1, n_epochs + 1):
        print(f"\nEpoch {epoch}/{n_epochs}")
        train_loss = train(model, train_loader, optimizer, criterion, clip, device)
        val_loss = evaluate(model, val_loader, criterion, device)
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # If validation improved, save checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), new_ckpt_path)
            print(f"  [*] Saved improved model to {new_ckpt_path}")

    # --------------------------
    # Step 5: Evaluate on test set
    # --------------------------
    print("\n=== Testing best model on newly created test set ===")
    # Load the best checkpoint
    model.load_state_dict(torch.load(new_ckpt_path, map_location=device))
    test_loss = evaluate(model, test_loader, criterion, device)
    print(f"Final Test Loss: {test_loss:.4f}")

    # Show a few sample predictions
    print("\n=== Sample predictions on test data ===")
    for i in range(5):
        sample = test_dataset[i]
        src_tensor = sample["input"].unsqueeze(0).to(device)
        pred_indices = predict(
            device,
            model,
            src_tensor,
            max_len=max_length,
            sos_idx=processor.token2idx["<SOS>"],
            eos_idx=processor.token2idx["<EOS>"]
        )
        pred_str = decode_prediction(pred_indices, processor.idx2token)
        print(f"\nExample {i+1}:")
        print(f"Input:       {sample['input_string']}")
        print(f"True MGU:    {sample['output_string']}")
        print(f"Predicted:   {pred_str}")

    print(f"\nDone! New fine-tuned model is stored as {new_checkpoint} in the {models_folder} folder.")

if __name__ == "__main__":
    # Example usage:
    # 1) First run: uses "mgu_seq2seq_best.pt" as existing model, saves "mgu_seq2seq_finetuned.pt"
    # 2) Next run:  pass existing_checkpoint="mgu_seq2seq_finetuned.pt" to build on top of previous fine-tune
    fine_tune_once(
        data_folder="mgu_data",
        models_folder="models",
        existing_checkpoint="mgu_seq2seq_finetuned1.pt",   # your old model file
        new_checkpoint="mgu_seq2seq_finetuned2.pt",   # new fine-tuned model
        train_size=8000,
        val_size=1000,
        test_size=1000,
        batch_size=32,
        lr=1e-4,
        n_epochs=10,
        clip=1.0,
        embedding_dim=256,
        hidden_dim=512,
        n_layers=2,
        dropout=0.1,
        max_length=200
    )
