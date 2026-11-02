"""
TrustLayer: Cross-Encoder NLI Fine-Tuning Pipeline

Fine-tunes a transformer-based cross-encoder (e.g., DeBERTa-v3 or RoBERTa) on
claim-evidence pairs for multi-class classification:
0: Contradiction
1: Neutral / Unsupported
2: Entailment / Grounded
"""

import os
import argparse
from typing import List, Dict, Any

def create_sample_dataset() -> List[Dict[str, Any]]:
    """Sample claim-evidence pairs formatted for NLI fine-tuning."""
    return [
        {
            "premise": "The James Webb Space Telescope operates at the Sun-Earth L2 Lagrange point.",
            "hypothesis": "JWST orbits around the Sun-Earth L2 point.",
            "label": 2  # Entailment
        },
        {
            "premise": "The James Webb Space Telescope operates at the Sun-Earth L2 Lagrange point.",
            "hypothesis": "The Hubble Space Telescope was launched in 2021.",
            "label": 1  # Neutral
        },
        {
            "premise": "The James Webb Space Telescope operates at the Sun-Earth L2 Lagrange point.",
            "hypothesis": "JWST operates in low Earth orbit at 500 km altitude.",
            "label": 0  # Contradiction
        },
        {
            "premise": "Python 3.10 was officially released on October 4, 2021.",
            "hypothesis": "Python 3.10 became available in late 2021.",
            "label": 2  # Entailment
        },
        {
            "premise": "Python 3.10 was officially released on October 4, 2021.",
            "hypothesis": "Python 3.10 was introduced in 1995.",
            "label": 0  # Contradiction
        }
    ]

def train(
    model_name: str = "cross-encoder/nli-deberta-v3-small",
    output_dir: str = "./models/trustlayer-nli-checkpoint",
    epochs: int = 3,
    batch_size: int = 8,
    lr: float = 2e-5
):
    print(f"[*] Initializing Cross-Encoder fine-tuning with base: {model_name}")
    print(f"[*] Output checkpoint directory: {output_dir}")
    
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
        import torch
        from datasets import Dataset

        device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[*] Training device: {device}")

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=3,
            id2label={0: "contradiction", 1: "neutral", 2: "entailment"},
            label2id={"contradiction": 0, "neutral": 1, "entailment": 2}
        )

        data = create_sample_dataset()
        raw_dataset = Dataset.from_list(data)

        def preprocess(examples):
            return tokenizer(
                examples["premise"],
                examples["hypothesis"],
                truncation=True,
                max_length=256,
                padding="max_length"
            )

        tokenized = raw_dataset.map(preprocess, batched=True)

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            learning_rate=lr,
            weight_decay=0.01,
            logging_steps=10,
            save_strategy="epoch",
            report_to="none"
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized
        )

        print("[*] Starting Trainer loop...")
        trainer.train()
        print(f"[*] Saving fine-tuned model checkpoint to {output_dir}...")
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        print("[+] Training completed successfully!")

    except ImportError as e:
        print(f"[!] Optional dependency missing for training: {e}")
        print("[!] Install via: pip install transformers datasets accelerate torch")
    except Exception as e:
        print(f"[!] Training error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune NLI Cross-Encoder for TrustLayer")
    parser.add_argument("--model", type=str, default="cross-encoder/nli-deberta-v3-small")
    parser.add_argument("--output", type=str, default="./models/trustlayer-nli-checkpoint")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    args = parser.parse_args()

    train(
        model_name=args.model,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr
    )
