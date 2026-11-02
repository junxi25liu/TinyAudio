---
language:
- en
license: mit
pipeline_tag: feature-extraction
library_name: transformers
tags:
- modernbert
---

# Ettin: an Open Suite of Paired Encoders and Decoders

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Paper](https://img.shields.io/badge/Paper-Arxiv-red)](https://arxiv.org/abs/2507.11412)
[![Models](https://img.shields.io/badge/🤗%20Hugging%20Face-12%20Models-blue)](https://huggingface.co/jhu-clsp)
[![Data](https://img.shields.io/badge/🤗%20Training%20Data-2T%20Tokens-green)](https://huggingface.co/datasets/jhu-clsp)
[![GitHub](https://img.shields.io/badge/GitHub-Code-black)](https://github.com/jhu-clsp/ettin-encoder-vs-decoder)

> 🎯 **TL;DR**: State-of-the-art paired encoder and decoder models (17M-1B params) trained identically for fair comparison with open data. Encoders beat ModernBERT. Decoders beat Llama 3.2/SmolLM2.

📄 [Paper](https://arxiv.org/abs/2507.11412) | 🚀 [GitHub Repository](https://github.com/jhu-clsp/ettin-encoder-vs-decoder)

This model is part of the Ettin suite - the first collection of paired encoder-only and decoder-only models trained with identical data, architecture, and training recipes. Ettin enables fair comparisons between encoder and decoder architectures across multiple scales, providing state-of-the-art performance for open-data models in their respective size categories.

## Table of Contents
- [Performance Highlights](#performance-highlights)
- [Quick Start](#quick-start)
- [Model Description](#model-description)
- [Training Data](#training-data)
- [Model Family](#model-family)
  - [Encoder Models](#encoder-models)
  - [Decoder Models](#decoder-models)
  - [Cross-Objective Models](#cross-objective-models)
- [Accessing Training Checkpoints](#accessing-training-checkpoints)
- [Research Applications](#research-applications)
- [Training Details](#training-details)
- [Model Architecture](#model-architecture)
- [Usage Examples](#usage-examples)
- [Fine-tuning Examples](#fine-tuning-examples)
- [Citation](#citation)

## 📊 Performance Highlights

### Encoder Tasks (vs. ModernBERT)
- **GLUE Average**: 88.9 vs 88.4 (Base), 90.8 vs 90.4 (Large)
- **MTEB v2 English Retrieval**: 45.7 vs 43.9 (Base), 48.4 vs 47.0 (Large)
- **Code Search and Long Context**: Superior performance on CodeSearchNet and MLDR

### Decoder Tasks (vs. SmolLM2 & Llama 3.2)
- **Average Score**: 46.2 vs 45.2 (SmolLM2-135M)
- **1B Model**: 59.0 vs 56.6 (Llama 3.2-1B)
- **Generative Tasks**: Competitive across all model sizes

### Key Finding
**Architecture-specific advantages persist**: A 400M encoder outperforms a 1B decoder on classification tasks, while a 400M decoder outperforms a 1B encoder on generation tasks.

## 🚀 Quick Start

### Installation
```bash
pip install torch>=1.9.0
# until the new pip release, install from main to use decoders (transformers>=4.54.X will contain it)
# encoders work with transformers>=4.48.0
pip install git+https://github.com/huggingface/transformers.git
```

### 30-Second Examples

**Encoder for Classification/Embeddings:**
```python
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("jhu-clsp/ettin-encoder-150m")
model = AutoModel.from_pretrained("jhu-clsp/ettin-encoder-150m")
```

**Decoder for Text Generation:**
```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("jhu-clsp/ettin-decoder-150m")
model = AutoModelForCausalLM.from_pretrained("jhu-clsp/ettin-decoder-150m")
```

## Model Description

Ettin models are designed to provide a foundation for comparing encoder-only and decoder-only architectures. Unlike previous comparisons that were limited by different training data, architectures, and recipes, Ettin models use:

1.  **Identical training data** - Same high-quality mixture across all models
2.  **Open Training Data** - Data is available now with batch-level training data for each of the 250+ checkpoints
3.  **Matched architectures** - Only differing in attention patterns (bidirectional vs causal) and training objectives (MLM vs CLM)
4.  **Consistent training recipe** - Three-phase training with 2T tokens
5.  **Multiple scales** - From 17M to 1B parameters

This approach allows for true apples-to-apples comparisons between encoder and decoder models, revealing the inherent strengths of each architecture.

## Training Data

The training data is publicly available and split across different phases:

-   **Pre-training Data**: [jhu-clsp/ettin-pretraining-data](https://huggingface.co/datasets/jhu-clsp/ettin-pretraining-data) - 1.7T tokens of diverse data mixture
-   **Mid-training/Extension Data**: [jhu-clsp/ettin-extension-data](https://huggingface.co/datasets/jhu-clsp/ettin-extension-data) - 250B tokens of higher-quality filtered data
-   **Decay Phase Data**: [jhu-clsp/ettin-decay-data](https://huggingface.co/datasets/jhu-clsp/ettin-decay-data) - 100B tokens of premium data sources
-   **Training Data Order**: [jhu-clsp/ettin-data-order](https://huggingface.co/datasets/jhu-clsp/ettin-data-order) - Batch-level training order (columns: input_ids, step)

## Model Family

### Encoder Models

| Size | Model | Parameters | Best For | Download |
|:-----|:------|:-----------|:---------|:---------|
| XXS | [ettin-encoder-17m](https://huggingface.co/jhu-clsp/ettin-encoder-17m) | 17M | Mobile/Edge devices | [![Download](https://img.shields.io/badge/🤗-Download-blue)](https://huggingface.co/jhu-clsp/ettin-encoder-17m) |
| XS | [ettin-encoder-32m](https://huggingface.co/jhu-clsp/ettin-encoder-32m) | 32M | Fast inference | [![Download](https://img.shields.io/badge/🤗-Download-blue)](https://huggingface.co/jhu-clsp/ettin-encoder-32m) |
| Small | [ettin-encoder-68m](https://huggingface.co/jhu-clsp/ettin-encoder-68m) | 68M | Balanced performance | [![Download](https://img.shields.io/badge/🤗-Download-blue)](https://huggingface.co/jhu-clsp/ettin-encoder-68m) |
| Base | [ettin-encoder-150m](https://huggingface.co/jhu-clsp/ettin-encoder-150m) | 150M | Standard use cases | [![Download](https://img.shields.io/badge/🤗-Download-blue)](https://huggingface.co/jhu-clsp/ettin-encoder-150m) |
| Large | [ettin-encoder-400m](https://huggingface.co/jhu-clsp/ettin-encoder-400m) | 400M | High accuracy needs | [![Download](https://img.shields.io/badge/🤗-Download-blue)](https://huggingface.co/jhu-clsp/ettin-encoder-400m) |
| XL | [ettin-encoder-1b](https://huggingface.co/jhu-clsp/ettin-encoder-1b) | 1B | Best performance | [![Download](https://img.shields.io/badge/🤗-Download-blue)](https://huggingface.co/jhu-clsp/ettin-encoder-1b) |

### Decoder Models

| Size | Model | Parameters | Best For | Download |
|:-----|:------|:-----------|:---------|:---------|
| XXS | [ettin-decoder-17m](https://huggingface.co/jhu-clsp/ettin-decoder-17m) | 17M | Lightweight generation | [![Download](https://img.shields.io/badge/🤗-Download-blue)](https://huggingface.co/jhu-clsp/ettin-decoder-17m) |
| XS | [ettin-decoder-32m](https://huggingface.co/jhu-clsp/ettin-decoder-32m) | 32M | Quick prototyping | [![Download](https://img.shields.io/badge/🤗-Download-blue)](https://huggingface.co/jhu-clsp/ettin-decoder-32m) |
| Small | [ettin-decoder-68m](https://huggingface.co/jhu-clsp/ettin-decoder-68m) | 68M | Efficient generation | [![Download](https://img.shields.io/badge/🤗-Download-blue)](https://huggingface.co/jhu-clsp/ettin-decoder-68m) |
| Base | [ettin-decoder-150m](https://huggingface.co/jhu-clsp/ettin-decoder-150m) | 150M | Standard generation | [![Download](https://img.shields.io/badge/🤗-Download-blue)](https://huggingface.co/jhu-clsp/ettin-decoder-150m) |
| Large | [ettin-decoder-400m](https://huggingface.co/jhu-clsp/ettin-decoder-400m) | 400M | Quality generation | [![Download](https://img.shields.io/badge/🤗-Download-blue)](https://huggingface.co/jhu-clsp/ettin-decoder-400m) |
| XL | [ettin-decoder-1b](https://huggingface.co/jhu-clsp/ettin-decoder-1b) | 1B | Best generation | [![Download](https://img.shields.io/badge/🤗-Download-blue)](https://huggingface.co/jhu-clsp/ettin-decoder-1b) |

### Cross-Objective Models

These models demonstrate what happens when you continue training encoders as decoders (and vice versa). **Important**: Load these models using the architecture they were *converted to*, not their original architecture.

#### Encoders Trained from Decoders (Decoder → MLM)
**Load as encoders** using `AutoModel` or `AutoModelForMaskedLM`:

| Size | Model | Parameters | Description | Download |
|:-----|:------|:-----------|:------------|:---------|
| XXS | [ettin-encoder-from-decoder-17m](https://huggingface.co/jhu-clsp/ettin-encoder-from-decoder-17m) | 17M | Decoder → MLM continued training | [![Download](https://img.shields.io/badge/🤗-Download-blue)](https://huggingface.co/jhu-clsp/ettin-encoder-from-decoder-17m) |
| XS | [ettin-encoder-from-decoder-32m](https://huggingface.co/jhu-clsp/ettin-encoder-from-decoder-32m) | 32M | Decoder → MLM continued training | [![Download](https://img.shields.io/badge/🤗-Download-blue)](https://huggingface.co/jhu-clsp/ettin-encoder-from-decoder-32m) |
| Small | [ettin-encoder-from-decoder-68m](https://huggingface.co/jhu-clsp/ettin-encoder-from-decoder-68m) | 68M | Decoder → MLM continued training | [![Download](https://img.shields.io/badge/🤗-Download-blue)](https://huggingface.co/jhu-clsp/ettin-encoder-from-decoder-68m) |
| Base | [ettin-encoder-from-decoder-150m](https://huggingface.co/jhu-clsp/ettin-encoder-from-decoder-150m) | 150M | Decoder → MLM continued training | [![Download](https://img.shields.io/badge/🤗-Download-blue)](https://huggingface.co/jhu-clsp/ettin-encoder-from-decoder-150m) |
| Large | [ettin-encoder-from-decoder-400m](https://huggingface.co/jhu-clsp/ettin-encoder-from-decoder-400m) | 400M | Decoder → MLM continued training | [![Download](https://img.shields.io/badge/🤗-Download-blue)](https://huggingface.co/jhu-clsp/ettin-encoder-from-decoder-400m) |
| XL | [ettin-encoder-from-decoder-1b](https://huggingface.co/jhu-clsp/ettin-encoder-from-decoder-1b) | 1B | Decoder → MLM continued training | [![Download](https://img.shields.io/badge/🤗-Download-blue)](https://huggingface.co/jhu-clsp/ettin-encoder-from-decoder-1b) |

#### Decoders Trained from Encoders (Encoder → CLM)
**Load as decoders** using `AutoModelForCausalLM`:

| Size | Model | Parameters | Description | Download |
|:-----|:------|:-----------|:------------|:---------|
| XXS | [ettin-decoder-from-encoder-17m](https://huggingface.co/jhu-clsp/ettin-decoder-from-encoder-17m) | 17M | Encoder → CLM continued training | [![Download](https://img.shields.io/badge/🤗-Download-blue)](https://huggingface.co/jhu-clsp/ettin-decoder-from-encoder-17m) |
| XS | [ettin-decoder-from-encoder-32m](https://huggingface.co/jhu-clsp/ettin-decoder-from-encoder-32m) | 32M | Encoder → CLM continued training | [![Download](https://img.shields.io/badge/🤗-Download-blue)](https://huggingface.co/jhu-clsp/ettin-decoder-from-encoder-32m) |
| Small | [ettin-decoder-from-encoder-68m](https://huggingface.co/jhu-clsp/ettin-decoder-from-encoder-68m) | 68M | Encoder → CLM continued training | [![Download](https://img.shields.io/badge/🤗-Download-blue)](https://huggingface.co/jhu-clsp/ettin-decoder-68m) |
| Base | [ettin-decoder-from-encoder-150m](https://huggingface.co/jhu-clsp/ettin-decoder-from-encoder-150m) | 150M | Encoder → CLM continued training | [![Download](https://img.shields.io/badge/🤗-Download-blue)](https://huggingface.co/jhu-clsp/ettin-decoder-from-encoder-150m) |
| Large | [ettin-decoder-from-encoder-400m](https://huggingface.co/jhu-clsp/ettin-decoder-from-encoder-400m) | 400M | Encoder → CLM continued training | [![Download](https://img.shields.io/badge/🤗-Download-blue)](https://huggingface.co/jhu-clsp/ettin-decoder-400m) |
| XL | [ettin-decoder-from-encoder-1b](https://huggingface.co/jhu-clsp/ettin-decoder-from-encoder-1b) | 1B | Encoder → CLM continued training | [![Download](https://img.shields.io/badge/🤗-Download-blue)](https://huggingface.co/jhu-clsp/ettin-decoder-1b) |

**Example Usage for Cross-Objective Models:**
```python
# Encoder-from-decoder: Load as encoder
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("jhu-clsp/ettin-encoder-from-decoder-150m")
model = AutoModel.from_pretrained("jhu-clsp/ettin-encoder-from-decoder-150m")

# Decoder-from-encoder: Load as decoder  
from transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained("jhu-clsp/ettin-decoder-from-encoder-150m")
model = AutoModelForCausalLM.from_pretrained("jhu-clsp/ettin-decoder-from-encoder-150m")
```

## Accessing Training Checkpoints

Beyond the final models listed above, we provide access to intermediate training checkpoints for research and analysis purposes. These checkpoints allow you to study model behavior and performance throughout the training process. You can get the checkpoints either in HF format or raw for continued pre-training (e.g. Composer format).

#### Raw Checkpoints
All raw training checkpoints are available in the [jhu-clsp/ettin-checkpoints](https://huggingface.co/datasets/jhu-clsp/ettin-checkpoints) dataset.

#### HuggingFace Format Checkpoints
Each model repository contains multiple tagged versions representing different training stages:

-   **`step{number}`** - Pretraining phase checkpoints (e.g., `step599525`, `step596528`)
-   **`ext{number}`** - Extension/mid-training phase checkpoints (e.g., `ext1000`, `ext2000`) 
-   **`decay{number}`** - Decay phase checkpoints (e.g., `decay100`, `decay500`)

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load a specific pretraining checkpoint
model = AutoModelForCausalLM.from_pretrained(
    "jhu-clsp/ettin-decoder-400m", 
    revision="step590532"  # Specific checkpoint tag
)

# Load an extension phase checkpoint
model = AutoModelForCausalLM.from_pretrained(
    "jhu-clsp/ettin-decoder-400m", 
    revision="ext1000"
)

# Load a decay phase checkpoint  
model = AutoModelForCausalLM.from_pretrained(
    "jhu-clsp/ettin-decoder-400m", 
    revision="decay100"
)
```

This checkpoint availability enables detailed analysis of training dynamics, loss curves, and capability emergence across the complete 2T token training process.


## 🔬 Research Applications

### What Makes Ettin Unique

Ettin provides the first **controlled comparison** of encoder vs. decoder architectures:

-   **Identical Training Data**: Same 2T token mixture across all models
-   **Matched Architectures**: Only attention patterns and objectives differ  
-   **Open Everything**: Training data, model weights, and batch-level training order
-   **Multiple Scales**: Fair comparison from 17M to 1B parameters
-   **250+ Checkpoints**: Complete training trajectory analysis

### Use Cases for Researchers

-   **Architecture Studies**: Compare encoder vs decoder capabilities fairly
-   **Training Dynamics**: Analyze 250+ checkpoints with batch-level data ordering  
-   **Scaling Laws**: Study how architectural advantages change with scale
-   **Transfer Learning**: Investigate cross-objective training effectiveness
-   **Replication Studies**: First open replication of ModernBERT training recipe

### Reproducibility

All training artifacts are publicly available:
- Training data with exact batch ordering
- Model checkpoints every 8.5B tokens
- Complete hyperparameter configurations
- Training code and evaluation scripts

## Training Details

**Data:** High-quality mixture including DCLM, Dolma v1.7, scientific papers, code, and curated sources totaling 2T+ tokens

**Architecture:** Transformer with RoPE, GLU activations, and prenorm layers

**Training Phases:**
-   **Pre-training**: 1.7T tokens with diverse data mixture
-   **Mid-training**: 250B tokens with higher-quality filtered data and context extension to 8K
-   **Decay phase**: 100B tokens with premium data sources

**Key Features:**
-   Context length: Up to 8K tokens
-   Vocabulary: 50,368 tokens (ModernBERT tokenizer)
-   Deep but efficient architectures following MobileLLM principles

## Model Architecture

| Parameter | 17M | 32M | 68M | 150M | 400M | 1B |
|:----------|:----|:----|:----|:-----|:-----|:---|
| Layers | 7 | 10 | 19 | 22 | 28 | 28 |
| Hidden Size | 256 | 384 | 512 | 768 | 1024 | 1792 |
| Intermediate Size | 384 | 576 | 768 | 1152 | 2624 | 3840 |
| Attention Heads | 4 | 6 | 8 | 12 | 16 | 28 |



## Usage Examples

### Encoder: Masked Language Modeling
<details>
<summary>Click to expand **encoder** usage examples</summary>

```python
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch

# Load MLM model
tokenizer = AutoTokenizer.from_pretrained("jhu-clsp/ettin-encoder-150m")
model = AutoModelForMaskedLM.from_pretrained("jhu-clsp/ettin-encoder-150m")

def predict_masked_token(text):
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get predictions for [MASK] tokens
    mask_indices = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)
    predictions = outputs.logits[mask_indices]
    
    # Get top 5 predictions
    top_tokens = torch.topk(predictions, 5, dim=-1)
    return [tokenizer.decode(token) for token in top_tokens.indices[0]]

# Example
masked_text = "The capital of France is [MASK]."
predictions = predict_masked_token(masked_text)
print(f"Predictions: {predictions}")
```

</details>

### Decoder: Text Generation

<details>
<summary>Click to expand **decoder text generation**</summary>
  
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load model and tokenizer  
tokenizer = AutoTokenizer.from_pretrained("jhu-clsp/ettin-decoder-150m")
model = AutoModelForCausalLM.from_pretrained("jhu-clsp/ettin-decoder-150m")

# Set pad token if needed
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def generate_text(prompt, max_length=100, temperature=0.7):
    inputs = tokenizer(prompt, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_length=max_length,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            num_return_sequences=1
        )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Example usage
prompt = "The future of artificial intelligence is"
generated = generate_text(prompt)
print(generated)
```

</details>


## Fine-tuning Examples

### Encoders
<details><summary>Click to see how to finetune this into a dense embedding model using Sentence Transformers</summary> 

```python
import argparse

from datasets import load_dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.evaluation import TripletEvaluator
from sentence_transformers.losses import CachedMultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers

def main():
    # parse the lr & model name
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=8e-5)
    parser.add_argument("--model_name", type=str, default="jhu-clsp/ettin-encoder-150m")
    args = parser.parse_args()
    lr = args.lr
    model_name = args.model_name
    model_shortname = model_name.split("/")[-1]

    # 1. Load a model to finetune
    model = SentenceTransformer(model_name)

    # 2. Load a dataset to finetune on
    dataset = load_dataset(
        "sentence-transformers/msmarco-co-condenser-margin-mse-sym-mnrl-mean-v1",
        "triplet-hard",
        split="train",
    )
    dataset_dict = dataset.train_test_split(test_size=1_000, seed=12)
    train_dataset = dataset_dict["train"].select(range(1_250_000))
    eval_dataset = dataset_dict["test"]

    # 3. Define a loss function
    loss = CachedMultipleNegativesRankingLoss(model, mini_batch_size=16)  # Increase mini_batch_size if you have enough VRAM

    run_name = f"{model_shortname}-DPR-{lr}"
    # 4. (Optional) Specify training arguments
    args = SentenceTransformerTrainingArguments(
        # Required parameter:
        output_dir=f"output/{model_shortname}/{run_name}",
        # Optional training parameters:
        num_train_epochs=1,
        per_device_train_batch_size=512,
        per_device_eval_batch_size=512,
        warmup_ratio=0.05,
        fp16=False,  # Set to False if GPU can't handle FP16
        bf16=True,  # Set to True if GPU supports BF16
        batch_sampler=BatchSamplers.NO_DUPLICATES,  # (Cached)MultipleNegativesRankingLoss benefits from no duplicates
        learning_rate=lr,
        # Optional tracking/debugging parameters:
        save_strategy="steps",
        save_steps=500,
        save_total_limit=2,
        logging_steps=500,
        run_name=run_name,  # Used in `wandb`, `tensorboard`, `neptune`, etc. if installed
    )

    # 5. (Optional) Create an evaluator & evaluate the base model
    dev_evaluator = TripletEvaluator(
        anchors=eval_dataset["query"],
        positives=eval_dataset["positive"],
        negatives=eval_dataset["negative"],
        name="msmarco-co-condenser-dev",
    )
    dev_evaluator(model)

    # 6. Create a trainer & train
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=loss,
        evaluator=dev_evaluator,
    )
    trainer.train()

    # 7. (Optional) Evaluate the trained model on the evaluator after training
    dev_evaluator(model)

    # 8. Save the model
    model.save_pretrained(f"output/{model_shortname}/{run_name}/final")

    # 9. (Optional) Push it to the Hugging Face Hub
    model.push_to_hub(run_name, private=False)

if __name__ == "__main__":
    main()
```
</details>


<details><summary>Click to see how to finetune this into a multi-vector embedding model with PyLate</summary>

```python
from datasets import load_dataset
from pylate import losses, models, utils
from sentence_transformers import (
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)

def main():
    # Load the datasets required for knowledge distillation (train, queries, documents)
    train = load_dataset(
        path="lightonai/ms-marco-en-bge",
        name="train",
    )

    queries = load_dataset(
        path="lightonai/ms-marco-en-bge",
        name="queries",
    )

    documents = load_dataset(
        path="lightonai/ms-marco-en-bge",
        name="documents",
    )

    # Set the transformation to load the documents/queries texts using the corresponding ids on the fly
    train.set_transform(
        utils.KDProcessing(queries=queries, documents=documents).transform,
    )

    # Define the base model, training parameters, and output directory
    num_train_epochs = 1
    lr = 8e-5
    batch_size = 16
    accum_steps = 1
    model_name = "jhu-clsp/ettin-encoder-150m"
    model_shortname = model_name.split("/")[-1]

    # Set the run name for logging and output directory
    run_name = f"{model_shortname}-colbert-KD-{lr}"
    output_dir = f"output/{model_shortname}/{run_name}"

    # Initialize the ColBERT model from the base model
    model = models.ColBERT(model_name_or_path=model_name)

    # Configure the training arguments (e.g., epochs, batch size, learning rate)
    args = SentenceTransformerTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=batch_size,
        fp16=False,  # Set to False if you get an error that your GPU can't run on FP16
        bf16=True,  # Set to True if you have a GPU that supports BF16
        run_name=run_name,
        logging_steps=10,
        learning_rate=lr,
        gradient_accumulation_steps=accum_steps,
        warmup_ratio=0.05,
    )

    # Use the Distillation loss function for training
    train_loss = losses.Distillation(model=model)

    # Initialize the trainer
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train,
        loss=train_loss,
        data_collator=utils.ColBERTCollator(tokenize_fn=model.tokenize),
    )

    # Start the training process
    trainer.train()

    model.save_pretrained(f"{output_dir}/final")

if __name__ == "__main__":
    main()

```
</details>

<details><summary>Click to see how to finetune this into a sparse retrieval model using Sentence Transformers</summary>

```python
import logging

from datasets import load_dataset

from sentence_transformers import (
    SparseEncoder,
    SparseEncoderModelCardData,
    SparseEncoderTrainer,
    SparseEncoderTrainingArguments,
)
from sentence_transformers.sparse_encoder.evaluation import SparseNanoBEIREvaluator
from sentence_transformers.sparse_encoder.losses import SparseMultipleNegativesRankingLoss, SpladeLoss
from sentence_transformers.training_args import BatchSamplers

logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)

# 1. Load a model to finetune with 2. (Optional) model card data
model = SparseEncoder(
    "jhu-clsp/ettin-encoder-150m",
    model_card_data=SparseEncoderModelCardData(
        language="en",
        license="apache-2.0",
    )
)

# 3. Load a dataset to finetune on
full_dataset = load_dataset("sentence-transformers/natural-questions", split="train").select(range(100_000))
dataset_dict = full_dataset.train_test_split(test_size=1_000, seed=12)
train_dataset = dataset_dict["train"]
eval_dataset = dataset_dict["test"]

# 4. Define a loss function
loss = SpladeLoss(
    model=model,
    loss=SparseMultipleNegativesRankingLoss(model=model),
    query_regularizer_weight=5e-5,
    document_regularizer_weight=3e-5,
)

# 5. (Optional) Specify training arguments
run_name = "splade-distilbert-base-uncased-nq"
args = SparseEncoderTrainingArguments(
    # Required parameter:
    output_dir=f"models/{run_name}",
    # Optional training parameters:
    num_train_epochs=1,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=2e-5,
    warmup_ratio=0.1,
    fp16=True,  # Set to False if you get an error that your GPU can't run on FP16
    bf16=False,  # Set to True if you have a GPU that supports BF16
    batch_sampler=BatchSamplers.NO_DUPLICATES,  # MultipleNegativesRankingLoss benefits from no duplicate samples in a batch
    # Optional tracking/debugging parameters:
    eval_strategy="steps",
    eval_steps=1000,
    save_strategy="steps",
    save_steps=1000,
    save_total_limit=2,
    logging_steps=200,
    run_name=run_name,  # Will be used in W&B if `wandb` is installed
)

# 6. (Optional) Create an evaluator & evaluate the base model
dev_evaluator = SparseNanoBEIREvaluator(dataset_names=["msmarco", "nfcorpus", "nq"], batch_size=16)

# 7. Create a trainer & train
trainer = SparseEncoderTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    loss=loss,
    evaluator=dev_evaluator,
)
trainer.train()

# 8. Evaluate the model performance again after training
dev_evaluator(model)

# 9. Save the trained model
model.save_pretrained(f"models/{run_name}/final")

# 10. (Optional) Push it to the Hugging Face Hub
model.push_to_hub(run_name)

```
</details>

<details><summary>Click to see how to finetune this into a reranker model using Sentence Transformers</summary>

```python
import logging
import traceback

import torch
from datasets import load_dataset

from sentence_transformers import SentenceTransformer
from sentence_transformers.cross_encoder import (
    CrossEncoder,
    CrossEncoderModelCardData,
    CrossEncoderTrainer,
    CrossEncoderTrainingArguments,
)
from sentence_transformers.cross_encoder.evaluation import (
    CrossEncoderNanoBEIREvaluator,
    CrossEncoderRerankingEvaluator,
)
from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss
from sentence_transformers.evaluation import SequentialEvaluator
from sentence_transformers.util import mine_hard_negatives

# Set the log level to INFO to get more information
logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)


def main():
    model_name = "jhu-clsp/ettin-encoder-150m"

    train_batch_size = 64
    num_epochs = 1
    num_hard_negatives = 5  # How many hard negatives should be mined for each question-answer pair

    # 1a. Load a model to finetune with 1b. (Optional) model card data
    model = CrossEncoder(
        model_name,
        model_card_data=CrossEncoderModelCardData(
            language="en",
            license="apache-2.0",
        ),
    )
    print("Model max length:", model.max_length)
    print("Model num labels:", model.num_labels)

    # 2a. Load the GooAQ dataset: https://huggingface.co/datasets/sentence-transformers/gooaq
    logging.info("Read the gooaq training dataset")
    full_dataset = load_dataset("sentence-transformers/gooaq", split="train").select(range(100_000))
    dataset_dict = full_dataset.train_test_split(test_size=1_000, seed=12)
    train_dataset = dataset_dict["train"]
    eval_dataset = dataset_dict["test"]
    logging.info(train_dataset)
    logging.info(eval_dataset)

    # 2b. Modify our training dataset to include hard negatives using a very efficient embedding model
    embedding_model = SentenceTransformer("sentence-transformers/static-retrieval-mrl-en-v1", device="cpu")
    hard_train_dataset = mine_hard_negatives(
        train_dataset,
        embedding_model,
        num_negatives=num_hard_negatives,  # How many negatives per question-answer pair
        margin=0,  # Similarity between query and negative samples should be x lower than query-positive similarity
        range_min=0,  # Skip the x most similar samples
        range_max=100,  # Consider only the x most similar samples
        sampling_strategy="top",  # Sample the top negatives from the range
        batch_size=4096,  # Use a batch size of 4096 for the embedding model
        output_format="labeled-pair",  # The output format is (query, passage, label), as required by BinaryCrossEntropyLoss
        use_faiss=True,
    )
    logging.info(hard_train_dataset)

    # 2c. (Optionally) Save the hard training dataset to disk
    # hard_train_dataset.save_to_disk("gooaq-hard-train")
    # Load again with:
    # hard_train_dataset = load_from_disk("gooaq-hard-train")

    # 3. Define our training loss.
    # pos_weight is recommended to be set as the ratio between positives to negatives, a.k.a. `num_hard_negatives`
    loss = BinaryCrossEntropyLoss(model=model, pos_weight=torch.tensor(num_hard_negatives))

    # 4a. Define evaluators. We use the CrossEncoderNanoBEIREvaluator, which is a light-weight evaluator for English reranking
    nano_beir_evaluator = CrossEncoderNanoBEIREvaluator(
        dataset_names=["msmarco", "nfcorpus", "nq"],
        batch_size=train_batch_size,
    )

    # 4b. Define a reranking evaluator by mining hard negatives given query-answer pairs
    # We include the positive answer in the list of negatives, so the evaluator can use the performance of the
    # embedding model as a baseline.
    hard_eval_dataset = mine_hard_negatives(
        eval_dataset,
        embedding_model,
        corpus=full_dataset["answer"],  # Use the full dataset as the corpus
        num_negatives=30,  # How many documents to rerank
        batch_size=4096,
        include_positives=True,
        output_format="n-tuple",
        use_faiss=True,
    )
    logging.info(hard_eval_dataset)
    reranking_evaluator = CrossEncoderRerankingEvaluator(
        samples=[
            {
                "query": sample["question"],
                "positive": [sample["answer"]],
                "documents": [sample[column_name] for column_name in hard_eval_dataset.column_names[2:]],
            }
            for sample in hard_eval_dataset
        ],
        batch_size=train_batch_size,
        name="gooaq-dev",
        # Realistic setting: only rerank the positives that the retriever found
        # Set to True to rerank *all* positives
        always_rerank_positives=False,
    )

    # 4c. Combine the evaluators & run the base model on them
    evaluator = SequentialEvaluator([reranking_evaluator, nano_beir_evaluator])
    evaluator(model)

    # 5. Define the training arguments
    short_model_name = model_name if "/" not in model_name else model_name.split("/")[-1]
    run_name = f"reranker-{short_model_name}-gooaq-bce"
    args = CrossEncoderTrainingArguments(
        # Required parameter:
        output_dir=f"models/{run_name}",
        # Optional training parameters:
        num_train_epochs=num_epochs,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=train_batch_size,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        fp16=False,  # Set to False if you get an error that your GPU can't run on FP16
        bf16=True,  # Set to True if you have a GPU that supports BF16
        dataloader_num_workers=4,
        load_best_model_at_end=True,
        metric_for_best_model="eval_gooaq-dev_ndcg@10",
        # Optional tracking/debugging parameters:
        eval_strategy="steps",
        eval_steps=1000,
        save_strategy="steps",
        save_steps=1000,
        save_total_limit=2,
        logging_steps=200,
        logging_first_step=True,
        run_name=run_name,  # Will be used in W&B if `wandb` is installed
        seed=12,
    )

    # 6. Create the trainer & start training
    trainer = CrossEncoderTrainer(
        model=model,
        args=args,
        train_dataset=hard_train_dataset,
        loss=loss,
        evaluator=evaluator,
    )
    trainer.train()

    # 7. Evaluate the final model, useful to include these in the model card
    evaluator(model)

    # 8. Save the final model
    final_output_dir = f"models/{run_name}/final"
    model.save_pretrained(final_output_dir)

    # 9. (Optional) save the model to the Hugging Face Hub!
    # It is recommended to run `huggingface-cli login` to log into your Hugging Face account first
    try:
        model.push_to_hub(run_name)
    except Exception:
        logging.error(
            f"Error uploading model to the Hugging Face Hub:
{traceback.format_exc()}To upload it manually, you can run "
            f"`huggingface-cli login`, followed by loading the model using `model = CrossEncoder({final_output_dir!r})` "
            f"and saving it using `model.push_to_hub('{run_name}')`."
        )


if __name__ == "__main__":
    main()

```
</details>

### Decoders

<details>
<summary>Click to expand decoder training code</summary>

# Full training
```bash
python trl/scripts/sft.py \
    --model_name_or_path jhu-clsp/ettin-decoder-17m \
    --dataset_name trl-lib/Capybara \
    --learning_rate 2.0e-5 \
    --num_train_epochs 1 \
    --packing \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing \
    --eos_token '<|im_end|>' \
    --eval_strategy steps \
    --eval_steps 100 \
    --output_dir ettin-decoder-17m \
    --push_to_hub
```

# LoRA
```bash
python trl/scripts/sft.py \
    --model_name_or_path jhu-clsp/ettin-decoder-17m \
    --dataset_name trl-lib/Capybara \
    --learning_rate 2.0e-4 \
    --num_train_epochs 1 \
    --packing \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing \
    --eos_token '<|im_end|>' \
    --eval_strategy steps \
    --eval_steps 100 \
    --use_peft \
    --lora_r 32 \
    --lora_alpha 16 \
    --output_dir ettin-decoder-17m \
    --push_to_hub
```

with `sft.py`:
```python
import argparse

from datasets import load_dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.models.auto.modeling_auto import MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES

from trl import (
    ModelConfig,
    ScriptArguments,
    SFTConfig,
    SFTTrainer,
    TrlParser,
    clone_chat_template,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)


def main(script_args, training_args, model_args):
    ################
    # Model init kwargs & Tokenizer
    ################
    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=model_args.torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )

    # Create model
    config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    valid_image_text_architectures = MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES.values()

    if config.architectures and any(arch in valid_image_text_architectures for arch in config.architectures):
        from transformers import AutoModelForImageTextToText

        model_kwargs.pop("use_cache", None)  # Image models do not support cache
        model = AutoModelForImageTextToText.from_pretrained(model_args.model_name_or_path, **model_kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, **model_kwargs)

    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code, use_fast=True
    )

    # Set default chat template if needed
    if tokenizer.chat_template is None:
        # TODO: source should be passed as an argument
        model, tokenizer = clone_chat_template(model, tokenizer, "Qwen/Qwen3-0.6B")

    ################
    # Dataset
    ################
    dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)

    ################
    # Training
    ################
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args),
    )

    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


def make_parser(subparsers: argparse._SubParsersAction = None):
    dataclass_types = (ScriptArguments, SFTConfig, ModelConfig)
    if subparsers is not None:
        parser = subparsers.add_parser("sft", help="Run the SFT training script", dataclass_types=dataclass_types)
    else:
        parser = TrlParser(dataclass_types)
    return parser


if __name__ == "__main__":
    parser = make_parser()
    # When using the trl cli, this script may be run with additional arguments, corresponding accelerate arguments.
    # To ensure that their parsing does not interfere with the script arguments, parse the arguments with
    # `return_remaining_strings=True`, then ignore the remaining strings.
    script_args, training_args, model_args, _ = parser.parse_args_and_config(return_remaining_strings=True)
    main(script_args, training_args, model_args)

```
</details>

## Citation

If you use Ettin models in your research, please cite our work:

```bibtex
@misc{weller2025seqvsseqopen,
      title={Seq vs Seq: An Open Suite of Paired Encoders and Decoders}, 
      author={Orion Weller and Kathryn Ricci and Marc Marone and Antoine Chaffin and Dawn Lawrie and Benjamin Van Durme},
      year={2025},
      eprint={2507.11412},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2507.11412}, 
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Contact**: For questions about the models or research, please open an issue or contact the authors.