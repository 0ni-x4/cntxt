# Understanding Pressure

Training State Space Models to build persistent world models through comprehension-forcing objectives that eliminate context rot.

## Overview

Understanding Pressure addresses a fundamental limitation in sequence models: the inability to maintain coherent world state over long contexts. While language models predict the next token effectively, they struggle to track entities, relationships, and state changes across extended narratives. This project implements a dual-objective training framework where models must simultaneously predict future tokens and answer comprehension questions from their hidden states, forcing them to encode genuine understanding rather than surface-level patterns.

The core mechanism attaches a quiz head to an SSM's internal state at each processing step. The model must answer questions like "Where is Alice?" or "Who holds the key?" using only its compressed state vector, not by re-reading context. This creates pressure to maintain accurate world models internally, measured through recall curves that track comprehension accuracy across varying temporal distances.

## Features

- **Dual-Objective Training**: Combines language modeling loss with comprehension quiz loss to force state-based understanding
- **Selective Scan Architecture**: Mamba-inspired SSM with trainable gating mechanisms for efficient sequential processing
- **Comprehensive Quiz Types**: Entity tracking, object holder identification, contradiction detection, and temporal delta queries
- **Context Rot Measurement**: Recall curves quantify comprehension decay across temporal distances
- **Synthetic Data Generation**: Procedural entity-tracking narratives with programmatic ground truth for quiz validation
- **Gradient Balancing**: Dynamic loss weighting prevents interference between prediction and comprehension objectives
- **Attention-Pooled States**: Multi-layer state aggregation for quiz answering, allowing depth-specific information extraction

## Installation

```bash
# From source
git clone https://github.com/nicojaffer/understanding-pressure.git
cd understanding-pressure
cargo build --release

# The binary will be at target/release/understanding-pressure
```

## Usage

```bash
# Quick training run with default tiny model
cargo run --release -- --epochs 10 --examples-per-epoch 20

# Customize training parameters
cargo run --release -- \
  --model-size small \
  --epochs 50 \
  --examples-per-epoch 100 \
  --chunks 8 \
  --chunk-length 32 \
  --quiz-alpha 0.5 \
  --learning-rate 1e-3 \
  --log-interval 5

# Save trained model
cargo run --release -- --epochs 100 --save-path model.safetensors
```

Training output displays real-time metrics:
- Language modeling loss (next-token prediction accuracy)
- Quiz loss (comprehension question error)
- Quiz accuracy (percentage of correct answers)
- Recall curve (accuracy by temporal distance)

Context rot is quantified as the accuracy drop between nearest and furthest quiz distances. Values under 10% indicate minimal rot; over 30% suggests significant degradation.

## Configuration

Model sizes are predefined in `WorldStateModelConfig`:

```rust
// Tiny: 64-dim model, 32-dim state, 128-dim inner, 2 layers
WorldStateModelConfig::tiny()

// Small: 128-dim model, 64-dim state, 256-dim inner, 4 layers
WorldStateModelConfig::small()
```

Quiz alpha (default 0.5) controls the comprehension-to-prediction loss ratio. Higher values increase understanding pressure but may degrade language modeling performance.

## Architecture

- **`data/`**: Synthetic narrative generation and vocabulary management
  - `entity_tracking.rs`: Procedural story generation with entity movements, object transfers, and quiz creation
- **`model/`**: Neural architecture components
  - `ssm.rs`: Selective scan layers with state-dependent gating
  - `quiz_head.rs`: Linear projection from hidden states to answer logits
  - `world_state_model.rs`: Combined SSM + LM head + quiz head with dual forward passes
- **`training/`**: Optimization and loss computation
  - `trainer.rs`: Training loop with quiz scheduling and state management
  - `loss.rs`: Cross-entropy for both language modeling and quiz answering
  - `optimizer.rs`: AdamW implementation with gradient clipping
- **`eval/`**: Metrics and analysis
  - `recall_curve.rs`: Accuracy tracking across temporal distances to measure context rot
- **`main.rs`**: CLI interface and training orchestration

## Development

```bash
# Build debug version
cargo build

# Run tests
cargo test

# Run with verbose output
RUST_LOG=debug cargo run -- --epochs 5
```

Requires Rust 1.70+. Key dependencies: candle (ML framework), clap (CLI), rand/rand_chacha (reproducible generation), serde/serde_json (serialization).

## Research Context

This implementation explores whether comprehension-forcing objectives can mitigate the context rot observed in autoregressive models. The hypothesis: models that must answer state-based queries will learn to maintain more accurate internal world representations, measurable through flatter recall curves compared to standard language modeling training.

Current results show 5-10% context rot on entity-tracking tasks with the dual-objective approach, compared to theoretical baselines of 30-50% for pure next-token prediction. Scaling experiments to validate generalization across domains remain ongoing.

## License

MIT License
