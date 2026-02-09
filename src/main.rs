mod data;
mod eval;
mod model;
mod training;

use candle_core::Device;
use clap::Parser;

use model::world_state_model::WorldStateModelConfig;
use training::trainer::{Trainer, TrainerConfig};

#[derive(Parser)]
#[command(name = "understanding-pressure")]
#[command(about = "Train SSMs with understanding pressure to eliminate context rot")]
struct Cli {
    #[arg(long, default_value = "tiny")]
    model_size: String,

    #[arg(long, default_value_t = 50)]
    epochs: usize,

    #[arg(long, default_value_t = 100)]
    examples_per_epoch: usize,

    #[arg(long, default_value_t = 10)]
    chunks: usize,

    #[arg(long, default_value_t = 32)]
    chunk_length: usize,

    #[arg(long, default_value_t = 0.5)]
    quiz_alpha: f64,

    #[arg(long, default_value_t = 1e-3)]
    learning_rate: f64,

    #[arg(long, default_value_t = 42)]
    seed: u64,

    #[arg(long, default_value_t = 5)]
    log_interval: usize,

    #[arg(long)]
    save_path: Option<String>,
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    println!("╔══════════════════════════════════════════════════╗");
    println!("║       Understanding Pressure v0.1.0             ║");
    println!("║  Training SSMs to UNDERSTAND, not just predict  ║");
    println!("╚══════════════════════════════════════════════════╝");
    println!();

    let device = Device::Cpu;

    let mut model_config = match cli.model_size.as_str() {
        "tiny" => WorldStateModelConfig::tiny(),
        "small" => WorldStateModelConfig::small(),
        _ => {
            eprintln!("Unknown model size '{}', using tiny", cli.model_size);
            WorldStateModelConfig::tiny()
        }
    };
    model_config.quiz_alpha = cli.quiz_alpha;

    let trainer_config = TrainerConfig {
        learning_rate: cli.learning_rate,
        epochs: cli.epochs,
        examples_per_epoch: cli.examples_per_epoch,
        chunks_per_example: cli.chunks,
        chunk_length: cli.chunk_length,
        log_interval: cli.log_interval,
        eval_interval: cli.log_interval,
        seed: cli.seed,
    };

    let mut trainer = Trainer::new(model_config, trainer_config, device)?;
    trainer.train()?;

    if let Some(path) = &cli.save_path {
        trainer.save(path)?;
    }

    Ok(())
}
