use candle_core::{DType, Device, Result, Tensor};
use candle_nn::optim::{AdamW, Optimizer, ParamsAdamW};
use candle_nn::VarMap;

use crate::data::entity_tracking::{EntityTrackingGenerator, TrainingExample, Vocabulary};
use crate::eval::recall_curve::RecallTracker;
use crate::model::world_state_model::{WorldStateModel, WorldStateModelConfig};
use crate::training::loss;

pub struct TrainerConfig {
    pub learning_rate: f64,
    pub epochs: usize,
    pub examples_per_epoch: usize,
    pub chunks_per_example: usize,
    pub chunk_length: usize,
    pub log_interval: usize,
    pub eval_interval: usize,
    pub seed: u64,
}

impl Default for TrainerConfig {
    fn default() -> Self {
        TrainerConfig {
            learning_rate: 1e-3,
            epochs: 50,
            examples_per_epoch: 100,
            chunks_per_example: 10,
            chunk_length: 32,
            log_interval: 10,
            eval_interval: 10,
            seed: 42,
        }
    }
}

pub struct Trainer {
    model: WorldStateModel,
    varmap: VarMap,
    optimizer: AdamW,
    vocab: Vocabulary,
    data_gen: EntityTrackingGenerator,
    config: TrainerConfig,
    device: Device,
    recall_tracker: RecallTracker,
}

impl Trainer {
    pub fn new(
        model_config: WorldStateModelConfig,
        trainer_config: TrainerConfig,
        device: Device,
    ) -> Result<Self> {
        let varmap = VarMap::new();
        let vb = candle_nn::VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let mut vocab = Vocabulary::new();

        let actual_config = WorldStateModelConfig {
            ssm: crate::model::ssm::SsmConfig {
                d_model: model_config.ssm.d_model,
                d_state: model_config.ssm.d_state,
                d_inner: model_config.ssm.d_inner,
                n_layers: model_config.ssm.n_layers,
                vocab_size: vocab.size().max(model_config.ssm.vocab_size),
            },
            num_quiz_answers: model_config.num_quiz_answers,
            quiz_alpha: model_config.quiz_alpha,
        };

        let model = WorldStateModel::new(actual_config, vb)?;

        let optimizer = AdamW::new(
            varmap.all_vars(),
            ParamsAdamW {
                lr: trainer_config.learning_rate,
                weight_decay: 0.01,
                ..Default::default()
            },
        )?;

        let data_gen = EntityTrackingGenerator::new(trainer_config.seed);
        let recall_tracker = RecallTracker::new();

        Ok(Self {
            model,
            varmap,
            optimizer,
            vocab,
            data_gen,
            config: trainer_config,
            device,
            recall_tracker,
        })
    }

    pub fn train(&mut self) -> Result<()> {
        println!("=== Understanding Pressure Training ===");
        println!("Device: {:?}", self.device);
        println!("Vocab size: {}", self.vocab.size());
        println!(
            "Model: d_model={}, d_state={}, d_inner={}, layers={}",
            self.model.d_model(),
            self.model.config().ssm.d_state,
            self.model.config().ssm.d_inner,
            self.model.config().ssm.n_layers,
        );
        println!("Quiz alpha: {}", self.model.config().quiz_alpha);
        println!(
            "Epochs: {}, Examples/epoch: {}",
            self.config.epochs, self.config.examples_per_epoch
        );
        println!();

        for epoch in 0..self.config.epochs {
            let mut epoch_lm_loss = 0.0;
            let mut epoch_quiz_loss = 0.0;
            let mut epoch_quiz_acc = 0.0;
            let mut quiz_count = 0;
            let mut total_examples = 0;

            for ex_idx in 0..self.config.examples_per_epoch {
                let example = self
                    .data_gen
                    .generate(&mut self.vocab, self.config.chunks_per_example);

                match self.train_example(&example) {
                    Ok(stats) => {
                        epoch_lm_loss += stats.lm_loss;
                        epoch_quiz_loss += stats.quiz_loss;
                        epoch_quiz_acc += stats.quiz_accuracy * stats.num_quizzes as f64;
                        quiz_count += stats.num_quizzes;
                        total_examples += 1;

                        for (distance, correct) in &stats.quiz_results {
                            self.recall_tracker.record(*distance, *correct);
                        }
                    }
                    Err(e) => {
                        eprintln!("Error in example {}: {:?}", ex_idx, e);
                    }
                }
            }

            if total_examples > 0 {
                let avg_lm = epoch_lm_loss / total_examples as f64;
                let avg_quiz = if quiz_count > 0 {
                    epoch_quiz_loss / total_examples as f64
                } else {
                    0.0
                };
                let avg_acc = if quiz_count > 0 {
                    epoch_quiz_acc / quiz_count as f64
                } else {
                    0.0
                };

                if epoch % self.config.log_interval == 0 || epoch == self.config.epochs - 1 {
                    println!(
                        "Epoch {}/{}: lm_loss={:.4}, quiz_loss={:.4}, quiz_acc={:.2}%, quizzes={}",
                        epoch + 1,
                        self.config.epochs,
                        avg_lm,
                        avg_quiz,
                        avg_acc * 100.0,
                        quiz_count,
                    );
                }

                if epoch % self.config.eval_interval == 0 || epoch == self.config.epochs - 1 {
                    self.print_recall_curve();
                }
            }
        }

        println!("\n=== Training Complete ===");
        self.print_final_stats();
        Ok(())
    }

    fn train_example(&mut self, example: &TrainingExample) -> Result<ExampleStats> {
        let batch_size = 1;
        let mut states = self.model.init_states(batch_size, &self.device)?;
        let quiz_alpha = self.model.config().quiz_alpha;

        let mut total_lm_loss = 0.0;
        let mut total_quiz_loss = 0.0;
        let mut total_quiz_acc = 0.0;
        let mut num_quizzes = 0;
        let mut quiz_results: Vec<(usize, bool)> = Vec::new();

        let quizzes_by_chunk: std::collections::HashMap<
            usize,
            Vec<&crate::data::entity_tracking::Quiz>,
        > = example.quizzes.iter().fold(
            std::collections::HashMap::new(),
            |mut map, (idx, quiz)| {
                map.entry(*idx).or_default().push(quiz);
                map
            },
        );

        for (chunk_idx, chunk_tokens) in example.chunks.iter().enumerate() {
            if chunk_tokens.is_empty() {
                continue;
            }

            let padded = self.pad_or_truncate(chunk_tokens, self.config.chunk_length);
            let token_tensor = Tensor::new(&padded[..], &self.device)?
                .unsqueeze(0)?
                .to_dtype(DType::U32)?;

            let output = self.model.forward(&token_tensor, &states)?;
            states = output.states;

            let target_tensor = Tensor::new(&padded[..], &self.device)?
                .unsqueeze(0)?
                .to_dtype(DType::U32)?;

            let lm_loss = loss::next_token_loss(&output.logits, &target_tensor)?;
            let lm_loss_val = lm_loss.to_scalar::<f32>()? as f64;
            total_lm_loss += lm_loss_val;

            if let Some(chunk_quizzes) = quizzes_by_chunk.get(&chunk_idx) {
                for quiz in chunk_quizzes {
                    let q_padded =
                        self.pad_or_truncate(&quiz.question_tokens, self.config.chunk_length);
                    let q_tensor = Tensor::new(&q_padded[..], &self.device)?
                        .unsqueeze(0)?
                        .to_dtype(DType::U32)?;
                    let q_emb = self.model.embed_tokens(&q_tensor)?;
                    let q_emb_mean = q_emb.mean(1)?;

                    let quiz_logits = self.model.quiz(&states, &q_emb_mean)?;

                    let answer_tensor = Tensor::new(&[quiz.answer_idx as u32], &self.device)?
                        .to_dtype(DType::U32)?;

                    let q_loss = loss::quiz_loss(&quiz_logits, &answer_tensor)?;
                    let q_loss_val = q_loss.to_scalar::<f32>()? as f64;
                    total_quiz_loss += q_loss_val;

                    let q_acc = loss::quiz_accuracy(&quiz_logits, &answer_tensor)?;
                    total_quiz_acc += q_acc;
                    num_quizzes += 1;

                    let correct = q_acc > 0.5;
                    quiz_results.push((quiz.distance, correct));

                    let combined = (&lm_loss + &(q_loss * quiz_alpha)?)?;
                    self.optimizer.backward_step(&combined)?;
                }
            } else {
                self.optimizer.backward_step(&lm_loss)?;
            }
        }

        let num_chunks = example.chunks.len().max(1) as f64;
        Ok(ExampleStats {
            lm_loss: total_lm_loss / num_chunks,
            quiz_loss: if num_quizzes > 0 {
                total_quiz_loss / num_quizzes as f64
            } else {
                0.0
            },
            quiz_accuracy: if num_quizzes > 0 {
                total_quiz_acc / num_quizzes as f64
            } else {
                0.0
            },
            num_quizzes,
            quiz_results,
        })
    }

    fn pad_or_truncate(&self, tokens: &[u32], target_len: usize) -> Vec<u32> {
        if tokens.len() >= target_len {
            tokens[..target_len].to_vec()
        } else {
            let mut padded = tokens.to_vec();
            padded.resize(target_len, 0);
            padded
        }
    }

    fn print_recall_curve(&self) {
        let curve = self.recall_tracker.get_curve();
        if curve.is_empty() {
            return;
        }
        print!("  Recall by distance: ");
        for (dist, acc) in &curve {
            print!("d{}={:.0}% ", dist, acc * 100.0);
        }
        println!();
    }

    fn print_final_stats(&self) {
        println!("\n--- Recall Curve (Context Rot Measurement) ---");
        let curve = self.recall_tracker.get_curve();
        if curve.is_empty() {
            println!("  No quiz data collected.");
            return;
        }
        for (dist, acc) in &curve {
            let bar_len = (acc * 40.0) as usize;
            let bar: String = "█".repeat(bar_len);
            let space: String = " ".repeat(40 - bar_len);
            println!(
                "  Distance {:2}: [{}{}] {:.1}%",
                dist,
                bar,
                space,
                acc * 100.0
            );
        }

        if curve.len() >= 2 {
            let first_acc = curve.first().map(|(_, a)| *a).unwrap_or(0.0);
            let last_acc = curve.last().map(|(_, a)| *a).unwrap_or(0.0);
            let rot = first_acc - last_acc;
            println!(
                "\n  Context rot: {:.1}% drop from distance {} to {}",
                rot * 100.0,
                curve.first().unwrap().0,
                curve.last().unwrap().0,
            );
            if rot < 0.1 {
                println!("  ✓ Minimal context rot detected!");
            } else if rot < 0.3 {
                println!(
                    "  △ Moderate context rot - consider increasing d_state or training longer"
                );
            } else {
                println!("  ✗ Significant context rot - architecture may need changes");
            }
        }
    }

    pub fn save(&self, path: &str) -> Result<()> {
        self.varmap.save(path)?;
        println!("Model saved to {}", path);
        Ok(())
    }
}

struct ExampleStats {
    lm_loss: f64,
    quiz_loss: f64,
    quiz_accuracy: f64,
    num_quizzes: usize,
    quiz_results: Vec<(usize, bool)>,
}
