use candle_core::{DType, Result, Tensor};
use candle_nn::loss::cross_entropy;

pub fn next_token_loss(logits: &Tensor, targets: &Tensor) -> Result<Tensor> {
    let (batch, seq_len, vocab_size) = logits.dims3()?;
    let shift_logits = logits.narrow(1, 0, seq_len - 1)?;
    let shift_targets = targets.narrow(1, 1, seq_len - 1)?;

    let flat_logits = shift_logits.reshape((batch * (seq_len - 1), vocab_size))?;
    let flat_targets = shift_targets.reshape(batch * (seq_len - 1))?;

    cross_entropy(&flat_logits, &flat_targets)
}

pub fn quiz_loss(quiz_logits: &Tensor, quiz_targets: &Tensor) -> Result<Tensor> {
    cross_entropy(quiz_logits, quiz_targets)
}

pub fn quiz_accuracy(quiz_logits: &Tensor, quiz_targets: &Tensor) -> Result<f64> {
    let predictions = quiz_logits.argmax(1)?.to_dtype(DType::U32)?;
    let targets_u32 = quiz_targets.to_dtype(DType::U32)?;
    let correct = predictions.eq(&targets_u32)?.to_dtype(DType::F32)?;
    let acc = correct.mean_all()?.to_scalar::<f32>()?;
    Ok(acc as f64)
}

pub struct DynamicAlpha {
    ema_lm: f64,
    ema_quiz: f64,
    decay: f64,
    base_alpha: f64,
    min_alpha: f64,
    max_alpha: f64,
}

impl DynamicAlpha {
    pub fn new(base_alpha: f64) -> Self {
        DynamicAlpha {
            ema_lm: 1.0,
            ema_quiz: 1.0,
            decay: 0.95,
            base_alpha,
            min_alpha: 0.1,
            max_alpha: 2.0,
        }
    }

    pub fn update(&mut self, lm_loss: f64, quiz_loss: f64) {
        let lm_clamped = lm_loss.max(1e-8);
        let quiz_clamped = quiz_loss.max(1e-8);
        self.ema_lm = self.decay * self.ema_lm + (1.0 - self.decay) * lm_clamped;
        self.ema_quiz = self.decay * self.ema_quiz + (1.0 - self.decay) * quiz_clamped;
    }

    pub fn get_alpha(&self) -> f64 {
        let ratio = self.ema_lm / self.ema_quiz.max(1e-8);
        let alpha = self.base_alpha * ratio;
        alpha.clamp(self.min_alpha, self.max_alpha)
    }
}
