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

pub fn combined_loss(
    lm_logits: &Tensor,
    lm_targets: &Tensor,
    quiz_logits: &Tensor,
    quiz_targets: &Tensor,
    alpha: f64,
) -> Result<(Tensor, Tensor, Tensor)> {
    let l_next_token = next_token_loss(lm_logits, lm_targets)?;
    let l_quiz = quiz_loss(quiz_logits, quiz_targets)?;
    let total = (&l_next_token + &(l_quiz.clone() * alpha)?)?;
    Ok((total, l_next_token, l_quiz))
}

pub fn quiz_accuracy(quiz_logits: &Tensor, quiz_targets: &Tensor) -> Result<f64> {
    let predictions = quiz_logits.argmax(1)?.to_dtype(DType::U32)?;
    let targets_u32 = quiz_targets.to_dtype(DType::U32)?;
    let correct = predictions.eq(&targets_u32)?.to_dtype(DType::F32)?;
    let acc = correct.mean_all()?.to_scalar::<f32>()?;
    Ok(acc as f64)
}
