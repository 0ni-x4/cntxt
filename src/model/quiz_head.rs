use candle_core::{Result, Tensor};
use candle_nn::{linear, Linear, Module, VarBuilder};

pub struct QuizHead {
    state_proj: Linear,
    question_proj: Linear,
    output_proj: Linear,
    d_quiz: usize,
}

impl QuizHead {
    pub fn new(
        d_state_total: usize,
        d_model: usize,
        num_answers: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let d_quiz = d_model;
        let state_proj = linear(d_state_total, d_quiz, vb.pp("state_proj"))?;
        let question_proj = linear(d_model, d_quiz, vb.pp("question_proj"))?;
        let output_proj = linear(d_quiz, num_answers, vb.pp("output_proj"))?;
        Ok(Self {
            state_proj,
            question_proj,
            output_proj,
            d_quiz,
        })
    }

    pub fn forward(&self, flat_state: &Tensor, question_emb: &Tensor) -> Result<Tensor> {
        let state_h = self.state_proj.forward(flat_state)?.tanh()?;
        let question_h = self.question_proj.forward(question_emb)?.tanh()?;
        let combined = (state_h * question_h)?;
        self.output_proj.forward(&combined)
    }

    pub fn d_quiz(&self) -> usize {
        self.d_quiz
    }
}
