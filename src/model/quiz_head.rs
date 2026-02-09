use candle_core::{Result, Tensor};
use candle_nn::{linear, Linear, Module, VarBuilder};

pub struct QuizHead {
    layer_proj: Linear,
    query_proj: Linear,
    key_proj: Linear,
    value_proj: Linear,
    output_proj: Linear,
    d_quiz: usize,
    n_layers: usize,
}

impl QuizHead {
    pub fn new(
        d_state_per_layer: usize,
        d_model: usize,
        n_layers: usize,
        num_answers: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let d_quiz = d_model;

        let layer_proj = linear(d_state_per_layer, d_quiz, vb.pp("layer_proj"))?;

        let query_proj = linear(d_model, d_quiz, vb.pp("query_proj"))?;

        let key_proj = linear(d_quiz, d_quiz, vb.pp("key_proj"))?;

        let value_proj = linear(d_quiz, d_quiz, vb.pp("value_proj"))?;

        let output_proj = linear(d_quiz, num_answers, vb.pp("output_proj"))?;
        Ok(Self {
            layer_proj,
            query_proj,
            key_proj,
            value_proj,
            output_proj,
            d_quiz,
            n_layers,
        })
    }

    pub fn forward(&self, layer_states: &[Tensor], question_emb: &Tensor) -> Result<Tensor> {
        let _batch = question_emb.dims()[0];

        let projected: Vec<Tensor> = layer_states
            .iter()
            .map(|s| {
                let (b, d_inner, d_state) = s.dims3().unwrap();
                let flat = s.reshape((b, d_inner * d_state)).unwrap();
                self.layer_proj.forward(&flat).unwrap()
            })
            .collect();

        let stacked = Tensor::stack(&projected, 1)?;

        let query = self.query_proj.forward(question_emb)?;
        let query = query.unsqueeze(1)?;

        let keys = self.key_proj.forward(&stacked)?;
        let values = self.value_proj.forward(&stacked)?;

        let scale = (self.d_quiz as f64).sqrt();
        let attn_scores = query.matmul(&keys.transpose(1, 2)?)?;
        let attn_scores = (attn_scores * (1.0 / scale))?;

        let attn_weights = candle_nn::ops::softmax(&attn_scores, 2)?;

        let attended = attn_weights.matmul(&values)?;
        let attended = attended.squeeze(1)?;

        let question_h = self.query_proj.forward(question_emb)?.tanh()?;
        let combined = (attended * question_h)?;

        self.output_proj.forward(&combined)
    }

    pub fn d_quiz(&self) -> usize {
        self.d_quiz
    }

    pub fn n_layers(&self) -> usize {
        self.n_layers
    }
}
