use candle_core::{Device, Result, Tensor};
use candle_nn::{embedding, linear, Embedding, Linear, Module, VarBuilder};

use super::quiz_head::QuizHead;
use super::ssm::{SsmConfig, SsmStack};

pub struct WorldStateModelConfig {
    pub ssm: SsmConfig,
    pub num_quiz_answers: usize,
    pub quiz_alpha: f64,
}

impl WorldStateModelConfig {
    pub fn tiny() -> Self {
        WorldStateModelConfig {
            ssm: SsmConfig::tiny(),
            num_quiz_answers: 16,
            quiz_alpha: 0.5,
        }
    }

    pub fn small() -> Self {
        WorldStateModelConfig {
            ssm: SsmConfig::small(),
            num_quiz_answers: 32,
            quiz_alpha: 0.5,
        }
    }
}

pub struct WorldStateModel {
    token_emb: Embedding,
    ssm: SsmStack,
    lm_head: Linear,
    quiz_head: QuizHead,
    config: WorldStateModelConfig,
}

impl WorldStateModel {
    pub fn new(config: WorldStateModelConfig, vb: VarBuilder) -> Result<Self> {
        let token_emb = embedding(
            config.ssm.vocab_size,
            config.ssm.d_model,
            vb.pp("token_emb"),
        )?;
        let ssm = SsmStack::new(
            SsmConfig {
                d_model: config.ssm.d_model,
                d_state: config.ssm.d_state,
                d_inner: config.ssm.d_inner,
                n_layers: config.ssm.n_layers,
                vocab_size: config.ssm.vocab_size,
            },
            vb.pp("ssm"),
        )?;
        let lm_head = linear(config.ssm.d_model, config.ssm.vocab_size, vb.pp("lm_head"))?;

        let d_state_per_layer = config.ssm.d_inner * config.ssm.d_state;
        let quiz_head = QuizHead::new(
            d_state_per_layer,
            config.ssm.d_model,
            config.ssm.n_layers,
            config.num_quiz_answers,
            vb.pp("quiz_head"),
        )?;

        Ok(Self {
            token_emb,
            ssm,
            lm_head,
            quiz_head,
            config,
        })
    }

    pub fn forward(&self, token_ids: &Tensor, prev_states: &[Tensor]) -> Result<ForwardOutput> {
        let embedded = self.token_emb.forward(token_ids)?;

        let (hidden, new_states) = self.ssm.forward_with_states(&embedded, prev_states)?;

        let logits = self.lm_head.forward(&hidden)?;

        Ok(ForwardOutput {
            logits,
            hidden,
            states: new_states,
        })
    }

    pub fn quiz(&self, states: &[Tensor], question_emb: &Tensor) -> Result<Tensor> {
        self.quiz_head.forward(states, question_emb)
    }

    pub fn embed_tokens(&self, token_ids: &Tensor) -> Result<Tensor> {
        self.token_emb.forward(token_ids)
    }

    pub fn init_states(&self, batch_size: usize, device: &Device) -> Result<Vec<Tensor>> {
        self.ssm.init_states(batch_size, device)
    }

    pub fn config(&self) -> &WorldStateModelConfig {
        &self.config
    }

    pub fn d_model(&self) -> usize {
        self.config.ssm.d_model
    }

    pub fn vocab_size(&self) -> usize {
        self.config.ssm.vocab_size
    }
}

pub struct ForwardOutput {
    pub logits: Tensor,
    pub hidden: Tensor,
    pub states: Vec<Tensor>,
}
