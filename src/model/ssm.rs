use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{linear, Linear, Module, VarBuilder};

pub struct SsmConfig {
    pub d_model: usize,
    pub d_state: usize,
    pub d_inner: usize,
    pub n_layers: usize,
    pub vocab_size: usize,
}

impl SsmConfig {
    pub fn tiny() -> Self {
        SsmConfig {
            d_model: 64,
            d_state: 32,
            d_inner: 128,
            n_layers: 2,
            vocab_size: 256,
        }
    }

    pub fn small() -> Self {
        SsmConfig {
            d_model: 256,
            d_state: 128,
            d_inner: 512,
            n_layers: 4,
            vocab_size: 512,
        }
    }
}

pub struct SelectiveScanLayer {
    proj_in: Linear,
    proj_delta: Linear,
    proj_b: Linear,
    proj_c: Linear,
    proj_out: Linear,
    d_state: usize,
    d_inner: usize,
}

impl SelectiveScanLayer {
    pub fn new(d_model: usize, d_state: usize, d_inner: usize, vb: VarBuilder) -> Result<Self> {
        let proj_in = linear(d_model, d_inner, vb.pp("proj_in"))?;
        let proj_delta = linear(d_inner, d_inner, vb.pp("proj_delta"))?;
        let proj_b = linear(d_inner, d_state, vb.pp("proj_b"))?;
        let proj_c = linear(d_inner, d_state, vb.pp("proj_c"))?;
        let proj_out = linear(d_inner, d_model, vb.pp("proj_out"))?;
        Ok(Self {
            proj_in,
            proj_delta,
            proj_b,
            proj_c,
            proj_out,
            d_state,
            d_inner,
        })
    }

    pub fn forward_with_state(&self, x: &Tensor, prev_state: &Tensor) -> Result<(Tensor, Tensor)> {
        let (batch, seq_len, _d_model) = x.dims3()?;

        let z = self.proj_in.forward(x)?;
        let z = z.silu()?;

        let delta = self.proj_delta.forward(&z)?;
        let delta = delta.clamp(-20.0, 20.0)?;
        let delta = (delta.exp()? + 1.0)?.log()?;

        let b = self.proj_b.forward(&z)?;
        let c = self.proj_c.forward(&z)?;

        let mut state = prev_state.clone();
        let mut outputs = Vec::with_capacity(seq_len);

        for t in 0..seq_len {
            let z_t = z.narrow(1, t, 1)?.squeeze(1)?;
            let delta_t = delta.narrow(1, t, 1)?.squeeze(1)?;
            let b_t = b.narrow(1, t, 1)?.squeeze(1)?;
            let c_t = c.narrow(1, t, 1)?.squeeze(1)?;

            let decay = delta_t.neg()?.exp()?;

            let decay_expanded = decay
                .unsqueeze(2)?
                .expand((batch, self.d_inner, self.d_state))?;
            let state_decayed = state.mul(&decay_expanded)?;

            let z_expanded = z_t
                .unsqueeze(2)?
                .expand((batch, self.d_inner, self.d_state))?;
            let b_expanded = b_t
                .unsqueeze(1)?
                .expand((batch, self.d_inner, self.d_state))?;
            let delta_expanded =
                delta_t
                    .unsqueeze(2)?
                    .expand((batch, self.d_inner, self.d_state))?;
            let input_contrib = z_expanded.mul(&b_expanded)?.mul(&delta_expanded)?;

            state = (state_decayed + input_contrib)?;

            let y_t = (&state
                * &c_t
                    .unsqueeze(1)?
                    .expand((batch, self.d_inner, self.d_state))?)?
                .sum(2)?;

            outputs.push(y_t.unsqueeze(1)?);
        }

        let output = Tensor::cat(&outputs, 1)?;
        let output = self.proj_out.forward(&output)?;
        Ok((output, state))
    }

    pub fn init_state(&self, batch_size: usize, device: &Device) -> Result<Tensor> {
        Tensor::zeros((batch_size, self.d_inner, self.d_state), DType::F32, device)
    }
}

pub struct SsmStack {
    layers: Vec<SelectiveScanLayer>,
    pub config: SsmConfig,
}

impl SsmStack {
    pub fn new(config: SsmConfig, vb: VarBuilder) -> Result<Self> {
        let mut layers = Vec::with_capacity(config.n_layers);
        for i in 0..config.n_layers {
            let layer = SelectiveScanLayer::new(
                config.d_model,
                config.d_state,
                config.d_inner,
                vb.pp(format!("layer_{}", i)),
            )?;
            layers.push(layer);
        }
        Ok(Self { layers, config })
    }

    pub fn forward_with_states(
        &self,
        x: &Tensor,
        prev_states: &[Tensor],
    ) -> Result<(Tensor, Vec<Tensor>)> {
        let mut current = x.clone();
        let mut new_states = Vec::with_capacity(self.layers.len());
        for (i, layer) in self.layers.iter().enumerate() {
            let (output, state) = layer.forward_with_state(&current, &prev_states[i])?;
            current = (&current + &output)?;
            new_states.push(state);
        }
        Ok((current, new_states))
    }

    pub fn init_states(&self, batch_size: usize, device: &Device) -> Result<Vec<Tensor>> {
        self.layers
            .iter()
            .map(|layer| layer.init_state(batch_size, device))
            .collect()
    }
}
