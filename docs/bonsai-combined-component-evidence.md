# Bonsai combined component evidence

Canonical report:

```text
docs/ci/bonsai-combined-component-probe-latest.json
```

Supplemental manifest:

```text
docs/bonsai-combined-component-verification-manifest.json
```

## Verified in the current combined report

The combined component report completed with:

```text
status: passed
ci_conclusion: success
claim_promotable_to_manifest: true
run_transformer_load: false
source_report_commit: bc59ac59f84f7d86dfb1e1843982f289776ec708
```

The following stage-level claims are verified and promotable:

```text
bonsai_tokenizer_execution_verified
bonsai_text_encoder_execution_verified
bonsai_scheduler_step_execution_verified
bonsai_vae_decoder_execution_verified
bonsai_transformer_config_boundary_verified_not_runtime_execution
```

## Tokenizer execution

```text
stage: tokenizer_execution
component: tokenizer
class: Qwen2Tokenizer
prompt: a small bonsai tree in a ceramic pot
input_ids_shape: [1, 9]
input_ids_sha256_int64_le: a3c323ac42dc29072a5896a724ab160c1192d596bd617128113630855b763b68
attention_mask_shape: [1, 9]
attention_mask_sha256_int64_le: df070c0849900928e1e85833cce32020c89d2303a9e960510c3c3b79746448b7
```

## Text encoder execution

```text
stage: text_encoder_execution
component: text_encoder
class: Qwen3Model
hidden_state_shape: [1, 9, 2560]
hidden_state_dtype: float32
hidden_state_sha256_float32_le: 6d4ce391fc0cb7664bc52303e8ec0f676023e124327983de943147be41d817b0
```

## Scheduler execution

```text
stage: scheduler_execution
component: scheduler
class: FlowMatchEulerDiscreteScheduler
set_timesteps_kwargs: {"mu": 0.0}
timesteps: [1000.0, 1.0]
prev_sample_shape: [1, 4, 8, 8]
prev_sample_dtype: float32
prev_sample_sha256_float32_le: e067c3fa7c505915ba6536e0f7a4ff1265605484327484df70785a51b8bbf281
```

## VAE decoder execution

```text
stage: vae_execution
component: vae_decoder
class: AutoencoderKL
input_shape: [1, 32, 8, 8]
decoded_sample_shape: [1, 3, 64, 64]
decoded_sample_dtype: float32
decoded_sample_sha256_float32_le: 6107e6c616b578184ceac14b086f5fe9fe32569113184f0727d498b3b96004dc
```

## Transformer config boundary

```text
stage: transformer_config_load
component: transformer
load_kind: config_only
class: Flux2Transformer2DModel
num_layers: 5
num_single_layers: 20
in_channels: 128
joint_attention_dim: 7680
num_attention_heads: 24
attention_head_dim: 128
```

This is not transformer runtime execution. It is a transformer configuration boundary claim only.

## Not verified yet

The current combined report did not verify:

```text
real transformer weight load
real transformer ONNX execution
full Bonsai pipeline composition
prompt-to-image generation
single monolithic multi-block ONNX
```

Full pipeline composition is now blocked only by transformer weight load in the current combined report. Scheduler execution is no longer blocking the full pipeline composition boundary.

## Evidence boundary

This evidence advances component execution coverage, but it still does not prove full Bonsai ONNX pipeline execution or prompt-to-image generation. Those require separate successful stage-level claims.
