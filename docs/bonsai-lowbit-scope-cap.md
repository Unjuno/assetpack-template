# Bonsai low-bit ONNX scope cap

Canonical evidence source:

```text
docs/bonsai-lowbit-verification-manifest.json
```

This document closes the current low-bit critical-path scope after the verified ten-by-two input-boundary report.

## Closed scope

The following low-bit critical-path evidence boundary is complete for this repository stage:

```text
attention to_out path
single block modulated attention to_out residual
two single blocks
four blocks via 2x2 segmented ONNX
eight blocks via 4x2 segmented ONNX
sixteen blocks via 8x2 segmented ONNX
all ten pair segments for blocks 0-19
twenty blocks via ten-by-two chained segmented export probe
twenty blocks via split persistent two-block ONNX segment artifacts
segment-to-segment chain-state handoff report
hidden/temb-only external ONNX input-boundary report
```

The strongest verified claim for this scope is:

```text
twenty single-stream low-bit critical-path blocks are validated as a reusable ten-segment ONNX Runtime CPU chain, with persisted ONNX segment artifacts, segment-to-segment hidden-state handoff evidence, and an input-boundary report showing external ONNX inputs are limited to hidden and temb
```

## Evidence artifacts

| Evidence | Run | Artifact | SHA-256 |
|---|---:|---:|---|
| all ten pair segments | 27366197393 | 7572369505 | `653af8965e7181b9d7063239e438f21e08de7284337258d1f4ee98e30a4ada8a` |
| ten-by-two chained export probe | 27368101404 | 7573239516 | `7e5a75327d81e576512546b3e0c611ae3bbf464c5cdbb1043831fb67d63b99eb` |
| split persistent ONNX segment artifacts | 27370454697 | 7574146614 | `a09ce6303c70b0fd95556803cfd45b3765db1a216ad58f70fdc2e8871d758ab3` |
| chain-state handoff report | 27372009301 | 7574781425 | `5fdc2278ced66cbaed794943d3975c18dfbeda1cd849328e29a0a7b3cfabd5fb` |
| input-boundary report | 27399310016 | 7584863386 | `321133159edaa0136412c09efc04e59f61e53da502cc27d737282b565b8665ac` |

## Explicit non-claims

The closed scope does not verify:

- full Bonsai ONNX pipeline execution;
- real transformer block ONNX execution;
- prompt-to-image generation;
- text encoder execution;
- scheduler execution;
- VAE execution as part of this low-bit chain;
- image latent production or consumption;
- single monolithic 20-block ONNX execution.

These remain forbidden unless a future manifest entry verifies them.

## Why the scope is capped here

The input-boundary report establishes that the current low-bit ONNX chain is not a pipeline input boundary. Its external ONNX inputs are synthetic/reference `hidden` and `temb` tensors, not prompt tokens, encoded text, image latents, scheduler state, or VAE tensors.

Therefore the current work product is best treated as a completed reusable critical-path ONNX chain, not as an incomplete full-pipeline artifact.

## Required rule for future expansion

Any future expansion must start a new manifest key and a new evidence boundary before README or docs may promote the claim.

Acceptable next scopes include:

1. real architecture double-block / single-block integration evidence;
2. text-encoder or prompt-token input boundary evidence;
3. scheduler / latent boundary evidence;
4. VAE boundary evidence;
5. composed full-pipeline evidence.

Each next scope must state its own forbidden claims before execution.
