# Slice 1 notes: RE-USE / SEMamba reference inventory

Source: `nvidia/RE-USE` (HF) at commit `761905064ea1ea882e015e20a64e2e9d28458890`.
Vendored code: `.references/RE-USE/` (no weights, no sample audio). Weights:
`models/reuse/original/model.safetensors` (gitignored). Mamba reference math:
`.references/mamba_ssm/` (`state-spaces/mamba` `v2.2.2`, `8ffd905`).

License: **NSCLv1 (non-commercial)** on the weights.

## Checkpoint summary

- 1416 keys, **9,607,747 params (~9.61M)**.
- Prefixes: `TSMamba` 1320 (30 blocks), `phase_decoder` 34, `mask_decoder` 32,
  `dense_encoder` 30.

## Config (`config.json`)

- `model_cfg`: `num_tfmamba=30`, `hid_feature=64`, `d_state=16`, `d_conv=4`,
  `expand=4`, `input_channel=2`, `output_channel=1`,
  `compress_factor="relu_log1p"`, `beta=2.0`, `mapping=true`,
  `inner_mamba_nlayer=1`.
- `stft_cfg`: `n_fft=320`, `hop_size=40`, `win_size=320`, `sampling_rate=8000`,
  `sfi=true`. (DramaBox scales these to the op rate: `param * op_sr // 8000`,
  made even; denoise-only uses `target_sr == in_sr`.)

## Top-level forward (`models/generator_SEMamba_time_d4.py`)

Inputs `noisy_mag [B,F,T]`, `noisy_pha [B,F,T]`:
1. `[B,F,T] -> [B,1,T,F]` each; concat -> `x [B,2,T,F]`.
2. Anti-error pad: +2 on F then +2 on T -> `[B,2,T+2,F+2]`.
3. `dense_encoder(x)` -> `[B,64,T',F']`.
4. 30x `TFMambaBlock`.
5. `mask_decoder` -> `denoised_mag`; `phase_decoder` -> `denoised_pha` (atan2 of
   two conv branches). Crop back to `[:F,:T]`.
6. `denoised_com = stack(mag*cos(pha), mag*sin(pha), -1)`.

## TFMambaBlock (`models/mamba_block2_SEMamba.py`)

`x [B,C=64,T,F]`:
- reshape `(b*f, t, c)` -> `time_mamba` -> `+ residual`
- reshape `(b*t, f, c)` -> `freq_mamba` -> `+ residual`
- back to `[B,C,T,F]`.

### MambaBlock (bidirectional combine rule — critical for Slices 2/4)

```
out_fw = forward_blocks(x) + x
xf     = flip(x, time)
out_bw = flip(backward_blocks(xf) + xf, time)
out    = output_proj(cat([out_fw, out_bw], dim=-1))   # Linear(2*64 -> 64)
return LayerNorm(out)                                  # nn.LayerNorm(64)
```

Each direction is a standard `mamba_ssm.Mamba` module (no custom kwargs).

## Mamba module layout (per direction, d_model=64)

`d_inner = expand*d_model = 256`, `d_state = 16`, `d_conv = 4`,
`dt_rank = ceil(64/16) = 4`. Keys + shapes:

| key | shape | role |
| --- | --- | --- |
| `in_proj.weight` | `[512,64]` | `x -> (x_in, z)`, each `d_inner=256` (no bias) |
| `conv1d.weight` / `.bias` | `[256,1,4]` / `[256]` | depthwise causal conv on `x_in`, then SiLU |
| `x_proj.weight` | `[36,256]` | `-> (dt[4], B[16], C[16])` (no bias) |
| `dt_proj.weight` / `.bias` | `[256,4]` / `[256]` | `dt_rank -> d_inner`; `delta_bias = dt_proj.bias` |
| `A_log` | `[256,16]` | `A = -exp(A_log)` |
| `D` | `[256]` | skip term |
| `out_proj.weight` | `[64,256]` | `d_inner -> d_model` (no bias) |

Selective scan (from `.references/mamba_ssm/selective_scan_interface.py`,
`selective_scan_ref`): `delta = softplus(dt + delta_bias)`; discretize
`A` (input-independent, per-(d_inner,d_state)) and variable `B`/`C`
(input-dependent, per-(d_state,t)); recurrence
`h_t = exp(delta_t * A) * h_{t-1} + (delta_t * B_t) * x_t`,
`y_t = sum_state(C_t * h_t) + D * x_t`; gate `y = y * SiLU(z)`; then `out_proj`.
`delta_softplus=True`. Slice 2 mirrors THIS function, not an invented recurrence.

## Encoder / decoders (`models/codec_module_time_d4.py`) — detail for Slice 4

- `DenseEncoder`: Conv2d(2->64,(1,1)) + InstanceNorm2d(affine) + PReLU; then
  `DenseBlock(depth=4)` (4x dilated Conv2d/IN/PReLU with growing channels); then
  Conv2d(64->64,(1,3), stride=(4,2)) + IN + PReLU (downsamples F).
- `MagDecoder`: DenseBlock + 2x [`SPConvTranspose2d`(r=2 then r=4) + IN + PReLU]
  + final Conv2d(64->1,(1,1)).
- `PhaseDecoder`: DenseBlock + 2x upsample + `phase_conv_r` + `phase_conv_i`
  (Conv2d 64->1); phase = `atan2(i, r)`.
- `SPConvTranspose2d`: pad + Conv2d to `out*r` then pixel-shuffle along F by `r`.

## STFT front end (`models/stfts.py`) — detail for Slice 3

`mag_phase_stft` / `mag_phase_istft` with `compress_factor="relu_log1p"`.
Read `stfts.py` in Slice 3 for the exact compress/expand and window handling.

## Chunked inference (denoise path)

Reference: `.references/RE-USE/inference_chunk.py` and the DramaBox wrapper
`.references/DramaBox/src/super_resolution.py` (already studied): Hann window,
50% hop overlap-add, per-chunk STFT -> SEMamba -> sweep-artifact filter
(`expm1(relu(amp))`, zero-portion mask) -> iSTFT, normalize by window sum.
Denoise-only: `target_sr == in_sr`, no resample, no BWE.

## Pinned commits (also in `docs/references.md`)

- `nvidia/RE-USE`: `761905064ea1ea882e015e20a64e2e9d28458890`
- `state-spaces/mamba`: `v2.2.2` (`8ffd905c91d207f5c0cc84fc2a2fb748655094f0`)
