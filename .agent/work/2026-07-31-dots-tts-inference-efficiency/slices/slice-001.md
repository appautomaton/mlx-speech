# Slice 1 Canonical Comparison Contract

This linked slice detail is the durable input to the final performance and quality gates. Gitignored JSON under `outputs/` is diagnostic output only; neither final verdict may require it after this slice commits.

## Contract behavior

- `profile_dots_tts_inference.py` and `dots_tts_quant_gate.py` share one stdlib-only reader/writer for the single JSON block under `## Canonical comparison data`.
- A freeze operation replaces only its owned `performance` or `quality` member and preserves the other member plus this Markdown prose.
- Readers fail closed on a missing or duplicate data block, a non-complete status, absent cases or trials, incompatible schema, changed workload settings, or mismatched host, model-artifact, reference-audio, manifest, corpus, or ASR identities.
- The quality runner accepts repeated exact `--case` keys for the focused Slice 3 precision decision and recomputes its comparison over only that declared subset; no `--case` arguments retain the complete final matrix.
- The performance member retains the starting commit plus working-tree digest, host and MLX identity, model/reference digests, fixed workload, warmup and compilation data, and every measured batch/stream MF/SOAR trial with output-health, stage, memory, TTFC, duration, RTF, and total-time fields.
- The quality member retains the source-report digest, manifest/corpus/ASR/artifact identities, thresholds, and every MF/SOAR × base/int8 × continuation/speaker-only comparison metric needed to recompute WER and speaker-cosine regression.
- Slice 1 is the only slice that populates this block. Later slices consume it read-only. Execution records the raw diagnostic report digests beside the completed block.

## Canonical comparison data

```json
{
  "schema_version": 1,
  "status": "complete",
  "performance": {
    "report_sha256": "5798bdb624bcd025f22f691158af7371ad53aff32c066430f53ae18fc7d556d3",
    "baseline": {
      "schema_version": 1,
      "host": {
        "platform": "macOS-26.5.2-arm64-arm-64bit-Mach-O",
        "machine": "arm64",
        "processor": "arm"
      },
      "mlx_version": "0.31.1",
      "source": {
        "commit": "bc852f8cb1793c098ce4b2893080bc0ce37654fd",
        "branch": "main",
        "source_tree_sha256": "14675f66b5b5ea2072c9674f02f6dbe7e5d2ee67ca78757ef230b03f3333c995",
        "tracked_diff_sha256": "6e4fbb60f21a5fae1d988113207673bb11d4ec035358aebbdddc9e246c7546cd",
        "file_count": 538,
        "paths": [
          "src",
          "scripts",
          "tests",
          "pyproject.toml",
          "uv.lock"
        ]
      },
      "command": "scripts/eval/profile_dots_tts_inference.py --model-root models/dots_tts --reference-audio outputs/source/hank_hill_ref.wav --variants mf soar --paths batch stream --warmup-runs 1 --runs 3 --max-audio-patches 128 --seed 42 --eos-threshold 1.0 --memory-limit-gib 30 --output outputs/dots_tts/inference_efficiency/baseline.json --freeze-comparison-contract .agent/work/2026-07-31-dots-tts-inference-efficiency/slices/slice-001.md",
      "config": {
        "model_root": "models/dots_tts",
        "artifact_class": "base",
        "text": "Technology is most useful when it gives people more time to think, create, and care for one another.",
        "seed": 42,
        "max_audio_patches": 128,
        "eos_threshold": 1.0,
        "warmup_runs": 1,
        "runs": 3,
        "variants": [
          "mf",
          "soar"
        ],
        "paths": [
          "batch",
          "stream"
        ],
        "memory_limit_gib": 30.0,
        "solver_steps": "artifact_default"
      },
      "reference": {
        "path": "outputs/source/hank_hill_ref.wav",
        "bytes": 432078,
        "sha256": "73b90d2c528f2b120c8cba83e2aa1a99627a7b923ca5dc0ab0d9ff097485bc7d"
      },
      "artifacts": {
        "mf": {
          "path": "models/dots_tts/mf/mlx-base",
          "artifact_class": "base",
          "source": {
            "manifest_sha256": "c5b8638c5083a06b052a2472685ae5933a8f6b2bcbe96cc00cd11624bc12a6af",
            "repo_id": "rednote-hilab/dots.tts-mf",
            "resolved_repo_id": "dots-studio/dots.tts-mf",
            "revision": "25c53fb462e57087e52237daa5ea30df1c5cc328"
          },
          "quantization": null,
          "digest": "b671893535eeb684edf93abdd525a4966ca965df5cc45d43c371e449342a1eb0",
          "files": {
            "core.safetensors": {
              "bytes": 4398911066,
              "sha256": "bbfca574cc825d7256783f20a4c1ea1c483f49c4ca6b163468a1ea66429d217e"
            },
            "vocoder.safetensors": {
              "bytes": 451779177,
              "sha256": "e5b229d076aec43c20f7f259d2a2bbc75a4384b34adf7a669eefc69bff25e1c0"
            },
            "speaker.safetensors": {
              "bytes": 29122718,
              "sha256": "105356b29dfe40c986412db337c252ab2d227703f433de64b525078de6ca5858"
            },
            "latent_stats.safetensors": {
              "bytes": 1160,
              "sha256": "370ea7fe3e54c06336ff050eaf14f9554417f39b002b2dad5228f67c86c7260f"
            },
            "mlx_config.json": {
              "bytes": 800,
              "sha256": "908698c6aca8f2be396088894585f4ba34e43c863b02a9644d7374ecb559db80"
            }
          }
        },
        "soar": {
          "path": "models/dots_tts/soar/mlx-base",
          "artifact_class": "base",
          "source": {
            "manifest_sha256": "c5b8638c5083a06b052a2472685ae5933a8f6b2bcbe96cc00cd11624bc12a6af",
            "repo_id": "rednote-hilab/dots.tts-soar",
            "resolved_repo_id": "dots-studio/dots.tts-soar",
            "revision": "e3520f75254d0020a0406db31c51a79d00d22d55"
          },
          "quantization": null,
          "digest": "abb8b62eedd492b01447cd9601a678fb96a2ba83795a1745d40a934e30538e1e",
          "files": {
            "core.safetensors": {
              "bytes": 4396285089,
              "sha256": "d16bce6f079ac3b3162ff25bc7ee92d70e8724fccd17791e15bf5cc6b817d43a"
            },
            "vocoder.safetensors": {
              "bytes": 451779177,
              "sha256": "e5b229d076aec43c20f7f259d2a2bbc75a4384b34adf7a669eefc69bff25e1c0"
            },
            "speaker.safetensors": {
              "bytes": 29122718,
              "sha256": "105356b29dfe40c986412db337c252ab2d227703f433de64b525078de6ca5858"
            },
            "latent_stats.safetensors": {
              "bytes": 1160,
              "sha256": "370ea7fe3e54c06336ff050eaf14f9554417f39b002b2dad5228f67c86c7260f"
            },
            "mlx_config.json": {
              "bytes": 811,
              "sha256": "0d9f32141aa66172bbee722db679cfeae07bc46714e88b244bb7e8b5fc9c6507"
            }
          }
        }
      },
      "cases": [
        {
          "variant": "mf",
          "path": "batch",
          "warmup": {
            "runs": [
              {
                "path": "batch",
                "seed": 42,
                "patch_count": 128,
                "chunk_count": 1,
                "waveform_samples": 983040,
                "sample_rate": 48000,
                "output_seconds": 20.48,
                "total_seconds": 51.93470849993173,
                "first_output_seconds": 51.934702083002776,
                "completion_after_first_output_seconds": 6.416928954422474e-06,
                "rtf": 2.535874438473229,
                "stage_seconds": {
                  "acoustic": 7.620913871913217,
                  "decoder": 38.35864354111254,
                  "prefill": 0.0021092499373480678,
                  "prompt": 2.21162366704084,
                  "residual": 3.7414181699277833
                },
                "baseline_memory_bytes": 4880682060,
                "peak_memory_bytes": 7867413026,
                "incremental_peak_bytes": 2986730966,
                "output_health": {
                  "finite": true,
                  "non_silent": true,
                  "peak_absolute": 0.7693974375724792
                },
                "run": 1
              }
            ],
            "total_seconds": 51.93470849993173,
            "explicit_compile_seconds": 0.0,
            "prompt_cache_cleared_after": false
          },
          "medians": {
            "total_seconds": 52.599544457974844,
            "first_output_seconds": 52.599541208008304,
            "output_seconds": 20.48,
            "rtf": 2.568337131737053,
            "peak_memory_bytes": 7867348514,
            "stage_seconds": {
              "acoustic": 7.92801595677156,
              "decoder": 38.659477836685255,
              "prefill": 0.0013432500418275595,
              "prompt": 2.199191708001308,
              "residual": 3.8115267475368455
            }
          },
          "trials": [
            {
              "path": "batch",
              "seed": 42,
              "patch_count": 128,
              "chunk_count": 1,
              "waveform_samples": 983040,
              "sample_rate": 48000,
              "output_seconds": 20.48,
              "total_seconds": 52.37244379206095,
              "first_output_seconds": 52.37243870808743,
              "completion_after_first_output_seconds": 5.08397351950407e-06,
              "rtf": 2.557248232034226,
              "stage_seconds": {
                "acoustic": 7.855068453820422,
                "decoder": 38.56421062606387,
                "prefill": 0.001354833017103374,
                "prompt": 2.1861508330330253,
                "residual": 3.7656590461265296
              },
              "baseline_memory_bytes": 4880682580,
              "peak_memory_bytes": 7867348514,
              "incremental_peak_bytes": 2986665934,
              "output_health": {
                "finite": true,
                "non_silent": true,
                "peak_absolute": 0.7693974375724792
              },
              "run": 1,
              "reference_cache": "cold"
            },
            {
              "path": "batch",
              "seed": 42,
              "patch_count": 128,
              "chunk_count": 1,
              "waveform_samples": 983040,
              "sample_rate": 48000,
              "output_seconds": 20.48,
              "total_seconds": 52.599544457974844,
              "first_output_seconds": 52.599541208008304,
              "completion_after_first_output_seconds": 3.2499665394425392e-06,
              "rtf": 2.568337131737053,
              "stage_seconds": {
                "acoustic": 7.92801595677156,
                "decoder": 38.659477836685255,
                "prefill": 0.0013322089798748493,
                "prompt": 2.199191708001308,
                "residual": 3.8115267475368455
              },
              "baseline_memory_bytes": 4880682580,
              "peak_memory_bytes": 7867347234,
              "incremental_peak_bytes": 2986664654,
              "output_health": {
                "finite": true,
                "non_silent": true,
                "peak_absolute": 0.7693974375724792
              },
              "run": 2,
              "reference_cache": "warm"
            },
            {
              "path": "batch",
              "seed": 42,
              "patch_count": 128,
              "chunk_count": 1,
              "waveform_samples": 983040,
              "sample_rate": 48000,
              "output_seconds": 20.48,
              "total_seconds": 54.433969374978915,
              "first_output_seconds": 54.43396649998613,
              "completion_after_first_output_seconds": 2.874992787837982e-06,
              "rtf": 2.6579086608876423,
              "stage_seconds": {
                "acoustic": 8.081194343278185,
                "decoder": 40.13148145994637,
                "prefill": 0.0013432500418275595,
                "prompt": 2.199974333983846,
                "residual": 4.019975987728685
              },
              "baseline_memory_bytes": 4880682580,
              "peak_memory_bytes": 7867347746,
              "incremental_peak_bytes": 2986665166,
              "output_health": {
                "finite": true,
                "non_silent": true,
                "peak_absolute": 0.7693974375724792
              },
              "run": 3,
              "reference_cache": "warm"
            }
          ]
        },
        {
          "variant": "mf",
          "path": "stream",
          "warmup": {
            "runs": [
              {
                "path": "stream",
                "seed": 42,
                "patch_count": 128,
                "chunk_count": 35,
                "waveform_samples": 983040,
                "sample_rate": 48000,
                "output_seconds": 20.48,
                "total_seconds": 53.59648295794614,
                "first_output_seconds": 2.900980458012782,
                "completion_after_first_output_seconds": 50.695502499933355,
                "rtf": 2.6170157694309637,
                "stage_seconds": {
                  "acoustic": 7.90241812390741,
                  "decoder": 39.47823050082661,
                  "prefill": 0.00132395897526294,
                  "prompt": 2.197338959085755,
                  "residual": 4.017171415151097
                },
                "baseline_memory_bytes": 4880682580,
                "peak_memory_bytes": 7867365410,
                "incremental_peak_bytes": 2986682830,
                "output_health": {
                  "finite": true,
                  "non_silent": true,
                  "peak_absolute": 0.7693974375724792
                },
                "run": 1
              }
            ],
            "total_seconds": 53.59648295794614,
            "explicit_compile_seconds": 0.0,
            "prompt_cache_cleared_after": false
          },
          "medians": {
            "total_seconds": 54.895407916978,
            "first_output_seconds": 2.986880208016373,
            "output_seconds": 20.48,
            "rtf": 2.6804398396961915,
            "peak_memory_bytes": 7903981434,
            "stage_seconds": {
              "acoustic": 8.005405247211456,
              "decoder": 40.723702835384756,
              "prefill": 0.001333459047600627,
              "prompt": 2.190097542013973,
              "residual": 3.975008749170229
            }
          },
          "trials": [
            {
              "path": "stream",
              "seed": 42,
              "patch_count": 128,
              "chunk_count": 35,
              "waveform_samples": 983040,
              "sample_rate": 48000,
              "output_seconds": 20.48,
              "total_seconds": 53.97493783396203,
              "first_output_seconds": 2.8672788749681786,
              "completion_after_first_output_seconds": 51.10765895899385,
              "rtf": 2.6354950114239273,
              "stage_seconds": {
                "acoustic": 8.005405247211456,
                "decoder": 39.88822591654025,
                "prefill": 0.001438041916117072,
                "prompt": 2.18828925001435,
                "residual": 3.8915793782798573
              },
              "baseline_memory_bytes": 4880682580,
              "peak_memory_bytes": 7867380514,
              "incremental_peak_bytes": 2986697934,
              "output_health": {
                "finite": true,
                "non_silent": true,
                "peak_absolute": 0.7693974375724792
              },
              "run": 1,
              "reference_cache": "cold"
            },
            {
              "path": "stream",
              "seed": 42,
              "patch_count": 128,
              "chunk_count": 35,
              "waveform_samples": 983040,
              "sample_rate": 48000,
              "output_seconds": 20.48,
              "total_seconds": 54.895407916978,
              "first_output_seconds": 3.1129659169819206,
              "completion_after_first_output_seconds": 51.78244199999608,
              "rtf": 2.6804398396961915,
              "stage_seconds": {
                "acoustic": 7.971716288942844,
                "decoder": 40.723702835384756,
                "prefill": 0.0013227910967543721,
                "prompt": 2.2094347500242293,
                "residual": 3.989231251529418
              },
              "baseline_memory_bytes": 4880682580,
              "peak_memory_bytes": 7903981434,
              "incremental_peak_bytes": 3023298854,
              "output_health": {
                "finite": true,
                "non_silent": true,
                "peak_absolute": 0.7693974375724792
              },
              "run": 2,
              "reference_cache": "warm"
            },
            {
              "path": "stream",
              "seed": 42,
              "patch_count": 128,
              "chunk_count": 35,
              "waveform_samples": 983040,
              "sample_rate": 48000,
              "output_seconds": 20.48,
              "total_seconds": 55.21992404095363,
              "first_output_seconds": 2.986880208016373,
              "completion_after_first_output_seconds": 52.233043832937256,
              "rtf": 2.696285353562189,
              "stage_seconds": {
                "acoustic": 8.156513790600002,
                "decoder": 40.896970500121824,
                "prefill": 0.001333459047600627,
                "prompt": 2.190097542013973,
                "residual": 3.975008749170229
              },
              "baseline_memory_bytes": 4880682580,
              "peak_memory_bytes": 7867364642,
              "incremental_peak_bytes": 2986682062,
              "output_health": {
                "finite": true,
                "non_silent": true,
                "peak_absolute": 0.7693974375724792
              },
              "run": 3,
              "reference_cache": "warm"
            }
          ]
        },
        {
          "variant": "soar",
          "path": "batch",
          "warmup": {
            "runs": [
              {
                "path": "batch",
                "seed": 42,
                "patch_count": 128,
                "chunk_count": 1,
                "waveform_samples": 983040,
                "sample_rate": 48000,
                "output_seconds": 20.48,
                "total_seconds": 74.04821975005325,
                "first_output_seconds": 74.04821725003421,
                "completion_after_first_output_seconds": 2.500019036233425e-06,
                "rtf": 3.615635729983069,
                "stage_seconds": {
                  "acoustic": 27.468377915676683,
                  "decoder": 39.20280450198334,
                  "prefill": 0.0017648329958319664,
                  "prompt": 2.1933051249943674,
                  "residual": 5.18196737440303
                },
                "baseline_memory_bytes": 9758739096,
                "peak_memory_bytes": 14299095502,
                "incremental_peak_bytes": 4540356406,
                "output_health": {
                  "finite": true,
                  "non_silent": true,
                  "peak_absolute": 0.8522157669067383
                },
                "run": 1
              }
            ],
            "total_seconds": 74.04821975005325,
            "explicit_compile_seconds": 0.0,
            "prompt_cache_cleared_after": false
          },
          "medians": {
            "total_seconds": 74.88631266693119,
            "first_output_seconds": 74.88630912499502,
            "output_seconds": 20.48,
            "rtf": 3.6565582356899995,
            "peak_memory_bytes": 14291366094,
            "stage_seconds": {
              "acoustic": 27.567856630310416,
              "decoder": 39.64177354169078,
              "prefill": 0.0014395829057320952,
              "prompt": 2.1910570829641074,
              "residual": 5.388910171575844
            }
          },
          "trials": [
            {
              "path": "batch",
              "seed": 42,
              "patch_count": 128,
              "chunk_count": 1,
              "waveform_samples": 983040,
              "sample_rate": 48000,
              "output_seconds": 20.48,
              "total_seconds": 74.14628924999852,
              "first_output_seconds": 74.14628724998329,
              "completion_after_first_output_seconds": 2.00001522898674e-06,
              "rtf": 3.620424279785084,
              "stage_seconds": {
                "acoustic": 27.363125919597223,
                "decoder": 39.31884345645085,
                "prefill": 0.0014425839763134718,
                "prompt": 2.190424208994955,
                "residual": 5.272453080979176
              },
              "baseline_memory_bytes": 9758739608,
              "peak_memory_bytes": 14290586574,
              "incremental_peak_bytes": 4531846966,
              "output_health": {
                "finite": true,
                "non_silent": true,
                "peak_absolute": 0.8522157669067383
              },
              "run": 1,
              "reference_cache": "cold"
            },
            {
              "path": "batch",
              "seed": 42,
              "patch_count": 128,
              "chunk_count": 1,
              "waveform_samples": 983040,
              "sample_rate": 48000,
              "output_seconds": 20.48,
              "total_seconds": 76.13618625001982,
              "first_output_seconds": 76.13618425000459,
              "completion_after_first_output_seconds": 2.00001522898674e-06,
              "rtf": 3.717587219239249,
              "stage_seconds": {
                "acoustic": 27.906139914761297,
                "decoder": 40.64399153867271,
                "prefill": 0.0014184580650180578,
                "prompt": 2.195726166944951,
                "residual": 5.388910171575844
              },
              "baseline_memory_bytes": 9758739608,
              "peak_memory_bytes": 14288017870,
              "incremental_peak_bytes": 4529278262,
              "output_health": {
                "finite": true,
                "non_silent": true,
                "peak_absolute": 0.8522157669067383
              },
              "run": 2,
              "reference_cache": "warm"
            },
            {
              "path": "batch",
              "seed": 42,
              "patch_count": 128,
              "chunk_count": 1,
              "waveform_samples": 983040,
              "sample_rate": 48000,
              "output_seconds": 20.48,
              "total_seconds": 74.88631266693119,
              "first_output_seconds": 74.88630912499502,
              "completion_after_first_output_seconds": 3.5419361665844917e-06,
              "rtf": 3.6565582356899995,
              "stage_seconds": {
                "acoustic": 27.567856630310416,
                "decoder": 39.64177354169078,
                "prefill": 0.0014395829057320952,
                "prompt": 2.1910570829641074,
                "residual": 5.484185829060152
              },
              "baseline_memory_bytes": 9758739608,
              "peak_memory_bytes": 14291366094,
              "incremental_peak_bytes": 4532626486,
              "output_health": {
                "finite": true,
                "non_silent": true,
                "peak_absolute": 0.8522157669067383
              },
              "run": 3,
              "reference_cache": "warm"
            }
          ]
        },
        {
          "variant": "soar",
          "path": "stream",
          "warmup": {
            "runs": [
              {
                "path": "stream",
                "seed": 42,
                "patch_count": 128,
                "chunk_count": 35,
                "waveform_samples": 983040,
                "sample_rate": 48000,
                "output_seconds": 20.48,
                "total_seconds": 75.99138195894193,
                "first_output_seconds": 3.138317708973773,
                "completion_after_first_output_seconds": 72.85306424996816,
                "rtf": 3.7105166972139614,
                "stage_seconds": {
                  "acoustic": 27.744170669815503,
                  "decoder": 40.67893258028198,
                  "prefill": 0.0014002909883856773,
                  "prompt": 2.200139499967918,
                  "residual": 5.366738917888142
                },
                "baseline_memory_bytes": 9758739608,
                "peak_memory_bytes": 14299123790,
                "incremental_peak_bytes": 4540384182,
                "output_health": {
                  "finite": true,
                  "non_silent": true,
                  "peak_absolute": 0.8522157669067383
                },
                "run": 1
              }
            ],
            "total_seconds": 75.99138195894193,
            "explicit_compile_seconds": 0.0,
            "prompt_cache_cleared_after": false
          },
          "medians": {
            "total_seconds": 73.25881312496495,
            "first_output_seconds": 3.0624565829057246,
            "output_seconds": 20.48,
            "rtf": 3.577090484617429,
            "peak_memory_bytes": 14299095630,
            "stage_seconds": {
              "acoustic": 27.330480249831453,
              "decoder": 38.513349997927435,
              "prefill": 0.0013916249154135585,
              "prompt": 2.1963608750374988,
              "residual": 5.292724046506919
            }
          },
          "trials": [
            {
              "path": "stream",
              "seed": 42,
              "patch_count": 128,
              "chunk_count": 35,
              "waveform_samples": 983040,
              "sample_rate": 48000,
              "output_seconds": 20.48,
              "total_seconds": 74.24261795799248,
              "first_output_seconds": 3.0645125419832766,
              "completion_after_first_output_seconds": 71.1781054160092,
              "rtf": 3.6251278299801015,
              "stage_seconds": {
                "acoustic": 27.442467960063368,
                "decoder": 39.292526956647635,
                "prefill": 0.0013916249154135585,
                "prompt": 2.199502291972749,
                "residual": 5.306729124393314
              },
              "baseline_memory_bytes": 9758739608,
              "peak_memory_bytes": 14289673934,
              "incremental_peak_bytes": 4530934326,
              "output_health": {
                "finite": true,
                "non_silent": true,
                "peak_absolute": 0.8522157669067383
              },
              "run": 1,
              "reference_cache": "cold"
            },
            {
              "path": "stream",
              "seed": 42,
              "patch_count": 128,
              "chunk_count": 35,
              "waveform_samples": 983040,
              "sample_rate": 48000,
              "output_seconds": 20.48,
              "total_seconds": 73.13370866701007,
              "first_output_seconds": 3.0299780840286985,
              "completion_after_first_output_seconds": 70.10373058298137,
              "rtf": 3.570981868506351,
              "stage_seconds": {
                "acoustic": 27.237999410484917,
                "decoder": 38.406805959995836,
                "prefill": 0.0014981250278651714,
                "prompt": 2.1946811249945313,
                "residual": 5.292724046506919
              },
              "baseline_memory_bytes": 9758739608,
              "peak_memory_bytes": 14298701516,
              "incremental_peak_bytes": 4539961908,
              "output_health": {
                "finite": true,
                "non_silent": true,
                "peak_absolute": 0.8522157669067383
              },
              "run": 2,
              "reference_cache": "warm"
            },
            {
              "path": "stream",
              "seed": 42,
              "patch_count": 128,
              "chunk_count": 35,
              "waveform_samples": 983040,
              "sample_rate": 48000,
              "output_seconds": 20.48,
              "total_seconds": 73.25881312496495,
              "first_output_seconds": 3.0624565829057246,
              "completion_after_first_output_seconds": 70.19635654205922,
              "rtf": 3.577090484617429,
              "stage_seconds": {
                "acoustic": 27.330480249831453,
                "decoder": 38.513349997927435,
                "prefill": 0.0013715829700231552,
                "prompt": 2.1963608750374988,
                "residual": 5.217250419198535
              },
              "baseline_memory_bytes": 9758739608,
              "peak_memory_bytes": 14299095630,
              "incremental_peak_bytes": 4540356022,
              "output_health": {
                "finite": true,
                "non_silent": true,
                "peak_absolute": 0.8522157669067383
              },
              "run": 3,
              "reference_cache": "warm"
            }
          ]
        }
      ],
      "maximum_budget_smokes": [],
      "passed": true
    }
  },
  "quality": {
    "report_sha256": "82efb3676af04511e0821bf2a0c178fa4ddd9e664d4e3e1279d48cdadfb0c6d0",
    "manifest": {
      "path": "examples/clone_eval/dots_tts_macos_multilingual_v1.json",
      "sha256": "2dcc499b3cca9130572b9cda5acc861c88a2ab57c782a2b1114c21115b335a60"
    },
    "corpus_lock": {
      "path": "outputs/dots_tts/eval_corpus/manifest.lock.json",
      "sha256": "a77897c60a690e9a90f125a206496fd60a640a87fa886c946a1920100b04c197"
    },
    "asr": {
      "path": "models/qwen3_asr_1_7b/mlx-int8",
      "weights_sha256": "8a9aca31c5715d080f7d891dbac08146aeddf8c34cd53e46cf24d665dcd33786"
    },
    "artifacts": {
      "soar/base": {
        "digest": "abb8b62eedd492b01447cd9601a678fb96a2ba83795a1745d40a934e30538e1e",
        "artifact_class": "base",
        "source": {
          "manifest_sha256": "c5b8638c5083a06b052a2472685ae5933a8f6b2bcbe96cc00cd11624bc12a6af",
          "repo_id": "rednote-hilab/dots.tts-soar",
          "resolved_repo_id": "dots-studio/dots.tts-soar",
          "revision": "e3520f75254d0020a0406db31c51a79d00d22d55"
        },
        "files": {
          "core.safetensors": {
            "bytes": 4396285089,
            "sha256": "d16bce6f079ac3b3162ff25bc7ee92d70e8724fccd17791e15bf5cc6b817d43a"
          },
          "vocoder.safetensors": {
            "bytes": 451779177,
            "sha256": "e5b229d076aec43c20f7f259d2a2bbc75a4384b34adf7a669eefc69bff25e1c0"
          },
          "speaker.safetensors": {
            "bytes": 29122718,
            "sha256": "105356b29dfe40c986412db337c252ab2d227703f433de64b525078de6ca5858"
          },
          "latent_stats.safetensors": {
            "bytes": 1160,
            "sha256": "370ea7fe3e54c06336ff050eaf14f9554417f39b002b2dad5228f67c86c7260f"
          },
          "mlx_config.json": {
            "bytes": 811,
            "sha256": "0d9f32141aa66172bbee722db679cfeae07bc46714e88b244bb7e8b5fc9c6507"
          }
        }
      },
      "soar/int8": {
        "digest": "0b2ad65cd4d2112dff9bc2a113a225a3501b8f606aee3f0bce06387ab1aea9f3",
        "artifact_class": "int8",
        "source": {
          "manifest_sha256": "c5b8638c5083a06b052a2472685ae5933a8f6b2bcbe96cc00cd11624bc12a6af",
          "repo_id": "rednote-hilab/dots.tts-soar",
          "resolved_repo_id": "dots-studio/dots.tts-soar",
          "revision": "e3520f75254d0020a0406db31c51a79d00d22d55"
        },
        "files": {
          "core.safetensors": {
            "bytes": 2949614907,
            "sha256": "9eb78ce331cc116b990dfc4276f49fd548de7a232a710375a02fe9d296a9bdb9"
          },
          "vocoder.safetensors": {
            "bytes": 451779177,
            "sha256": "e5b229d076aec43c20f7f259d2a2bbc75a4384b34adf7a669eefc69bff25e1c0"
          },
          "speaker.safetensors": {
            "bytes": 29122718,
            "sha256": "105356b29dfe40c986412db337c252ab2d227703f433de64b525078de6ca5858"
          },
          "latent_stats.safetensors": {
            "bytes": 1160,
            "sha256": "370ea7fe3e54c06336ff050eaf14f9554417f39b002b2dad5228f67c86c7260f"
          },
          "mlx_config.json": {
            "bytes": 9881,
            "sha256": "956c89107007ef6259594ea59fd99834e83ce992eb4e1a51b5bfb7bce88edeca"
          }
        }
      },
      "mf/base": {
        "digest": "b671893535eeb684edf93abdd525a4966ca965df5cc45d43c371e449342a1eb0",
        "artifact_class": "base",
        "source": {
          "manifest_sha256": "c5b8638c5083a06b052a2472685ae5933a8f6b2bcbe96cc00cd11624bc12a6af",
          "repo_id": "rednote-hilab/dots.tts-mf",
          "resolved_repo_id": "dots-studio/dots.tts-mf",
          "revision": "25c53fb462e57087e52237daa5ea30df1c5cc328"
        },
        "files": {
          "core.safetensors": {
            "bytes": 4398911066,
            "sha256": "bbfca574cc825d7256783f20a4c1ea1c483f49c4ca6b163468a1ea66429d217e"
          },
          "vocoder.safetensors": {
            "bytes": 451779177,
            "sha256": "e5b229d076aec43c20f7f259d2a2bbc75a4384b34adf7a669eefc69bff25e1c0"
          },
          "speaker.safetensors": {
            "bytes": 29122718,
            "sha256": "105356b29dfe40c986412db337c252ab2d227703f433de64b525078de6ca5858"
          },
          "latent_stats.safetensors": {
            "bytes": 1160,
            "sha256": "370ea7fe3e54c06336ff050eaf14f9554417f39b002b2dad5228f67c86c7260f"
          },
          "mlx_config.json": {
            "bytes": 800,
            "sha256": "908698c6aca8f2be396088894585f4ba34e43c863b02a9644d7374ecb559db80"
          }
        }
      },
      "mf/int8": {
        "digest": "424288e3a75930abc8af3aba91b3c42a319436dea927ab85a872f0690771132d",
        "artifact_class": "int8",
        "source": {
          "manifest_sha256": "c5b8638c5083a06b052a2472685ae5933a8f6b2bcbe96cc00cd11624bc12a6af",
          "repo_id": "rednote-hilab/dots.tts-mf",
          "resolved_repo_id": "dots-studio/dots.tts-mf",
          "revision": "25c53fb462e57087e52237daa5ea30df1c5cc328"
        },
        "files": {
          "core.safetensors": {
            "bytes": 2952240868,
            "sha256": "c63e2c5f3409399a976596aa3cf71f8e2ebab6754f6bebe6442bc6d30ca89bc6"
          },
          "vocoder.safetensors": {
            "bytes": 451779177,
            "sha256": "e5b229d076aec43c20f7f259d2a2bbc75a4384b34adf7a669eefc69bff25e1c0"
          },
          "speaker.safetensors": {
            "bytes": 29122718,
            "sha256": "105356b29dfe40c986412db337c252ab2d227703f433de64b525078de6ca5858"
          },
          "latent_stats.safetensors": {
            "bytes": 1160,
            "sha256": "370ea7fe3e54c06336ff050eaf14f9554417f39b002b2dad5228f67c86c7260f"
          },
          "mlx_config.json": {
            "bytes": 9870,
            "sha256": "52d146ac4f7bb3565deb5bb276f90fadb79a127ed7e573eb9d3230df7ec9836a"
          }
        }
      }
    },
    "thresholds": {
      "max_absolute_wer_regression": 0.01,
      "max_speaker_cosine_regression": 0.02
    },
    "records": [
      {
        "key": "mf/base/samantha_en_us/continuation",
        "variant": "mf",
        "artifact_class": "base",
        "reference_id": "samantha_en_us",
        "language": "en",
        "mode": "continuation",
        "artifact_digest": "b671893535eeb684edf93abdd525a4966ca965df5cc45d43c371e449342a1eb0",
        "reference_sha256": "cb971464eee15628d1aa910a876d8ae19eddfbc797cf493d1d794717d7271ae6",
        "reference_text": "My name is Samantha. I speak clearly and calmly.",
        "target_text": "Today the weather is bright and peaceful.",
        "sample_rate": 48000,
        "waveform_samples": 138240,
        "num_patches": 18,
        "asr_errors": 0,
        "asr_tokens": 7,
        "wer": 0.0,
        "speaker_cosine": 0.7789026635939477
      },
      {
        "key": "mf/base/samantha_en_us/speaker_only",
        "variant": "mf",
        "artifact_class": "base",
        "reference_id": "samantha_en_us",
        "language": "en",
        "mode": "speaker_only",
        "artifact_digest": "b671893535eeb684edf93abdd525a4966ca965df5cc45d43c371e449342a1eb0",
        "reference_sha256": "cb971464eee15628d1aa910a876d8ae19eddfbc797cf493d1d794717d7271ae6",
        "reference_text": "My name is Samantha. I speak clearly and calmly.",
        "target_text": "Today the weather is bright and peaceful.",
        "sample_rate": 48000,
        "waveform_samples": 153600,
        "num_patches": 20,
        "asr_errors": 2,
        "asr_tokens": 7,
        "wer": 0.2857142857142857,
        "speaker_cosine": 0.7352271837049196
      },
      {
        "key": "mf/base/tingting_zh_cn/continuation",
        "variant": "mf",
        "artifact_class": "base",
        "reference_id": "tingting_zh_cn",
        "language": "zh",
        "mode": "continuation",
        "artifact_digest": "b671893535eeb684edf93abdd525a4966ca965df5cc45d43c371e449342a1eb0",
        "reference_sha256": "0c7c65d550ea01140f827cd7596b8f4ea6f869739712a01362e5a6dd24f89d01",
        "reference_text": "你好，我叫婷婷。这个声音清晰平稳。",
        "target_text": "今天的天气晴朗而平静。",
        "sample_rate": 48000,
        "waveform_samples": 130560,
        "num_patches": 17,
        "asr_errors": 0,
        "asr_tokens": 10,
        "wer": 0.0,
        "speaker_cosine": 0.8207161497924784
      },
      {
        "key": "mf/base/tingting_zh_cn/speaker_only",
        "variant": "mf",
        "artifact_class": "base",
        "reference_id": "tingting_zh_cn",
        "language": "zh",
        "mode": "speaker_only",
        "artifact_digest": "b671893535eeb684edf93abdd525a4966ca965df5cc45d43c371e449342a1eb0",
        "reference_sha256": "0c7c65d550ea01140f827cd7596b8f4ea6f869739712a01362e5a6dd24f89d01",
        "reference_text": "你好，我叫婷婷。这个声音清晰平稳。",
        "target_text": "今天的天气晴朗而平静。",
        "sample_rate": 48000,
        "waveform_samples": 138240,
        "num_patches": 18,
        "asr_errors": 0,
        "asr_tokens": 10,
        "wer": 0.0,
        "speaker_cosine": 0.8121813784625209
      },
      {
        "key": "mf/int8/samantha_en_us/continuation",
        "variant": "mf",
        "artifact_class": "int8",
        "reference_id": "samantha_en_us",
        "language": "en",
        "mode": "continuation",
        "artifact_digest": "424288e3a75930abc8af3aba91b3c42a319436dea927ab85a872f0690771132d",
        "reference_sha256": "cb971464eee15628d1aa910a876d8ae19eddfbc797cf493d1d794717d7271ae6",
        "reference_text": "My name is Samantha. I speak clearly and calmly.",
        "target_text": "Today the weather is bright and peaceful.",
        "sample_rate": 48000,
        "waveform_samples": 138240,
        "num_patches": 18,
        "asr_errors": 0,
        "asr_tokens": 7,
        "wer": 0.0,
        "speaker_cosine": 0.7927545151348183
      },
      {
        "key": "mf/int8/samantha_en_us/speaker_only",
        "variant": "mf",
        "artifact_class": "int8",
        "reference_id": "samantha_en_us",
        "language": "en",
        "mode": "speaker_only",
        "artifact_digest": "424288e3a75930abc8af3aba91b3c42a319436dea927ab85a872f0690771132d",
        "reference_sha256": "cb971464eee15628d1aa910a876d8ae19eddfbc797cf493d1d794717d7271ae6",
        "reference_text": "My name is Samantha. I speak clearly and calmly.",
        "target_text": "Today the weather is bright and peaceful.",
        "sample_rate": 48000,
        "waveform_samples": 153600,
        "num_patches": 20,
        "asr_errors": 2,
        "asr_tokens": 7,
        "wer": 0.2857142857142857,
        "speaker_cosine": 0.7279747870471241
      },
      {
        "key": "mf/int8/tingting_zh_cn/continuation",
        "variant": "mf",
        "artifact_class": "int8",
        "reference_id": "tingting_zh_cn",
        "language": "zh",
        "mode": "continuation",
        "artifact_digest": "424288e3a75930abc8af3aba91b3c42a319436dea927ab85a872f0690771132d",
        "reference_sha256": "0c7c65d550ea01140f827cd7596b8f4ea6f869739712a01362e5a6dd24f89d01",
        "reference_text": "你好，我叫婷婷。这个声音清晰平稳。",
        "target_text": "今天的天气晴朗而平静。",
        "sample_rate": 48000,
        "waveform_samples": 130560,
        "num_patches": 17,
        "asr_errors": 0,
        "asr_tokens": 10,
        "wer": 0.0,
        "speaker_cosine": 0.8246287325987348
      },
      {
        "key": "mf/int8/tingting_zh_cn/speaker_only",
        "variant": "mf",
        "artifact_class": "int8",
        "reference_id": "tingting_zh_cn",
        "language": "zh",
        "mode": "speaker_only",
        "artifact_digest": "424288e3a75930abc8af3aba91b3c42a319436dea927ab85a872f0690771132d",
        "reference_sha256": "0c7c65d550ea01140f827cd7596b8f4ea6f869739712a01362e5a6dd24f89d01",
        "reference_text": "你好，我叫婷婷。这个声音清晰平稳。",
        "target_text": "今天的天气晴朗而平静。",
        "sample_rate": 48000,
        "waveform_samples": 138240,
        "num_patches": 18,
        "asr_errors": 0,
        "asr_tokens": 10,
        "wer": 0.0,
        "speaker_cosine": 0.8149613621225232
      },
      {
        "key": "soar/base/samantha_en_us/continuation",
        "variant": "soar",
        "artifact_class": "base",
        "reference_id": "samantha_en_us",
        "language": "en",
        "mode": "continuation",
        "artifact_digest": "abb8b62eedd492b01447cd9601a678fb96a2ba83795a1745d40a934e30538e1e",
        "reference_sha256": "cb971464eee15628d1aa910a876d8ae19eddfbc797cf493d1d794717d7271ae6",
        "reference_text": "My name is Samantha. I speak clearly and calmly.",
        "target_text": "Today the weather is bright and peaceful.",
        "sample_rate": 48000,
        "waveform_samples": 138240,
        "num_patches": 18,
        "asr_errors": 0,
        "asr_tokens": 7,
        "wer": 0.0,
        "speaker_cosine": 0.7584857752464498
      },
      {
        "key": "soar/base/samantha_en_us/speaker_only",
        "variant": "soar",
        "artifact_class": "base",
        "reference_id": "samantha_en_us",
        "language": "en",
        "mode": "speaker_only",
        "artifact_digest": "abb8b62eedd492b01447cd9601a678fb96a2ba83795a1745d40a934e30538e1e",
        "reference_sha256": "cb971464eee15628d1aa910a876d8ae19eddfbc797cf493d1d794717d7271ae6",
        "reference_text": "My name is Samantha. I speak clearly and calmly.",
        "target_text": "Today the weather is bright and peaceful.",
        "sample_rate": 48000,
        "waveform_samples": 145920,
        "num_patches": 19,
        "asr_errors": 0,
        "asr_tokens": 7,
        "wer": 0.0,
        "speaker_cosine": 0.7875355984477745
      },
      {
        "key": "soar/base/tingting_zh_cn/continuation",
        "variant": "soar",
        "artifact_class": "base",
        "reference_id": "tingting_zh_cn",
        "language": "zh",
        "mode": "continuation",
        "artifact_digest": "abb8b62eedd492b01447cd9601a678fb96a2ba83795a1745d40a934e30538e1e",
        "reference_sha256": "0c7c65d550ea01140f827cd7596b8f4ea6f869739712a01362e5a6dd24f89d01",
        "reference_text": "你好，我叫婷婷。这个声音清晰平稳。",
        "target_text": "今天的天气晴朗而平静。",
        "sample_rate": 48000,
        "waveform_samples": 130560,
        "num_patches": 17,
        "asr_errors": 0,
        "asr_tokens": 10,
        "wer": 0.0,
        "speaker_cosine": 0.8734471899358045
      },
      {
        "key": "soar/base/tingting_zh_cn/speaker_only",
        "variant": "soar",
        "artifact_class": "base",
        "reference_id": "tingting_zh_cn",
        "language": "zh",
        "mode": "speaker_only",
        "artifact_digest": "abb8b62eedd492b01447cd9601a678fb96a2ba83795a1745d40a934e30538e1e",
        "reference_sha256": "0c7c65d550ea01140f827cd7596b8f4ea6f869739712a01362e5a6dd24f89d01",
        "reference_text": "你好，我叫婷婷。这个声音清晰平稳。",
        "target_text": "今天的天气晴朗而平静。",
        "sample_rate": 48000,
        "waveform_samples": 130560,
        "num_patches": 17,
        "asr_errors": 0,
        "asr_tokens": 10,
        "wer": 0.0,
        "speaker_cosine": 0.7772160857747331
      },
      {
        "key": "soar/int8/samantha_en_us/continuation",
        "variant": "soar",
        "artifact_class": "int8",
        "reference_id": "samantha_en_us",
        "language": "en",
        "mode": "continuation",
        "artifact_digest": "0b2ad65cd4d2112dff9bc2a113a225a3501b8f606aee3f0bce06387ab1aea9f3",
        "reference_sha256": "cb971464eee15628d1aa910a876d8ae19eddfbc797cf493d1d794717d7271ae6",
        "reference_text": "My name is Samantha. I speak clearly and calmly.",
        "target_text": "Today the weather is bright and peaceful.",
        "sample_rate": 48000,
        "waveform_samples": 138240,
        "num_patches": 18,
        "asr_errors": 0,
        "asr_tokens": 7,
        "wer": 0.0,
        "speaker_cosine": 0.793678227944629
      },
      {
        "key": "soar/int8/samantha_en_us/speaker_only",
        "variant": "soar",
        "artifact_class": "int8",
        "reference_id": "samantha_en_us",
        "language": "en",
        "mode": "speaker_only",
        "artifact_digest": "0b2ad65cd4d2112dff9bc2a113a225a3501b8f606aee3f0bce06387ab1aea9f3",
        "reference_sha256": "cb971464eee15628d1aa910a876d8ae19eddfbc797cf493d1d794717d7271ae6",
        "reference_text": "My name is Samantha. I speak clearly and calmly.",
        "target_text": "Today the weather is bright and peaceful.",
        "sample_rate": 48000,
        "waveform_samples": 145920,
        "num_patches": 19,
        "asr_errors": 0,
        "asr_tokens": 7,
        "wer": 0.0,
        "speaker_cosine": 0.8004761971621952
      },
      {
        "key": "soar/int8/tingting_zh_cn/continuation",
        "variant": "soar",
        "artifact_class": "int8",
        "reference_id": "tingting_zh_cn",
        "language": "zh",
        "mode": "continuation",
        "artifact_digest": "0b2ad65cd4d2112dff9bc2a113a225a3501b8f606aee3f0bce06387ab1aea9f3",
        "reference_sha256": "0c7c65d550ea01140f827cd7596b8f4ea6f869739712a01362e5a6dd24f89d01",
        "reference_text": "你好，我叫婷婷。这个声音清晰平稳。",
        "target_text": "今天的天气晴朗而平静。",
        "sample_rate": 48000,
        "waveform_samples": 130560,
        "num_patches": 17,
        "asr_errors": 0,
        "asr_tokens": 10,
        "wer": 0.0,
        "speaker_cosine": 0.8700751662603964
      },
      {
        "key": "soar/int8/tingting_zh_cn/speaker_only",
        "variant": "soar",
        "artifact_class": "int8",
        "reference_id": "tingting_zh_cn",
        "language": "zh",
        "mode": "speaker_only",
        "artifact_digest": "0b2ad65cd4d2112dff9bc2a113a225a3501b8f606aee3f0bce06387ab1aea9f3",
        "reference_sha256": "0c7c65d550ea01140f827cd7596b8f4ea6f869739712a01362e5a6dd24f89d01",
        "reference_text": "你好，我叫婷婷。这个声音清晰平稳。",
        "target_text": "今天的天气晴朗而平静。",
        "sample_rate": 48000,
        "waveform_samples": 130560,
        "num_patches": 17,
        "asr_errors": 0,
        "asr_tokens": 10,
        "wer": 0.0,
        "speaker_cosine": 0.7944757980306567
      }
    ]
  }
}
```

## Slice evidence

**Status:** complete

**Evidence:** Adopted the compact DiT tail after `19 passed`; replaced the relative benchmark with cached-only profiling and fail-closed contract coverage (`34` focused and `792` full unit tests passed, Ruff and `git diff --check` clean). The ignored raw baseline digest is `5798bdb624bcd025f22f691158af7371ad53aff32c066430f53ae18fc7d556d3`; the imported quality-report digest is `82efb3676af04511e0821bf2a0c178fa4ddd9e664d4e3e1279d48cdadfb0c6d0`. Cached medians are MF batch `52.600s` (RTF `2.568`, peak `7.327 GiB`), MF stream `54.895s` (TTFC `2.987s`, peak `7.361 GiB`), SOAR batch `74.886s` (RTF `3.657`, peak `13.310 GiB`), and SOAR stream `73.259s` (TTFC `3.062s`, peak `13.317 GiB`). The canonical contract is complete with four performance cases and sixteen quality records.

**Risks / next:** None; raw JSON remains diagnostic and all later gates read this committed contract.
