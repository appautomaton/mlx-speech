from __future__ import annotations

import os

import pytest

from mlx_speech.checkpoints.layout import MODELS_ROOT


EDITX_DIR = MODELS_ROOT / "stepfun" / "step_audio_editx" / "original"
TOKENIZER_DIR = MODELS_ROOT / "stepfun" / "step_audio_tokenizer" / "original"
COSYVOICE_DIR = EDITX_DIR / "CosyVoice-300M-25Hz"
FUNASR_DIR = (
    TOKENIZER_DIR
    / "dengcunqin"
    / "speech_paraformer-large_asr_nat-zh-cantonese-en-16k-vocab8501-online"
)
PROMPT_AUDIO = MODELS_ROOT.parent / "outputs" / "source" / "hank_hill_ref.wav"
LOCAL_TOKENIZER_JSON = EDITX_DIR / "tokenizer.json"

HAS_EDITX_CHECKPOINT = EDITX_DIR.exists() and any(EDITX_DIR.glob("*.safetensors"))
HAS_TOKENIZER_CHECKPOINT = TOKENIZER_DIR.exists()
HAS_COSYVOICE_ASSETS = COSYVOICE_DIR.exists()
HAS_FUNASR_ASSETS = FUNASR_DIR.exists()
HAS_VQ06_ASSETS = (TOKENIZER_DIR / "speech_tokenizer_v1.onnx").exists()
HAS_LOCAL_TOKENIZER = LOCAL_TOKENIZER_JSON.exists()
RUN_LOCAL_INTEGRATION = os.environ.get("RUN_LOCAL_INTEGRATION") == "1"
HAS_LOCAL_PROMPT_AUDIO = PROMPT_AUDIO.exists()

skip_no_editx = pytest.mark.skipif(
    not HAS_EDITX_CHECKPOINT,
    reason="Step-Audio-EditX checkpoint not found",
)
skip_no_tokenizer = pytest.mark.skipif(
    not HAS_TOKENIZER_CHECKPOINT,
    reason="Step-Audio tokenizer not found",
)
skip_no_cosyvoice = pytest.mark.skipif(
    not HAS_COSYVOICE_ASSETS,
    reason="CosyVoice assets not found",
)
skip_no_funasr = pytest.mark.skipif(
    not HAS_FUNASR_ASSETS,
    reason="Step-Audio FunASR assets not found",
)
skip_no_vq06 = pytest.mark.skipif(
    not HAS_VQ06_ASSETS,
    reason="Step-Audio semantic tokenizer assets not found",
)
skip_no_local_tokenizer = pytest.mark.skipif(
    not HAS_LOCAL_TOKENIZER,
    reason="Step-Audio tokenizer.json not found",
)
skip_no_integration = pytest.mark.skipif(
    not RUN_LOCAL_INTEGRATION or not (HAS_EDITX_CHECKPOINT and HAS_TOKENIZER_CHECKPOINT and HAS_LOCAL_PROMPT_AUDIO),
    reason="manual local integration test; requires RUN_LOCAL_INTEGRATION=1 and local Step-Audio assets",
)
