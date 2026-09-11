# 🔗 Cross-Modal Multi-Engine Integration Guide

## 1. Unified Architecture Integration Flow

`ameva-runtime`은 하부 C++ HAL 라이브러리(`libameva_vulkan.so`)를 단일 진실 공급원(SSOT)으로 제공하며, 상위의 다양한 오픈소스 모달리티 엔진이 동일한 Vulkan 인스턴스를 공유합니다.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       Modality Packages (Python / JS)                       │
│  termux-stt  │  termux-diffusion  │  termux-bitnet  │  termux-vision       │
└──────┬────────────────┬───────────────────┬──────────────────┬───────────────┘
       │                │                   │                  │
       ▼                ▼                   ▼                  ▼
┌──────────────┐ ┌──────────────┐    ┌──────────────┐   ┌──────────────┐
│ whisper.cpp  │ │ stable-diff  │    │  llama.cpp   │   │  llava.cpp   │
└──────┬───────┘ └──────┬───────┘    └──────┬───────┘   └──────┬───────┘
       │                │                   │                  │
       └────────────────┴─────────┬─────────┴──────────────────┘
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    libameva_vulkan.so (Unified HAL)                         │
│  - Adreno 830 Subgroup Bypass & Mali 128-byte Strict Buffer Alignment       │
│  - Single ICD Loader Provenance & 12-Stage Safety Fallback (V0 ~ V11)       │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 2. Code Integration Example

### Whisper STT 가속 연동:
```python
from ameva_runtime import vulkan as avr
import termux_stt

# 1. Vulkan 가속 백엔드 확보
ctx = avr.create_context(device="auto")

# 2. STT 엔진에 가속 컨텍스트 바인딩
engine = termux_stt.create_engine("whisper", model="base", accelerator=ctx)
result = engine.transcribe("speech.wav")
print("Transcribed:", result.text)
```

### Stable Diffusion 가속 연동:
```python
from ameva_runtime import vulkan as avr
import termux_diffusion as td

# Vulkan 런타임 진단 후 고속 생성
if avr.is_available():
    image_path = td.generate("Cyberpunk Seoul", device="vulkan", preset="fast")
    print(f"Generated on GPU: {image_path}")
```
