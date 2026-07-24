# Qwen3-0.6B-diffusion-bd3lm-v0.1 — 베이스 성능 측정

JustGRPO 학습 전 베이스라인. [dLLM 논문](https://arxiv.org/abs/2602.22661)(Table 3) /
[dllm 레포 README](https://github.com/ZHZisZZ/dllm) "Reproduced" 수치와의 재현 비교.

- **일시**: 2026-07-24
- **하드웨어**: 1× H100 80GB
- **프레임워크**: [ZHZisZZ/dllm](https://github.com/ZHZisZZ/dllm) + lm-evaluation-harness (공식 재현 스크립트 `examples/a2d/bd3lm/eval.sh`와 동일 설정)
- **생성 설정**: `max_new_tokens=256, steps=256, block_size=32, cfg_scale=0.0`, 0-shot, chat template 적용
- **모델**: `~/data/models/Qwen3-0.6B-diffusion-bd3lm-v0.1` (HF `dllm-hub/Qwen3-0.6B-diffusion-bd3lm-v0.1`)
- **로그/원본 결과**: `~/data/bench/bd3lm-v0.1/`

## 결과 (JustGRPO 4개 태스크)

| 태스크 | lm-eval 태스크명 | Metric | 논문 (dLLM Table 3) | 우리 측정 | 차이 |
|---|---|---|---:|---:|---|
| GSM8K | `gsm8k_cot` | exact_match (flexible) | 46.3 | **45.72** (±1.37) | −0.58, 오차범위 내 ✅ |
| MATH | `minerva_math` | math_verify | 12.9 | **13.60** (±0.47) | +0.7, 근접 재현 ✅ |
| HumanEval | `humaneval_instruct` | pass@1 | 46.3 | **46.95** (±3.91) | +0.65, 오차범위 내 ✅ |
| MBPP | `mbpp_instruct` | pass@1 | 38.2 | **38.20** (±2.18) | ±0.00, 정확히 일치 ✅ |

> 참고: dLLM 논문 수치 자체가 이 프레임워크로 잰 "Reproduced" 값이므로, 동일 설정이면 오차범위 내 일치가 기대값.

## 학습 후 결과

### Run 1 — GSM8K 단독 LoRA 200스텝 (7/25 새벽 완료)

모델: [HF 병합본](https://huggingface.co/LLM-OS-Models2/Qwen3-0.6B-diffusion-bd3lm-justgrpo-run1-gsm8k-lora) · 로그 `~/data/bench/run1-gsm8k-lora/`

| 벤치마크 | 베이스 | Run 1 (GSM8K 학습) | Δ |
|---|---:|---:|---|
| **GSM8K** (타깃) | 45.72 | **48.67** (±1.38) | **+2.95** ✅ |
| MATH | 13.60 | 13.00 (±0.46) | −0.60 (보합) |
| HumanEval | 46.95 | **35.98** (±3.76) | **−10.97** ⚠️ |
| MBPP | 38.20 | (재측정 중 — 동시부하 OOM으로 1회 재실행) | |

**관찰**: 타깃 도메인(GSM8K)은 +3pt 향상, 인접 도메인(MATH)은 보합, **비관련 도메인(코드)은 −11pt 급락** —
단일 도메인 RL의 파멸적 간섭(catastrophic interference)이 0.6B diffusion LM에서도 뚜렷하게 재현됨.
[arXiv:2507.17512](https://arxiv.org/abs/2507.17512)가 보고한 현상과 일치하며,
**믹싱 런(Run 2)이 이 간섭을 막아주는지가 이 실험의 핵심 비교 포인트가 됨.**

## 실행 노트

- vocab 151,936이라 로짓 텐서가 커서 배치 64 × 4태스크 병렬은 OOM → 배치 16 × 3태스크 병렬로 재실행 (HumanEval은 배치 64에서 완료).
- vLLM 사용 불가: `AutoModelForMaskedLM` + 블록 확산 커스텀 디노이징 루프라 AR 서빙 엔진과 비호환.
- JustGRPO 원본 `eval.py`는 LLaDA 전용(mask 126336, LLaDA tokenizer 하드코딩)이라 이 모델 평가에는 dllm 프레임워크를 사용.
