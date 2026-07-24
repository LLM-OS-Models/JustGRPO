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

## 실행 노트

- vocab 151,936이라 로짓 텐서가 커서 배치 64 × 4태스크 병렬은 OOM → 배치 16 × 3태스크 병렬로 재실행 (HumanEval은 배치 64에서 완료).
- vLLM 사용 불가: `AutoModelForMaskedLM` + 블록 확산 커스텀 디노이징 루프라 AR 서빙 엔진과 비호환.
- JustGRPO 원본 `eval.py`는 LLaDA 전용(mask 126336, LLaDA tokenizer 하드코딩)이라 이 모델 평가에는 dllm 프레임워크를 사용.
