# JustGRPO × Qwen3-0.6B-diffusion-bd3lm

[LeapLabTHU/JustGRPO](https://github.com/LeapLabTHU/JustGRPO)를 기반으로,
[`dllm-hub/Qwen3-0.6B-diffusion-bd3lm-v0.1`](https://huggingface.co/dllm-hub/Qwen3-0.6B-diffusion-bd3lm-v0.1)
(block diffusion, [dLLM](https://arxiv.org/abs/2602.22661) 논문의 Tiny-A2D 모델)을 **LoRA + GRPO**로 학습하는 실험 저장소.

원본 README는 [README_original.md](README_original.md) 참고.

## 구성 요소

| 항목 | 위치 |
|---|---|
| 베이스 모델 (1.5GB) | `~/data/models/Qwen3-0.6B-diffusion-bd3lm-v0.1` |
| HF 데이터셋 캐시 (GSM8K, MATH-500) | `~/data/hf_cache` (`HF_HOME`으로 지정) |
| 코드 RL 데이터 (AceCode-Hard 21K) | `datasets/acecode_hard.jsonl` |
| 평가 프레임워크 (dllm + lm-eval-harness) | `/home/ubuntu/dllm` (uv venv: `.venv`) |
| 베이스 모델 벤치마크 기록 | [BENCHMARKS.md](BENCHMARKS.md) |
| JustGRPO 적용 분석 (수정 지점) | [ADAPTATION.md](ADAPTATION.md) |

## 학습 데이터 (JustGRPO 논문 기준, 태스크당 별도 모델)

- **GSM8K** — `openai/gsm8k` train 7,473
- **MATH-500** — `ankner/math-500` train 7,500
- **Code** — AceCode-Hard 21K (AceCoder-87K에서 DiffuCoder 파이프라인으로 선별)

평가는 GSM8K test(1,319) / MATH-500 test(500) / HumanEval / MBPP.

## 원본 대비 수정 사항

- `train.py`, `eval.py`, `data/math.py`: 데이터셋 경로 `"gsm8k"` → `"openai/gsm8k"`
  (legacy 네임스페이스 없는 경로가 최신 `huggingface_hub`에서 로드 불가)

## 베이스 모델 평가 (dllm 프레임워크)

```bash
cd /home/ubuntu/dllm && source .venv/bin/activate
bash run_eval4_parallel.sh   # gsm8k_cot / humaneval_instruct / mbpp_instruct / minerva_math
# 결과: ~/data/bench/bd3lm-v0.1/, 정리본은 BENCHMARKS.md
```

생성 설정은 dLLM 공식 eval.sh와 동일: `max_new_tokens=256, steps=256, block_size=32, cfg_scale=0.0`.

> vLLM은 사용 불가 — 이 모델은 `AutoModelForMaskedLM` + 커스텀 블록 확산 디노이징 루프로 생성하므로
> AR 전용 서빙 엔진(vLLM)과 호환되지 않는다.

## JustGRPO 학습 (LoRA)

> ⚠️ 원본 JustGRPO 코드는 LLaDA-8B 전용 하드코딩(mask token 126336, tokenizer, `utils/generate.py`)이
> 있어 이 모델(mask/pad 처리, vocab, chat template이 다름)에 그대로 쓸 수 없다. 적용 작업은 진행 중.

```bash
accelerate launch --num_processes 1 train.py \
  --dataset gsm8k \
  --grad_accum 64 \
  --lora \
  --total_steps 200
```
