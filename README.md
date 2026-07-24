# JustGRPO × Qwen3-0.6B-diffusion-bd3lm

[JustGRPO](https://arxiv.org/html/2601.15165v4) (ICML 2026 Outstanding Paper, LLaDA-8B 대상)의 RL 레시피를
[dLLM](https://arxiv.org/html/2602.22661) 논문의 초소형 block-diffusion 모델
[`dllm-hub/Qwen3-0.6B-diffusion-bd3lm-v0.1`](https://huggingface.co/dllm-hub/Qwen3-0.6B-diffusion-bd3lm-v0.1)에
**LoRA + GRPO**로 적용하는 실험. 원본 README: [README_original.md](README_original.md)

## 목표

> **0.6B block-diffusion 모델도 JustGRPO(AR-순서 GRPO)로 추론 성능을 끌어올릴 수 있는가?**

JustGRPO 논문이 LLaDA-8B에서 보인 향상 (gen length 256 기준):

| 벤치마크 | LLaDA-8B 베이스 | + JustGRPO | 향상 |
|---|---:|---:|---:|
| GSM8K | 78.6 | **89.1** | +10.5 |
| MATH-500 | 30.4 | **45.1** | +14.7 |
| HumanEval | 40.5 | **49.4** | +8.9 |
| MBPP | 40.7 | **52.4** | +11.7 |

우리 실험의 시작점 (Qwen3-0.6B-bd3lm 베이스, [dLLM 논문](https://arxiv.org/html/2602.22661) Table 3 vs 우리 재현 — [BENCHMARKS.md](BENCHMARKS.md)):

| 벤치마크 | 논문 수치 | 우리 재현 (2026-07-24) |
|---|---:|---:|
| GSM8K | 46.3 | **45.72** ✅ |
| MATH | 12.9 | 측정 중 (~17:12 완료) |
| HumanEval | 46.3 | **46.95** ✅ |
| MBPP | 38.2 | **38.20** ✅ (정확히 일치) |

세 태스크 모두 오차범위 내 재현 성공 → 평가 파이프라인 신뢰 가능.

학습된 모델은 HF org [`LLM-OS-Models2`](https://huggingface.co/LLM-OS-Models2)에 업로드 예정.

## 핵심 기술 변경 (원본 JustGRPO 대비)

원본은 LLaDA-8B 전용 하드코딩이 많아 그대로 적용 불가. 전체 분석은 [ADAPTATION.md](ADAPTATION.md).

1. **모델/토크나이저 일반화** — mask 126336 → `tokenizer.mask_token_id`(151669), `AutoModelForMaskedLM` 로드, 데이터 경로 `openai/gsm8k` 수정
2. **Rollout = 네이티브 블록 확산** — 이 0.6B 모델은 AR(블록=1) 생성이 붕괴함 (첫 위치 EOS 확률 49%, 숫자 반복 루프 — 실측). 블록 확산 샘플링(dllm `BD3LMSampler`, block 32)은 같은 문제를 8/8 정답 → rollout은 모델 본연의 방식 사용
3. **손실 = AR one-pass logprob** — BD3LM의 `[x0 ∥ xt]` 학습 마스크 트릭(블록=1)으로 **256회 forward 루프를 1회 forward로 대체**. fp32에서 순차 루프와 최대 오차 2e-5로 수학적 동일성 검증 완료 (`tests/test_adaptation.py`). GRPO의 ratio는 on-policy 첫 업데이트에서 1이므로 advantage-가중 AR 학습으로 원본과 동일하게 성립 — "학습은 AR 순서"라는 JustGRPO 철학 유지
4. **스모크 테스트 통과** — GSM8K 실문제 rollout 8개 중 5개 정답, advantage/LoRA gradient 정상 (rewards `[-1,1,1,1,-1,1,-1,1]`)

## 환경

| 항목 | 위치 |
|---|---|
| 베이스 모델 (1.5GB) | `~/data/models/Qwen3-0.6B-diffusion-bd3lm-v0.1` |
| 학습 venv (torch 2.13, transformers 4.57, dllm editable) | `/home/ubuntu/justgrpo-venv` |
| 평가 프레임워크 (dllm + lm-eval-harness) | `/home/ubuntu/dllm` (venv: `.venv`) |
| HF 데이터셋 캐시 | `~/data/hf_cache` (`HF_HOME`) |
| 코드 RL 데이터 (AceCode-Hard 21K) | `datasets/acecode_hard.jsonl` |
| 벤치마크 로그 | `~/data/bench/bd3lm-v0.1/` |

vLLM은 사용 불가 (AR 전용 엔진 ↔ 확산 디노이징 루프 비호환).

## 실행

```bash
# 학습 (GSM8K, LoRA) — 1× H100
cd ~/data/JustGRPO && source /home/ubuntu/justgrpo-venv/bin/activate
export HF_HOME=~/data/hf_cache
accelerate launch --num_processes 1 train.py --dataset gsm8k --grad_accum 8 --lora --total_steps 200

# 베이스/학습후 평가 (dllm 프레임워크)
cd /home/ubuntu/dllm && source .venv/bin/activate
bash run_eval4_parallel.sh   # 설정: max_new_tokens=256, steps=256, block_size=32, 0-shot
```

## 진행 로그 (2026-07-24)

- 15:28 모델·데이터셋(GSM8K/MATH-500/AceCode-Hard) 다운로드 완료
- 16:03 베이스 평가 4태스크 시작 (배치64 OOM → 배치16 재실행; vocab 152k 로짓이 원인)
- 16:03 HumanEval 46.95 ✅ / 16:14 MBPP 38.20 ✅ / 16:29 GSM8K 45.72 ✅
- 16:20 AR rollout 붕괴 발견 → 블록 확산 rollout + AR one-pass 손실 설계로 전환, 동등성·스모크 테스트 통과
- 진행 중: MATH 평가 (~17:12 완료 예정)
- 다음: **GSM8K LoRA 학습 시작** (~17:15, 200 스텝, 예상 6–8시간) → 학습 후 4태스크 재평가 → BENCHMARKS 갱신 → HF 업로드
