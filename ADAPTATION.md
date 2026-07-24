# JustGRPO → Qwen3-0.6B-diffusion-bd3lm 적용 분석

JustGRPO(LLaDA-8B 전용 구현)를 `dllm-hub/Qwen3-0.6B-diffusion-bd3lm-v0.1`에 적용하기 위한 분석. 2026-07-24 기준.

## 두 모델의 차이

| 항목 | LLaDA-8B (원본 대상) | Qwen3-0.6B-diffusion-bd3lm (우리 대상) |
|---|---|---|
| 파라미터 | 8B | 0.6B |
| 방식 | MDLM (전체 시퀀스 확산, full bidirectional attention) | **BD3LM** (블록 확산: 블록 간 causal, 블록 내 bidirectional) |
| mask token | 126336 | **151669** (`<|mask|>`) |
| eos / pad | (LLaDA 고유) | eos 151645 (`<|im_end|>`), pad 151643 (`<|endoftext|>`) |
| vocab | ~126k | **151,936** (로짓 텐서 큼 → 배치 크기 주의) |
| attention | 항상 full bidirectional | **block-causal mask + position_ids를 forward에 전달해야 함** |
| 로딩 | `AutoModel` | `AutoModelForMaskedLM` (`trust_remote_code=True`) |

## 결론: 적용 가능. 수정 필요 지점 4곳

**1. 하드코딩된 상수 교체** (쉬움)
- `grpo.py:26` `mask_id=126336` → 151669
- `train.py:95` `tokenizer.pad_token_id = 126336` → tokenizer 기본값 사용
- `utils/generate.py:45` `mask_id=126336` → 151669
- `eval.py:268` LLaDA tokenizer 하드코딩 → 대상 모델 tokenizer

**2. attention mask / position_ids** (핵심)
- 원본 JustGRPO는 `model(x)`만 호출 (LLaDA는 항상 full bidirectional이라 무방).
- BD3LM은 **block-causal attention mask**를 명시적으로 전달해야 학습 분포와 일치.
  참고 구현: `/home/ubuntu/dllm/dllm/core/samplers/bd3lm.py`의 `_prepare_for_sampling()`
  (block_ids 기반 `[B,1,T,T]` bool mask + padding 제외 logical position_ids).
- JustGRPO의 AR 모드 = block_length 1 → block-causal mask가 곧 causal mask.

**3. rollout (`grpo.py sample()`)**
- 원본은 `utils/generate.py`(LLaDA MDLM 스타일)로 block_length=1 생성.
- BD3LM용으로는 dllm의 `BD3LMSampler`를 쓰거나, causal mask + 한 토큰씩 마스크 예측하는
  AR 루프로 대체. **JustGRPO 철학(학습은 AR 순서) 그대로 유지 가능.**

**4. logprob 계산 (`grpo.py logprob_loss()`) — 개선 기회**
- 원본: 토큰 위치마다 별도 forward × gen_length(256회). LLaDA가 MDLM이라 불가피했던 구조.
- BD3LM은 학습 시 쓰는 "x0 ∥ xt 연결 시퀀스 + 블록 확산 attention mask" 트릭으로
  **한 번의 forward로 전 토큰의 조건부 logprob 계산 가능** (block=1이면 AR teacher-forcing과 동일).
  → 256× forward 루프를 1~2회 forward로 축소 가능. RL 학습 속도에 매우 큰 이득.

## VRAM / 속도 전망 (1× H100 80GB)

- 0.6B bf16 가중치 ≈ 1.2GB. LoRA(r=128) + AdamW 상태 포함해도 수 GB.
- 주의점은 vocab 151,936의 로짓 크기: 배치 64 × 시퀀스 1k 로짓 ≈ 20GB (평가에서 OOM 유발 확인).
  학습 시 배치/시퀀스 설계에서 로짓 메모리를 기준으로 잡을 것.
- 원본 하이퍼파라미터(글로벌 배치 64) 유지 시: `batch_size_per_device` 8~16,
  `grad_accum` 8~4로 재배분하면 1 GPU로 LoRA 200스텝이 수 시간 내 가능할 것으로 추정
  (위 4번 최적화 적용 시 추가 단축).

## 베이스라인 성능

[BENCHMARKS.md](BENCHMARKS.md) 참고 — dLLM 논문 수치와의 재현 비교.
