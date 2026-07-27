# JustGRPO → Qwen3-0.6B-diffusion-bd3lm 적용 분석

JustGRPO(LLaDA-8B 전용 구현)를 `dllm-hub/Qwen3-0.6B-diffusion-bd3lm-v0.1`에 적용하기 위한 분석. 2026-07-24 기준.

## ⚠️ 논문(JustGRPO)과 우리 방법론의 차이 — 정직한 요약

**동일한 것**: GRPO 목적함수(그룹 advantage 정규화 + PPO 클리핑), 정확한 AR 분해 likelihood로 학습,
LoRA 하이퍼파라미터(r=128, α=64, lr 5e-5 = full FT의 10×), 보상 설계(수학 ±1, 코드 format+pass rate),
gen length 256. one-pass 손실은 원본의 256회 forward 루프와 수학적으로 동일함을 fp32 오차 2e-5로 검증.

**다른 것 3가지**:

| # | 항목 | 논문 | 우리 | 이유/영향 |
|---|---|---|---|---|
| 1 | **Rollout 생성 방식** | AR 순서 (block=1) | **네이티브 블록 확산** (block=32) | 이 0.6B 모델은 AR 생성이 붕괴 (첫 토큰 EOS 확률 49% — SFT의 EOS 블록 패딩 아티팩트, 실측). 블록 확산은 같은 문제 8/8 정답. 결과적으로 rollout 분포 ≠ 학습 likelihood 분해 → 엄밀히는 순수 on-policy GRPO가 아니라 **advantage-가중 AR 학습**에 가까움 (GRPO ratio는 on-policy 첫 업데이트에서 1이므로 손실 형태는 동일) |
| 2 | **모델 체급** | LLaDA-8B (MDLM) | Qwen3-0.6B (BD3LM), 13× 작음 | RLVR은 기존 능력을 벼리는 성격이라 작은 모델은 향상폭도 통상 작음. 또한 MDLM(전체 양방향) vs BD3LM(블록 인과)이라 attention mask를 명시적으로 구성해야 함 |
| 3 | **글로벌 배치 = 총 학습량** | 스텝당 프롬프트 64 (8 GPU) | 스텝당 8 (1 GPU) | 스텝 수는 동일(200)하지만 **총 학습량은 논문이 8배 많음** (프롬프트 12,800 vs 1,600 / rollout 204,800 vs 25,600). 업데이트 노이즈도 우리가 큼 |

**기대치 조정**: 논문은 8B에서 +8.9 – +14.7pt. 우리 현실적 목표는 **절반 수준(+4–9pt)이면 성공**.
긍정 신호: 학습 보상이 베이스 기대치(-0.08)에서 20스텝 만에 +0.35–0.39로 상승
(학습 분포 기준 rollout 정답률 46% → 약 67%). 단, 0.6B dLLM에 RL을 얹은 공개 선례가 없어
결과가 어느 쪽이든 novel한 데이터 포인트가 된다.

## SFT-원자 / RL-모듈화 관점 분석 ([arXiv:2606.18089](https://arxiv.org/html/2606.18089v2)) — 다음 실험의 근거

이 논문의 주장: SFT는 원자적 스킬을 얽힌 채 공급하고, RL은 새 능력을 만드는 게 아니라 그것을
분해·모듈화한다. **SFT에 원자가 없으면 RL은 실패한다.** 우리 셋업에 대입한 결과:

핵심 조건까지 반영한 분석 (논문: **RL 데이터가 SFT 지원집합 밖의 새 조합일 때(C_SFT ∩ C_RL = ∅) 일반화 최강**).
베이스 모델의 SFT 스택([dLLM 논문](https://arxiv.org/html/2602.22661v1)): Qwen3-0.6B 사전학습 +
[tulu-3-sft-mixture](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture)(Persona MATH 15만 + Persona GSM 5만 등)
+ [smoltalk](https://huggingface.co/datasets/HuggingFaceTB/smoltalk)(**NuminaMath-CoT, MetaMathQA-50k, APIGen 포함**)
+ OpenCoder Python stage1·2. bd3lm SFT는 **max_length 512** (mdlm은 1024).

| 도메인 | 원자(SFT 내) | RL 데이터 ∩ SFT | 논문 기준 판정 | 실측 |
|---|---|---|---|---|
| GSM8K | ✅ 충분 (GSM 스타일 합성 20만+) | 실제 GSM8K train은 SFT에 없음 → **교집합 ∅, 같은 스킬족의 새 인스턴스** | **이상적 조건** | Run 1 v1: +2.95 ✅ / rollout 8개 중 5–8 정답 |
| 코드 | ✅ 충분 (OpenCoder·APIGen·Self-OSS) | AceCode-Hard는 SFT에 없는 별개 분포 → **교집합 ∅** | **이상적 조건** | Run 3 보상 시작부터 0.6–1.0 |
| MATH | ✅ 있음 (NuminaMath CoT) — 단 bd3lm SFT max_length 512로 **긴 CoT가 잘려 학습됐을 가능성** | **교집합 ≠ ∅** (NuminaMath 소스에 MATH train 포함) | 불리: ①0.6B 용량 ②512 잘림 ③RL이 SFT 지원집합 재탕 | Run 1 v1: 13.6→13.0 보합 ✅ |

**결론**: GSM8K·코드 트랙은 논문이 말하는 최적 배치(원자는 SFT에, RL은 새 인스턴스에)와 정확히
일치 — 추가 SFT 불필요. MATH 정체는 "원자 부재"가 아니라 **용량 + SFT 시퀀스 잘림 + 데이터 겹침**의
복합 원인으로 보는 것이 정확하다. 믹싱 런은 "SFT 밖 새 조합" 조건을 가장 잘 만족하는 트랙.

**📋 다음 실험 (RL 결과 확인 후 결정)**: *MATH-longCoT-SFT-then-RL* —
단순 수학 SFT 추가가 아니라 **긴 CoT가 잘리지 않는 조건(max_length 1024–2048)으로 SFT 한 겹**
(NuminaMath/OpenMathInstruct류) 후 JustGRPO를 얹어, "잘린 원자 복원 → RL 모듈화"가
0.6B diffusion LM의 MATH 정체(13%)를 깨는지 검증. RL 프롬프트는 SFT에 안 쓴 문제로 분리해
C_SFT ∩ C_RL = ∅ 조건 유지. 현 매트릭스(단독 vs 믹싱, bd3lm vs mdlm) 완료 후 착수.

## 왜 AR-순서 rollout이 이 모델에서 안 되는가 (모델 선택 가이드)

원인은 모델이 "최신"이라서가 아니라 **학습 목적함수의 차이**다:

- **MDLM 계열** (논문의 LLaDA-8B): 전체 시퀀스에 무작위 비율로 마스크를 씌워 복원하도록 학습.
  "앞부분 깨끗 + 뒤 전부 마스크"(= AR 순서 생성의 매 스텝 상태)가 학습 분포 안에 있음 → AR rollout 가능.
- **BD3LM 계열** (우리의 Qwen3-0.6B-bd3lm): 32토큰 블록 단위로만 노이즈를 학습. 블록=1 상황을
  본 적이 없고, SFT 시 마지막 블록을 EOS로 패딩해 첫 AR 위치에서 EOS 확률이 49%로 튐(실측)
  → AR rollout 붕괴. 대신 블록 확산 rollout은 완벽 작동.

**논문과 동일한 AR-순서 rollout을 쓰려면** (추후 결정용 후보):

| 모델 | 방식 | 비고 |
|---|---|---|
| [LLaDA-8B-Instruct](https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct) | MDLM 8B | 논문 그 자체. 1×H100에서 LoRA 가능하나 rollout 매우 느림 |
| [Dream-7B](https://huggingface.co/Dream-org/Dream-v0-Instruct-7B) | MDLM류 7B | 대안 대형 dLLM |
| [Qwen3-0.6B-diffusion-**mdlm**-v0.1](https://huggingface.co/dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1) | MDLM 0.6B | **우리 모델의 쌍둥이 MDLM 버전** — 같은 체급으로 논문 방식 재현 가능. 단 베이스 성능이 낮음 (GSM8K 29.3 vs bd3lm 46.3) |

## 두 모델의 차이

| 항목 | LLaDA-8B (원본 대상) | Qwen3-0.6B-diffusion-bd3lm (우리 대상) |
|---|---|---|
| 파라미터 | 8B | 0.6B |
| 방식 | MDLM (전체 시퀀스 확산, full bidirectional attention) | **BD3LM** (블록 확산: 블록 간 causal, 블록 내 bidirectional) |
| mask token | 126336 | **151669** (`<|mask|>`) |
| eos / pad | (LLaDA 고유) | eos 151645 (`<|im_end|>`), pad 151643 (`<|endoftext|>`) |
| vocab | 약 126k | **151,936** (로짓 텐서 큼 → 배치 크기 주의) |
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
  → 256× forward 루프를 1–2회 forward로 축소 가능. RL 학습 속도에 매우 큰 이득.

## VRAM / 속도 전망 (1× H100 80GB)

- 0.6B bf16 가중치 ≈ 1.2GB. LoRA(r=128) + AdamW 상태 포함해도 수 GB.
- 주의점은 vocab 151,936의 로짓 크기: 배치 64 × 시퀀스 1k 로짓 ≈ 20GB (평가에서 OOM 유발 확인).
  학습 시 배치/시퀀스 설계에서 로짓 메모리를 기준으로 잡을 것.
- 원본 하이퍼파라미터(글로벌 배치 64) 유지 시: `batch_size_per_device` 8–16,
  `grad_accum` 8–4로 재배분하면 1 GPU로 LoRA 200스텝이 수 시간 내 가능할 것으로 추정
  (위 4번 최적화 적용 시 추가 단축).

## 베이스라인 성능

[BENCHMARKS.md](BENCHMARKS.md) 참고 — dLLM 논문 수치와의 재현 비교.

## 후속 트랙 조사: diffu-GRPO로 bd3lm 베이스(46.6) 넘기 (7/27)

**문제**: JustGRPO(AR 방식)는 강한 bd3lm 베이스와 호환되지 않아(mismatch 붕괴), mdlm에서만 작동하는데
mdlm 베이스(29.8)가 너무 낮아 RL 성공(34.7)해도 bd3lm 베이스(46.6)를 못 넘는다.

**후보 해법**: 확산-네이티브 GRPO = [diffu-GRPO/d1](https://github.com/dllm-reasoning/d1).
AR로 억지 변환하지 않고 블록 확산 rollout + 확산 일관 손실 → mismatch 원천 제거.
강한 bd3lm 베이스를 유지한 채 RL 가능 → 46.6을 실제로 넘길 유일한 경로.

**조사 결과 (dllm 레포 `examples/rl`)**:
- `dllm/pipelines/rl/grpo/`에 `DiffuGRPOTrainer` 구현 존재 (d1/diffu-grpo 참조 구현)
- 제공 예제: `examples/rl/grpo/llada/train.py` (LLaDA), `examples/rl/grpo/a2d/mdlm/train.py` (Tiny-A2D MDLM)
- **bd3lm(블록 확산) 전용 예제는 없음** → 블록 인과 attention·블록 스케줄에 맞춘 적용 필요 (MDLM 예제를 베이스로 개조)
- 이 방법은 JustGRPO가 명시적으로 피한 "diffusion-specific adaptation"에 해당 — 즉 별개 방법론의 비교 실험이 됨

**우선순위**: 현재 mdlm 매트릭스(JustGRPO 방법론 검증) 완주 후 착수. "JustGRPO on mdlm" vs
"diffu-GRPO on bd3lm"의 비교 자체가 논문급 질문 — 전자는 방법이 단순하나 약한 베이스,
후자는 방법이 복잡하나 강한 베이스. 어느 쪽이 최종 성능에서 이기는지가 핵심.
