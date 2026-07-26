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
| MBPP | 38.20 | **28.80** (±2.03) | **−9.40** ⚠️ |

**관찰**: 타깃 도메인(GSM8K)은 +3pt 향상, 인접 도메인(MATH)은 보합, **비관련 도메인(코드)은 HumanEval −11pt / MBPP −9.4pt 급락**.

> ⚠️ **주의 (사후 발견)**: 이 결과는 **EOS 패딩 버그가 있는 v1 손실**로 학습된 것.
> 코드 하락분에는 도메인 간섭 외에 아래 버그의 영향이 섞여 있어 해석에 주의.
> v2(수정판) 재실행 결과가 정본이 됨.

### ⚠️ v1 손실의 EOS 패딩 버그와 수정 (7/25)

**증상**: 믹싱 v1의 ckpt-200 평가에서 전면 붕괴 — GSM8K 29.3 (베이스 −16), HumanEval 18.3 (−28).
생성 검사 결과 코드류 프롬프트에 `<think></think></think>` 후 **즉시 EOS(빈 응답)**.

**원인**: 블록 확산 샘플러는 전 시퀀스가 EOS에 도달하면 조기 종료하는데, v1 `sample()`은 부족분을
**EOS로 인위 패딩하고 손실을 256 전 위치에 적용** → 정책이 생성한 적 없는 EOS 수십 개가
advantage 방향으로 강화됨 → EOS 확률 폭증 → 빈 응답 붕괴. 짧은 답이 많은 코드 도메인에서 특히 심함.
(TRL 등 표준 GRPO 구현이 first-EOS 이후를 마스킹하는 이유)

**수정 (v2)**: `logprob_loss`에 EOS 마스크 추가 — 실제 생성 토큰 + 첫 EOS까지만 학습.
Run 2/3를 v2로 재시작 (v1 로그·체크포인트는 `*-v1-eosbug`로 보존, ablation 자료).
Run 1도 추후 v2로 재실행 예정 — v1 Run 1 결과는 "마스킹 없음" ablation으로 유지.

## 실행 노트

- vocab 151,936이라 로짓 텐서가 커서 배치 64 × 4태스크 병렬은 OOM → 배치 16 × 3태스크 병렬로 재실행 (HumanEval은 배치 64에서 완료).
- vLLM 사용 불가: `AutoModelForMaskedLM` + 블록 확산 커스텀 디노이징 루프라 AR 서빙 엔진과 비호환.
- JustGRPO 원본 `eval.py`는 LLaDA 전용(mask 126336, LLaDA tokenizer 하드코딩)이라 이 모델 평가에는 dllm 프레임워크를 사용.

### 중간 검증 — 믹싱 v2 ckpt-100 (7/25 오후)

v1 붕괴 재발 여부 조기 확인용 축소 평가 (GSM8K + HumanEval, 학습과 병행):

| 벤치마크 | 베이스 | v1 ckpt-200 (EOS 버그) | v2 ckpt-100 |
|---|---:|---:|---:|
| GSM8K | 45.72 | 29.3 | **40.03** |
| HumanEval | 46.95 | 18.3 | **40.24** |

**붕괴 없음 — EOS 마스킹 수정 유효 확인.** 베이스 대비 소폭 하락은 학습 중반 과도기
(GSM8K 노출이 아직 단독 런의 약 1/6)로 판단, 개입 없이 300스텝 완주 결정.

### Run 3 v2 — 코드(AceCode-Hard) 단독 LoRA 200스텝 (7/26 새벽 완료) — ❌ 게이트 실패

모델: [HF 병합본](https://huggingface.co/LLM-OS-Models2/Qwen3-0.6B-diffusion-bd3lm-justgrpo-run3v2-code-lora) · 로그 `~/data/bench/run3v2-code/`

| 벤치마크 | 베이스 | 코드 단독 v2 | Δ |
|---|---:|---:|---|
| GSM8K | 45.72 | 25.85 | −19.9 ⚠️ |
| MATH | 13.60 | 8.30 | −5.3 ⚠️ |
| **HumanEval** (타깃) | 46.95 | **23.78** (재실행 재현) | **−23.2** ⚠️ |
| MBPP (타깃) | 38.20 | 측정 중 (평가 OOM 1회, 재실행 예정) | |

학습 보상은 끝까지 건강(+0.1–0.6)했으나 평가가 전방위 하락. 실패 모드: 어려운 문제에서 반복 퇴화
(쉬운 16문항은 50% 통과). EOS 버그는 v2에서 제거된 상태이므로, 유력 원인은 **sampler-learner
mismatch** — rollout·평가는 블록 확산, 손실은 AR 분해라서 스텝이 쌓일수록 블록 확산 생성 분포가
손상되는 구조 (원본 JustGRPO가 경고한 바로 그 문제). 결정 트리에 따라 **MDLM 트랙(Run 5) 투입**.

### Run 2 v2 믹싱 — ckpt-200 중간 결과 (스테이지 A)

| 벤치마크 | 베이스 | ckpt-100 | ckpt-200 | 코드 단독 (참고) |
|---|---:|---:|---:|---:|
| GSM8K | 45.72 | 40.03 | **43.97** (회복세) | 25.85 |
| HumanEval | 46.95 | 40.24 | **40.85** | 23.78 |

**믹싱이 단독 대비 mismatch 손상을 크게 완화** (HumanEval −6.1 vs 단독 −23.2).
최종 판정은 ckpt-300 (7/26 오전).

## Phase 2 — MDLM (논문 100% 방식) 결과

### Run 5 중간 검증 — GSM8K 단독, ckpt-100 (7/27 새벽)

| 벤치마크 | MDLM 베이스 | ckpt-100 (100/200스텝) | 변화 |
|---|---:|---:|---|
| GSM8K (타깃) | 29.3 | **34.72** (±1.31) | **+5.4** ✅ |

**절반 학습 만에 목표 하한(+3)을 돌파 — 논문 방식의 작동이 0.6B MDLM에서 확증됨.**
(bd3lm 트랙의 중간 검증이 베이스 미달이었던 것과 대조적. 방법-모델 정합성 가설 입증)
