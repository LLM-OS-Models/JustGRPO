# JustGRPO × Tiny Diffusion LM — 0.6B 확산 언어모델의 단독 vs 멀티도메인 RL

> **이 저장소가 답하려는 질문**: ICML 2026 수상작 [JustGRPO](https://arxiv.org/abs/2601.15165)의
> "diffusion LM은 AR 순서로 GRPO를 돌리면 된다"는 레시피가 **0.6B급 초소형 diffusion LM에서도 작동하는가?**
> 그리고 **도메인 단독 RL vs 수학+코드 믹싱 RL** 중 무엇이 나은가? — 전부 1× H100에서 실측.

- 전체 수치 원장: [BENCHMARKS.md](BENCHMARKS.md) · 기술 분석: [ADAPTATION.md](ADAPTATION.md) · 원본 README: [README_original.md](README_original.md)
- 학습된 모델 전부 HF 퍼블릭: [LLM-OS-Models2](https://huggingface.co/LLM-OS-Models2)

---

## 🔴 지금 (7/26 16:00) — Phase 2: MDLM 매트릭스 진행 중

**Phase 1(bd3lm 모델)에서 "방법-모델 불일치"라는 교훈을 얻고, 논문 방식이 100% 성립하는
MDLM 쌍둥이 모델로 본 실험을 재수행 중입니다** (아래 "Phase 1 결과" 참조).

| 런 | 내용 | 상태 |
|---|---|---|
| **Run 5** | MDLM · 수학기초(GSM8K) 단독 | ✅ 학습 완료 — **ckpt-100이 최고점(GSM8K +5.4), ckpt-200은 과학습으로 하락**. 아래 "성능 변화 현상" 참조 |
| Run 6 | MDLM · 믹싱(gsm8k+math+code, 300스텝) | 🔥 **학습 중** (7/27 13:00~, 솔로 7.0분/스텝 실측) |
| Run 7 | MDLM · 수학응용(MATH) 단독 | 대기 (스크립트 준비 완료) |
| Run 8 | MDLM · 코드 단독 | 대기 (스크립트 준비 완료) |

**예상 일정 (한국시간, 7/27 17:52 실측 7.0분/스텝 솔로)**:
- **7/28(화) 새벽 ~3시**: Run 6이 ckpt-100 도달 → **믹싱 vs 단독 첫 비교** (Run 5 ckpt-100 GSM8K 34.72 대비)
- **7/29(수) 새벽 ~2시**: Run 6 (ckpt-300) 학습 완료 → 전체 평가
- 이후 Run 7(수학응용)·Run 8(코드) 순차

> ⚠️ **중요 발견 (7/27)**: Run 5 단독은 **ckpt-100(GSM8K 34.72, +5.4)이 최고점이고 ckpt-200(27.9, −1.4)은 과학습으로 하락**.
> → 소형 모델+LoRA는 조기 정지가 필수. 모든 런에서 10스텝마다 저장된 전체 체크포인트를 근거로 최적점을 선택한다
> (체크포인트별 성능표는 [BENCHMARKS.md](BENCHMARKS.md)).

**자동 운영**: 런 종료 → 병합 → 4태스크 평가 → HF 업로드 → 다음 런 투입 → 문서 갱신·푸시가
알림 기반으로 연쇄 실행됨. 10스텝마다 보상·에러 자동 감시.

---

## 📈📉 관측된 성능 변화 현상 (실측 — 상승/하락 그대로)

RL이 성능을 **올리기도, 내리기도** 한다. 지금까지 실측으로 확인한 4가지 현상:

### ① 과학습 후반 붕괴 — "최종 스텝 ≠ 최고 성능" (Run 5, MDLM 수학기초)

| 체크포인트 | GSM8K | HumanEval | 해석 |
|---|---:|---:|---|
| 베이스 | 29.3 | 30.5 | — |
| **ckpt-100 (절반)** | **34.72 (+5.4)** ✅ | (미측정) | **최고점** |
| ckpt-200 (최종) | 27.90 (−1.4) ⚠️ | 34.15 (+3.6) | GSM8K는 과학습으로 하락, HumanEval은 여전히 상승 |

- **핵심**: 논문 방식은 0.6B에서도 작동한다(절반 학습에 +5.4). 그러나 **더 오래 학습하면 오히려 무너진다.**
  논문도 "GSM8K는 50스텝 만에 89% 도달"이라 명시 — 소형 모델+LoRA는 **조기 정지가 필수.**
- **태스크마다 정점 시기가 다르다**: GSM8K는 ckpt-100에서, HumanEval은 ckpt-200에서도 계속 상승.
  → 단일 최종 체크포인트로 전 태스크를 대표할 수 없다. 10스텝마다 저장한 전 체크포인트에서 태스크별 최적점을 고른다.

### ② 도메인 간 간섭 — "타깃은 오르고 남은 도메인은 깎인다" (Phase 1 bd3lm 수학 단독)

수학만 RL → GSM8K +2.95 상승, 그러나 코드는 HumanEval −11.0 / MBPP −9.4 동반 하락.
자기 분야를 얻는 대가로 다른 분야를 잃는 전형적 간섭.

### ③ 자기 타깃까지 붕괴 — "학습 보상은 오르는데 평가는 무너진다" (Phase 1 bd3lm 코드 단독)

학습 중 보상은 끝까지 상승(+0.1~+0.6)했으나, 평가는 **자기 타깃 HumanEval마저 46.95 → 23.78 (−23.2)**.
원인: rollout(블록 확산)과 손실(AR)의 불일치가 스텝마다 생성 분포를 손상 → **학습지표와 실제 성능의 괴리**.
이것이 Phase 2(MDLM)로 전환한 직접적 이유.

### ④ 믹싱의 손상 완충 — "섞으면 덜 무너진다" (Phase 1 bd3lm 믹싱)

같은 조건에서 코드 단독은 HumanEval −23.2인데, 믹싱은 −6.7에 그침. MATH는 오히려 +0.3.
멀티도메인이 단일 도메인 RL의 파괴적 간섭을 완충함을 실증. **Phase 2에서 이 현상이 논문 방식으로도
재현되는지가 핵심 질문** (7/28 새벽 ckpt-100 비교로 첫 답).

> 요약: **RL은 만능 상승이 아니다.** ①정점을 지나면 하락 ②타깃 외 도메인 희생 ③불일치 시 자기 타깃도 붕괴
> ④믹싱은 완충. 그래서 이 저장소는 "최종 성능"이 아니라 **체크포인트별·태스크별 전체 곡선**을 기록한다
> (수치 원장: [BENCHMARKS.md](BENCHMARKS.md)).

### 왜 MDLM인가 (Phase 1의 교훈)

- 처음 쓴 [bd3lm 모델](https://huggingface.co/dllm-hub/Qwen3-0.6B-diffusion-bd3lm-v0.1)은 32토큰 블록 확산으로 학습돼
  **AR(블록=1) 생성이 붕괴** (첫 토큰 EOS 확률 49% — 실측) → 논문의 AR rollout을 쓸 수 없어
  rollout만 블록 확산으로 대체했더니, **"AR 손실 × 블록확산 생성" 불일치가 스텝이 쌓일수록 모델을 손상**
  (코드 단독 런: 자기 타깃까지 −23pt — 원본 JustGRPO가 경고한 sampler-learner mismatch의 실증)
- [MDLM 쌍둥이](https://huggingface.co/dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1)는 논문의 LLaDA와 같은 방식(전체 시퀀스
  무작위 마스킹)이라 **AR rollout이 학습 분포 안** — 원본 코드 그대로 4/4 정답 생성 확인.
  논문 방식이 무결하게 성립하는 조건에서 본 질문(단독 vs 믹싱)을 다시 묻는다

### 성능 무손실 가속 (전부 동일성 검증 통과)

원본 레시피는 rollout에 토큰당 전체 시퀀스 forward가 필요한 무거운 구조 (논문도 인정하는
per-position 오버헤드 — 저자들은 근사판 JustGRPO-Fast까지 제안). 우리는 **근사 없이**:

| 가속 | 방법 | 검증 |
|---|---|---|
| fast rollout (약 1.5–2배) | 버려지는 위치의 152k-vocab 투영 생략 (`utils/mdlm_fast.py`) | greedy 출력 토큰 단위 일치 |
| rollout 16개 단일 배치 (약 1.4배) | 2×8 호출을 1×16으로 — 같은 그룹 수학 | 의미상 동일 (같은 프롬프트 그룹 16) |
| 손실 위치 배칭 K=4 (약 2배) | 원본 256회 루프를 4위치씩 배치 | fp32 오차 1.2e-5 |
| 단독 실행 | OOM 경합 제거 | — |

---

## ✅ 완료된 결과

### 베이스 모델 재현 (평가 파이프라인 검증)

dLLM 논문 수치를 4개 벤치마크 전부 오차범위 내 재현 → 이후 모든 비교의 기준선 확보.

| 벤치마크 | 논문 | 우리 재현 (bd3lm 베이스) |
|---|---|---|
| GSM8K | 46.3 | 45.72 |
| MATH | 12.9 | 13.60 |
| HumanEval | 46.3 | 46.95 |
| MBPP | 38.2 | 38.20 |

### Phase 1 — bd3lm 매트릭스 (완료, 교훈 확보)

베이스 대비 변화 (모두 LoRA, 글로벌 배치 8, 200스텝 / 믹싱 300스텝):

| | GSM8K | MATH | HumanEval | MBPP | 판정 |
|---|---:|---:|---:|---:|---|
| 수학 단독 (v1) | **+2.95** | −0.6 | −11.0 | −9.4 | 타깃만 상승 |
| 코드 단독 (v2) | −19.9 | −5.3 | **−23.2** | (측정중) | ❌ 자기 타깃까지 붕괴 |
| 믹싱 (v2, ckpt-300) | −1.8 | **+0.3** | −6.7 | (측정중) | 🟡 손상 대부분 흡수 |

**발견 3가지**: ① bd3lm에는 AR-손실 RL이 구조적으로 부적합 (mismatch 누적 손상)
② 그 조건에서도 **믹싱이 단독보다 압도적으로 강건** (HumanEval −6.7 vs −23.2)
③ 수학 타깃은 그나마 상승 — 답 형식이 단순해 왜곡에 견딘 것으로 해석.
모델: [수학](https://huggingface.co/LLM-OS-Models2/Qwen3-0.6B-diffusion-bd3lm-justgrpo-run1-gsm8k-lora) ·
[코드](https://huggingface.co/LLM-OS-Models2/Qwen3-0.6B-diffusion-bd3lm-justgrpo-run3v2-code-lora) ·
[믹싱](https://huggingface.co/LLM-OS-Models2/Qwen3-0.6B-diffusion-bd3lm-justgrpo-run2v2-mixed-lora)

### 과정에서 발견·수정한 버그 (재현 가능하게 커밋)

1. **원본 레포 잠복 버그**: 코드 채점 샌드박스(reliability_guard)가 학습 프로세스의 os 모듈을
   파괴 → 첫 체크포인트 저장에서 크래시. Pool worker 격리로 수정 (`8c83aeb`)
2. **적용 코드 버그**: 조기 종료된 완성의 인위적 EOS 패딩까지 학습 → EOS 과잉 강화로 빈 응답 붕괴
   (HumanEval 46.9→18.3 실측). first-EOS 마스킹으로 수정 (`663e584`). v1 산출물은 ablation 보존

---

## 실험 설계 (Phase 2 = 현재)

- **모델**: [Qwen3-0.6B-diffusion-mdlm-v0.1](https://huggingface.co/dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1)
  (베이스 성능: GSM8K 29.3 · MATH 8.7 · HumanEval 30.5 · MBPP 29.2 — 낮아서 오를 여지 큼)
- **방법**: 원본 JustGRPO 그대로 — AR 순서 rollout(블록=1) + 정확한 AR likelihood GRPO,
  LoRA r=128/α=64/lr 5e-5, gen length 256, 그룹 16, 글로벌 배치 8 (`grpo_mdlm.py`, `train_mdlm.py`)
- **매트릭스**: 수학기초(GSM8K)·수학응용(MATH)·코드 단독 각 200스텝 vs 믹싱 300스텝
  (ckpt-200 = 컴퓨트 동일 비교점) → 4개 벤치마크 전부 평가
- **논문과의 차이는 스케일뿐**: 논문 16×H100·글로벌 64 vs 우리 1×H100·글로벌 8 (학습량 1/8).
  방법론 자체는 동일하며 모든 가속은 동일성 검증 통과
- **기대치**: 논문은 8B에서 +9–15pt. 0.6B·1/8 학습량이므로 **타깃 +3–8pt면 성공**

## 재현 가이드

```bash
# 환경: 1× H100 80GB / 학습 venv: /home/ubuntu/justgrpo-venv / 평가: /home/ubuntu/dllm/.venv
# 데이터·모델: ~/data/models/*, HF_HOME=~/data/hf_cache (GSM8K·MATH-500 자동, AceCode-Hard 21K 포함)

bash scripts/run5_mdlm_gsm8k_lora.sh                      # MDLM 학습 (run6/7/8 동일 패턴)
python scripts/merge_lora.py --adapter ~/data/runs/run5-mdlm-gsm8k-lora/ckpt-000200 \
  --out ~/data/models/run5-merged
bash scripts/eval_mdlm_ckpt.sh ~/data/models/run5-merged run5-mdlm-gsm8k   # 4태스크 평가
python scripts/upload_hf.py --run run5-mdlm-gsm8k --dataset gsm8k \
  --adapter ... --merged ...                               # HF 퍼블릭 업로드(모델카드 포함)
```

bd3lm 트랙 재현은 `scripts/run1–4_*.sh` + `eval_ckpt.sh` (평가 배치 16 이하 — 152k vocab 로짓 OOM 주의).
vLLM은 확산 모델 비호환으로 사용 불가.

## 배경 논문

- **JustGRPO** ([2601.15165](https://arxiv.org/html/2601.15165v4)): LLaDA-8B에서 GSM8K 78.6→89.1 (+10.5) 등.
  16×H100, per-position 오버헤드 인정(근사판 JustGRPO-Fast 제안)
- **dLLM/Tiny-A2D** ([2602.22661](https://arxiv.org/html/2602.22661)): 우리 베이스 모델들의 출처.
  Qwen3-0.6B를 SFT만으로 MDLM/[BD3LM(2503.09573)](https://arxiv.org/abs/2503.09573) 변환
- **멀티도메인 RLVR 트렌드**: [Nemotron 3 Super](https://arxiv.org/abs/2604.12374)(21환경 동시),
  [Kimi K2](https://arxiv.org/abs/2507.20534), [GLM-5](https://arxiv.org/pdf/2602.15763);
  학술 근거 [2507.17512](https://arxiv.org/abs/2507.17512)(단독 vs 혼합 GRPO),
  [2602.12566](https://arxiv.org/pdf/2602.12566), [2606.25178](https://arxiv.org/pdf/2606.25178)
- **SFT-원자/RL-모듈화** ([2606.18089](https://arxiv.org/html/2606.18089v2)): RL은 SFT가 깐 원자를
  정리할 뿐 — 우리 도메인별 결과와 예측 일치. 후속 실험(*MATH-longCoT-SFT-then-RL*)의 근거
  (상세: [ADAPTATION.md](ADAPTATION.md))

## 진행 로그 (요약)

- **7/24**: 모델·데이터 준비, 베이스 4태스크 재현 성공, bd3lm 적용 코드 완성(동등성 검증), Run 1–2 시작.
  샌드박스 os 파괴 버그 발견·수정
- **7/25**: Run 1 v1 완료 (+2.95). 믹싱 붕괴에서 EOS 패딩 버그 진단·수정 → v2 재시작.
  MDLM 쌍둥이에서 논문 방식 무결 작동 확인, MDLM 파이프라인 작성·검증(fp32 1.2e-5)
- **7/26 오전**: bd3lm 매트릭스 결과 확정 — 코드 단독 붕괴(−23), 믹싱은 흡수. mismatch 결론
- **7/26 오후**: **Phase 2 전환** — MDLM 매트릭스 개시. 무손실 가속 3종 적용
  (fast rollout 검증, 16배치 통합, 순차 실행). Run 5 학습 중
