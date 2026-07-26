# JustGRPO × Tiny Diffusion LM: 0.6B Block-Diffusion 모델의 멀티 도메인 RL 실험

> **한 줄 요약**: ICML 2026 수상작 [JustGRPO](https://arxiv.org/abs/2601.15165)의 RL 레시피(원본: LLaDA-8B 전용)를
> 13배 작은 block-diffusion 모델 [Qwen3-0.6B-diffusion-bd3lm](https://huggingface.co/dllm-hub/Qwen3-0.6B-diffusion-bd3lm-v0.1)에
> 이식하고, **단독 도메인 RL vs 멀티 도메인(수학+코드 믹싱) RL**의 성능을 4개 벤치마크에서 비교하는 실험 저장소.

- 원본 JustGRPO README: [README_original.md](README_original.md)
- 기술 분석 (하드코딩 제거·설계 결정·논문과의 차이): [ADAPTATION.md](ADAPTATION.md)
- 벤치마크 결과 원장 (전체 수치): [BENCHMARKS.md](BENCHMARKS.md)

## 📌 현재 상태 (7/26 14:05 갱신) — 🔄 **주 무대를 MDLM으로 전환**

**한눈 요약**: bd3lm 매트릭스에서 "AR-손실 × 블록확산 생성"의 구조적 불일치를 확인
(코드 단독 전방위 하락, 믹싱은 손상 흡수하나 순이득 없음). **본 목표(단독 vs 믹싱)를
논문 100% 방식이 성립하는 MDLM 쌍둥이 모델에서 재수행하기로 결정 (7/26 14시)** —
현재 MDLM-수학기초(GSM8K)와 MDLM-믹싱이 동시 학습 중. 이후 수학응용(MATH)·코드 단독 순.

| 런 | 데이터 (모델) | 상태 / 결과 | 다음 |
|---|---|---|---|
| Run 1 v1 | GSM8K 단독 (bd3lm) | ✅ GSM8K 45.72→**48.67 (+2.95)**, 단 EOS버그 혼재 · [HF](https://huggingface.co/LLM-OS-Models2/Qwen3-0.6B-diffusion-bd3lm-justgrpo-run1-gsm8k-lora) | v2가 정본 |
| Run 3 v2 | 코드 단독 (bd3lm) | ❌ **게이트 실패**: HumanEval 46.95→23.78, GSM8K −20, MATH −5 (sampler-learner mismatch) · [HF](https://huggingface.co/LLM-OS-Models2/Qwen3-0.6B-diffusion-bd3lm-justgrpo-run3v2-code-lora) | 완료 |
| Run 2 v2 | **믹싱** (bd3lm) | 🟡 완주. ckpt-300: GSM8K 43.97(−1.8) · HumanEval 40.24(−6.7) · **MATH 13.90(+0.3, ckpt-200)** — 단독 대비 손상 대폭 완화 · [HF](https://huggingface.co/LLM-OS-Models2/Qwen3-0.6B-diffusion-bd3lm-justgrpo-run2v2-mixed-lora) | 잔여 태스크 평가 마무리 |
| Run 1 v2 | GSM8K 단독 재실행 (bd3lm) | ⏸️ 140/200에서 일시정지 (체크포인트 보존, MDLM에 GPU 양보) | 여유 시 재개/ckpt-140 평가 |
| **Run 5** | GSM8K 기초 (**MDLM, 논문 100% 방식**) | 🔥 학습 중 (ckpt-20에서 재개) | 종료 후 4태스크 평가 |
| **Run 6** | **믹싱** (**MDLM**, gsm8k+math+code) | 🔥 학습 중 (7/26 14:05 시작, 300스텝) | ckpt-200/300 평가 → **MDLM 단독 vs 믹싱 비교** |
| Run 7/8 | MATH(수학응용)·코드 단독 (MDLM) | 대기 | Run 5/6 종료 후 순차 |

**발견·수정한 버그 2건** (둘 다 커밋·문서화·푸시 완료):
1. 원본 레포 잠복 버그: 코드 채점 샌드박스가 학습 프로세스의 os 모듈 파괴 → Pool worker 격리로 수정 (`8c83aeb`)
2. 우리 적용 코드 버그: 조기 종료된 완성의 EOS 패딩까지 학습해 EOS 과잉 강화(빈 응답 붕괴)
   → first-EOS 마스킹으로 수정 (`663e584`), v1 산출물은 `runs/*-v1-eosbug`에 ablation으로 보존

## 📐 논문 방법론과의 동일성 — 어디까지 같고 어디가 다른가

| 구성 요소 | [논문](https://arxiv.org/html/2601.15165v4)·[원본 레포](README_original.md) | 우리 (v2) | 동일? |
|---|---|---|---|
| GRPO 목적함수 (그룹 advantage 정규화 + 클리핑) | ✓ | ✓ | **동일** |
| 학습 likelihood: 정확한 AR 분해 (ELBO 근사 없음) | ✓ | ✓ (one-pass 트릭, fp32 오차 2e-5 검증) | **동일 (계산만 빠름)** |
| 보상 설계·데이터·gen length 256·그룹 크기 16 | ✓ | ✓ | **동일** |
| LoRA 설정 (r=128, α=64, lr 5e-5) | ✓ | ✓ | **동일** |
| Rollout 생성 | AR 순서 (block=1) | **블록 확산 (block=32)** | **다름** — 이 0.6B 모델은 AR 생성이 붕괴해 불가피. 학습은 여전히 AR likelihood |
| EOS 처리 | 이슈 없음 (LLaDA는 고정 길이 rollout) | first-EOS 마스킹 (v2) | 구조 차이에 따른 표준 보정 |
| 스케일·학습량 | LLaDA-8B, 글로벌 배치 64 (8 GPU) — 총 12,800 프롬프트 | 0.6B, 글로벌 배치 8 (1 GPU) — 총 1,600 프롬프트 | **다름 — 논문이 8배 더 학습** (스텝 수는 동일 200) |

정리: **"학습 목적함수와 하이퍼파라미터는 논문과 동일, rollout 방식과 스케일은 다름."**
rollout 차이로 원본이 지적한 sampler-learner mismatch가 일부 재도입되지만, GRPO의 ratio가
on-policy 첫 업데이트에서 1이므로 advantage-가중 AR 학습(RAFT류)으로 여전히 정당하다.
v1에서조차 타깃 +2.95가 나온 만큼 v2(버그 제거)에서는 **타깃 도메인의 확실한 향상**을 기대하되,
향상폭은 체급 차이로 논문(+10pt대)의 절반 수준(+3–9pt)이 현실적 목표다.

---

## 1. 배경: 이 실험이 서 있는 세 갈래 연구

**① JustGRPO — "diffusion LM은 AR 순서로 RL 학습하는 게 낫다"**
([arXiv:2601.15165](https://arxiv.org/html/2601.15165v4), ICML 2026 Outstanding Paper)
Diffusion LM의 임의 순서 생성은 오히려 추론 잠재력(Pass@k)을 깎아먹으므로, RL 학습만큼은
표준 GRPO를 AR 순서로 돌리면 된다는 논문. LLaDA-8B-Instruct에서의 결과 (gen length 256):

| 벤치마크 | LLaDA-8B 베이스 | + JustGRPO | 향상 |
|---|---|---|---|
| GSM8K | 78.6 | **89.1** | +10.5 |
| MATH-500 | 30.4 | **45.1** | +14.7 |
| HumanEval | 40.5 | **49.4** | +8.9 |
| MBPP | 40.7 | **52.4** | +11.7 |

**② dLLM / Tiny-A2D — 우리가 학습하는 베이스 모델의 출처**
([arXiv:2602.22661](https://arxiv.org/html/2602.22661), [GitHub](https://github.com/ZHZisZZ/dllm))
Qwen3-0.6B를 SFT만으로 block diffusion([BD3LM, arXiv:2503.09573](https://arxiv.org/abs/2503.09573)) 모델로
변환한 초소형 dLLM. 블록(32토큰) 단위로는 AR, 블록 내부는 확산으로 생성.

**③ 멀티 도메인 RLVR — 2026년 프런티어 모델들의 표준 레시피**
- [Nemotron 3 Super](https://arxiv.org/abs/2604.12374): 21개 환경 / 37개 데이터셋 동시 RLVR
  ([공식 레시피](https://github.com/NVIDIA-NeMo/Nemotron/blob/main/docs/nemotron/super3/README.md))
- [Kimi K2](https://arxiv.org/abs/2507.20534): Gym식 프레임워크로 검증가능 보상 태스크 대규모 혼합
- [GLM-5](https://arxiv.org/pdf/2602.15763): SFT → 추론 RL → 에이전틱 RL → 일반 RL 다단계
- 학술 근거: [Can One Domain Help Others? (arXiv:2507.17512)](https://arxiv.org/abs/2507.17512) —
  GRPO로 수학/코드/퍼즐 단독 vs 혼합 비교. 혼합은 대체로 이득이지만 도메인 간 간섭도 실재.
  관련: [To Mix or To Merge (arXiv:2602.12566)](https://arxiv.org/pdf/2602.12566),
  [멀티 도메인 커리큘럼 RLVR (arXiv:2606.25178)](https://arxiv.org/pdf/2606.25178)

**우리의 질문**: 이 트렌드가 0.6B급 diffusion LM에서도 성립하는가? —
diffusion LM + 멀티 도메인 RLVR 조합은 아직 발표된 사례가 없다.

**④ SFT-원자 / RL-모듈화 관점** ([arXiv:2606.18089](https://arxiv.org/html/2606.18089v2))
"SFT가 원자적 스킬을 깔고 RL은 그걸 정리할 뿐 새 능력은 못 만들며, **RL 데이터가 SFT 지원집합
밖일 때(C_SFT ∩ C_RL = ∅) 일반화가 최강**"이라는 관점을 우리 데이터 관계에 실제로 대입해 검증:
GSM8K·코드는 원자는 SFT에 있고(RL 데이터는 SFT와 겹치지 않는 새 인스턴스) → 이상적 조건 → RL 향상
(Run 1: +2.95). 반면 MATH는 RL 데이터가 SFT의 NuminaMath와 **겹치고**, bd3lm SFT의 max_length
512로 긴 CoT가 잘려 학습된 정황까지 겹쳐 RL만으로는 보합(13.6→13.0).
→ **다음 실험 후보**: 긴 CoT를 잘리지 않게 넣는 *MATH-longCoT-SFT-then-RL*
(상세 설계·데이터 겹침 분석은 [ADAPTATION.md](ADAPTATION.md)).

---

## 2. 실험 설계와 지금까지의 결과

**베이스 모델 재현 검증 (완료)** — 4개 벤치마크 전부 dLLM 논문 수치를 오차범위 내 재현:

| 벤치마크 | dLLM 논문 | 우리 재현 |
|---|---|---|
| GSM8K | 46.3 | **45.72** |
| MATH (minerva) | 12.9 | **13.60** |
| HumanEval | 46.3 | **46.95** |
| MBPP | 38.2 | **38.20** |

**Run 1 v1 결과** (GSM8K 단독, EOS 버그 있는 v1 손실 — 참고용 ablation):

| 벤치마크 | 베이스 | v1 학습 후 | 변화 |
|---|---|---|---|
| GSM8K (타깃) | 45.72 | **48.67** | **+2.95** |
| MATH | 13.60 | 13.00 | 보합 |
| HumanEval | 46.95 | 35.98 | −10.97 |
| MBPP | 38.20 | 28.80 | −9.40 |

타깃은 오르고 코드가 급락 — 다만 코드 하락분에는 도메인 간섭과 EOS 버그 효과가 섞여 있어,
**v2 재실행 결과가 정본**이 된다. 전체 수치와 버그 분석은 [BENCHMARKS.md](BENCHMARKS.md).

**학습량 설계**: 스텝당 프롬프트 8개(각 16 rollouts). 단독 런은 해당 도메인 1,600 프롬프트,
믹싱 런은 총 2,400개(도메인당 800개). 체크포인트가 10스텝마다 저장되므로 믹싱 런에서
ckpt-200(단독과 총 컴퓨트 동일)과 ckpt-300(도메인 노출 보정) 둘 다 평가한다.

> **논문 대비 학습량**: 옵티마이저 스텝 수는 논문 LoRA 세팅과 동일(200)하지만, 글로벌 배치가
> 논문 64(8 GPU) vs 우리 8(1 GPU)이라 **총 학습 프롬프트는 논문의 1/8** (12,800 vs 1,600).
> GPU가 이미 연산 포화(99%) 상태라 배치를 키우면 스텝 시간이 비례 증가하므로, 1 GPU 4런
> 매트릭스를 위해 글로벌 8을 유지하기로 결정 (7/25). 단독 vs 믹싱 비교는 동일 조건이라 내부적으로
> 유효하며, 논문 절대치와의 비교에는 이 차이를 감안할 것.

---

## 3. 방법: 원본 JustGRPO에서 바뀐 것

핵심 변경 3가지 + 수정 2가지 (상세: [ADAPTATION.md](ADAPTATION.md)):

1. **Rollout = 네이티브 블록 확산 샘플링.** 이 0.6B 모델은 AR(블록=1) 생성이 붕괴
   (첫 위치 EOS 확률 49% — SFT의 EOS 블록 패딩 아티팩트. 실측). 블록 확산은 같은 문제 8/8 정답
   → rollout은 dllm의 `BD3LMSampler`(block 32) 사용
2. **손실 = AR one-pass logprob.** BD3LM의 x0∥xt 학습 마스크 트릭(블록=1 특수화)으로
   원본의 256회 forward 루프를 **1회 forward로 대체**. fp32에서 순차 루프 대비 최대 오차 2e-5로
   동일성 검증 (`tests/test_adaptation.py`)
3. **멀티 도메인 믹싱 로더** (`data/mixed.py`): 배치 단위 라운드로빈 + 도메인별 보상 라우팅.
   GRPO 그룹 정규화가 보상 스케일 차이(수학 ±1, 코드 0–2)를 흡수
4. **샌드박스 격리 수정**: 코드 채점의 reliability_guard를 워커 프로세스로 격리 (부모 os 파괴 버그)
5. **first-EOS 마스킹 (v2)**: 손실을 실제 생성 토큰 + 첫 EOS까지만 적용 (표준 completion masking)

---

## 4. 재현 가이드

```bash
# 환경: 1× H100 80GB, uv venv 2개
#  - 학습: /home/ubuntu/justgrpo-venv (torch 2.13, transformers 4.57, peft, dllm editable)
#  - 평가: /home/ubuntu/dllm/.venv (+ lm-evaluation-harness)
# 모델/데이터: ~/data/models/Qwen3-0.6B-diffusion-bd3lm-v0.1, HF_HOME=~/data/hf_cache

bash scripts/run1_gsm8k_lora.sh     # 단독 GSM8K (Run 2/3/4도 동일 패턴)
python scripts/merge_lora.py --adapter ~/data/runs/run1-gsm8k-lora/ckpt-000200 \
  --out ~/data/models/run1-gsm8k-lora-merged
bash scripts/eval_ckpt.sh ~/data/models/run1-gsm8k-lora-merged run1-gsm8k-lora
python scripts/upload_hf.py --run run1-gsm8k-lora --dataset gsm8k \
  --adapter ~/data/runs/run1-gsm8k-lora/ckpt-000200 \
  --merged ~/data/models/run1-gsm8k-lora-merged   # HF LLM-OS-Models2에 퍼블릭 업로드
```

주의: vLLM 사용 불가(확산 디노이징 루프는 AR 전용 엔진과 비호환). 평가 배치는 16 이하 권장
(vocab 152k 로짓 텐서 때문에 배치 64는 OOM — 실측).

---

## 5. 진행 로그

**7/24**
- 15:28 모델 + 데이터셋(GSM8K, MATH-500, AceCode-Hard 21K) 다운로드
- 16:03 – 17:07 베이스 4태스크 평가 → 전부 재현 성공
- 오후 적용 코드 완성: one-pass 손실 동등성 검증, AR 붕괴 발견 → 블록 확산 rollout 설계, 스모크 테스트 통과
- 17:10 Run 1 v1 (GSM8K 단독) 시작 / 18:35 Run 2 v1 (믹싱) 동시 시작 — GPU 99% 포화
- 19:11 Run 2 크래시 → 샌드박스 os 파괴 버그 수정 후 재시작

**7/25**
- 새벽 Run 1 v1 200스텝 완주 → 평가 (GSM8K +2.95 / 코드 급락) → HF 업로드, Run 3 v1 (코드) 시작
- 오전 믹싱 ckpt-200 평가에서 빈 응답 붕괴 발견 → **EOS 패딩 버그 진단·수정 (first-EOS 마스킹)**
- 10:05 / 10:10 **Run 2 v2 (믹싱) · Run 3 v2 (코드) 재시작** ← 지금 학습 중

**실행 순서 결정 트리** (7/25 확정)

```
Run 2 v2 (믹싱) + Run 3 v2 (코드) 학습 중 → 종료 시 각각 평가
  ├─ 잘 나오면: 수학 2개 (Run 1 v2 GSM8K, Run 4 MATH-500) 학습·평가
  │             + MDLM 트랙(논문 100% 동일 방식, mdlm-v0.1) 병행
  └─ 잘 안 나오면: bd3lm 수학 런 생략, 바로 MDLM 트랙으로 전환
```

판정 기준: 코드 런 타깃 +2pt 이상 & 붕괴 없음, 믹싱 타깃 상승·보합 & 붕괴 없음 → "잘 나옴".
참고: MDLM 쌍둥이([Qwen3-0.6B-diffusion-mdlm-v0.1](https://huggingface.co/dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1))은
원본 JustGRPO 코드 그대로 AR-순서 생성이 작동함을 실측 확인 (temp 1.0에서 GSM8K 4/4 정답).

**예상 일정** (7/25 11:49 실측: 믹싱 4.0분/스텝, 코드 3.8분/스텝 — 2런 동시 기준)
- 7/25 밤 23시 전후: Run 3 v2 (코드) 종료 → 평가 → **1차 판정** → Run 1 v2 (GSM8K) 투입
- 7/26 새벽 06시 전후: Run 2 v2 (믹싱) 종료 → ckpt-200/300 평가 → **판정 확정, 단독 vs 믹싱 비교표** → Run 4 (MATH) 투입
- 7/26 오전 10–11시: Run 1 v2 종료·평가 / 저녁 19시 전후: Run 4 종료·평가
- 7/27 오전: Run 5 (MDLM 논문 방식) 종료·평가 → **최종 매트릭스 + HF 업로드 완료**
- 변동 요인: 학습이 진행되며 생성이 길어지면 스텝이 느려지는 경향(v1 실측) — 믹싱 종료는 ±1–2시간 유동
