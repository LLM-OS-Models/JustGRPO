# JustGRPO × Tiny Diffusion LM: 0.6B Block-Diffusion 모델의 멀티 도메인 RL 실험

> **한 줄 요약**: ICML 2026 수상작 [JustGRPO](https://arxiv.org/abs/2601.15165)의 RL 레시피(원본: LLaDA-8B 전용)를
> 13배 작은 block-diffusion 모델 [Qwen3-0.6B-diffusion-bd3lm](https://huggingface.co/dllm-hub/Qwen3-0.6B-diffusion-bd3lm-v0.1)에
> 이식하고, **단독 도메인 vs 멀티 도메인(수학+코드 믹싱) RL**의 성능을 4개 벤치마크에서 비교하는 실험 저장소.

- 원본 JustGRPO README: [README_original.md](README_original.md)
- 기술 분석 (하드코딩 제거·설계 결정): [ADAPTATION.md](ADAPTATION.md)
- 벤치마크 결과 원장: [BENCHMARKS.md](BENCHMARKS.md)

---

## 1. 배경: 이 실험이 서 있는 세 갈래 연구

**① JustGRPO — "diffusion LM은 AR 순서로 RL 학습하는 게 낫다"**
([arXiv:2601.15165](https://arxiv.org/html/2601.15165v4), ICML 2026 Outstanding Paper)
Diffusion LM의 임의 순서 생성은 오히려 추론 잠재력(Pass@k)을 깎아먹으므로, RL 학습만큼은
표준 GRPO를 AR 순서로 돌리면 된다는 논문. LLaDA-8B-Instruct에서의 결과 (gen length 256):

| 벤치마크 | LLaDA-8B 베이스 | + JustGRPO | 향상 |
|---|---:|---:|---:|
| GSM8K | 78.6 | **89.1** | +10.5 |
| MATH-500 | 30.4 | **45.1** | +14.7 |
| HumanEval | 40.5 | **49.4** | +8.9 |
| MBPP | 40.7 | **52.4** | +11.7 |

**② dLLM / Tiny-A2D — 우리가 학습할 베이스 모델의 출처**
([arXiv:2602.22661](https://arxiv.org/html/2602.22661), [GitHub](https://github.com/ZHZisZZ/dllm))
Qwen3-0.6B를 SFT만으로 block diffusion([BD3LM, arXiv:2503.09573](https://arxiv.org/abs/2503.09573)) 모델로
변환한 초소형 dLLM. 블록(32토큰) 단위로는 AR, 블록 내부는 확산으로 생성.

**③ 멀티 도메인 RLVR — 2026년 프런티어 모델들의 표준 레시피**
단일 태스크 RL은 이제 소수파다. 참고한 최신 사례:
- [Nemotron 3 Super](https://arxiv.org/abs/2604.12374): **21개 환경 / 37개 데이터셋 동시** RLVR ([공식 레시피](https://github.com/NVIDIA-NeMo/Nemotron/blob/main/docs/nemotron/super3/README.md), [RLVR 단계 문서](https://docs.nvidia.com/nemotron/latest/nemotron/super3/rl/rlvr.html))
- [Kimi K2](https://arxiv.org/abs/2507.20534): Gym식 확장 프레임워크로 검증가능 보상 태스크를 대규모 혼합
- [GLM-5](https://arxiv.org/pdf/2602.15763): SFT → 추론 RL → 에이전틱 RL → 일반 RL 다단계 + 스테이지 간 증류
- 학술 근거: [Can One Domain Help Others? (arXiv:2507.17512)](https://arxiv.org/abs/2507.17512) — GRPO로 수학/코드/퍼즐
  단독 vs 혼합을 체계 비교. 혼합은 대체로 이득이지만 도메인 간 간섭도 실재.
  후속: [To Mix or To Merge (arXiv:2602.12566)](https://arxiv.org/pdf/2602.12566),
  [멀티 도메인 커리큘럼 RLVR (arXiv:2606.25178)](https://arxiv.org/pdf/2606.25178)

**우리의 질문**: 이 트렌드가 0.6B급 diffusion LM에서도 성립하는가? — diffusion LM + 멀티 도메인 RLVR 조합은
아직 발표된 사례가 없다.

---

## 2. 실험 설계

**베이스 모델 재현 검증 (완료)** — 학습 전, dLLM 논문 수치를 우리 환경에서 재현해 평가 파이프라인부터 신뢰 확보:

| 벤치마크 | dLLM 논문 | 우리 재현 | 판정 |
|---|---:|---:|---|
| GSM8K | 46.3 | **45.72** ±1.37 | 오차범위 내 ✅ |
| MATH (minerva) | 12.9 | **13.60** ±0.47 | 근접 재현 (+0.7) ✅ |
| HumanEval | 46.3 | **46.95** ±3.91 | 오차범위 내 ✅ |
| MBPP | 38.2 | **38.20** ±2.18 | 정확히 일치 ✅ |

**4개 전부 재현 성공** — 평가 파이프라인 신뢰 확보 완료.

**학습 런 매트릭스** — 전부 LoRA(r=128, α=64, lr 5e-5), 재현 가능하도록 런별 스크립트 고정:

런 번호 = 실제 실행 순서. **수학 단독 기준선을 먼저 만들고 → 믹싱과 비교 → 그 다음 코드**
(믹싱 결과가 핵심 관심사이고, 비교가 성립하려면 단독 기준선이 선행되어야 함).

| 런 | 데이터 | 스텝 | 스크립트 | 상태 |
|---|---|---:|---|---|
| Run 1 | GSM8K(수학) 단독 | 200 | [`scripts/run1_gsm8k_lora.sh`](scripts/run1_gsm8k_lora.sh) | 🔥 **학습 중** (7/24 17:10~) |
| Run 2 | **믹싱** (gsm8k+math+code 라운드로빈) | 300 | [`scripts/run2_mixed_lora.sh`](scripts/run2_mixed_lora.sh) | 🔥 **학습 중** (7/24 18:35~, Run 1과 동시) |
| Run 3 | 코드(AceCode-Hard 21K) 단독 | 200 | [`scripts/run3_code_lora.sh`](scripts/run3_code_lora.sh) | 대기 (3순위) |
| Run 4 (옵션) | MATH-500 단독 | 200 | [`scripts/run4_math500_lora.sh`](scripts/run4_math500_lora.sh) | 대기 |

**학습량 설계**: 스텝당 프롬프트 8개 기준 단독 런은 해당 도메인 1,600개, 믹싱 런은 총 2,400개(도메인당 800개).
체크포인트가 10스텝마다 저장되므로 믹싱 런에서 **ckpt-200(단독 런과 총 컴퓨트 동일 비교)** 과
**ckpt-300(도메인 노출 보정)** 둘 다 평가해 두 관점의 비교표를 만든다.

각 런 종료 후: [`scripts/merge_lora.py`](scripts/merge_lora.py)로 어댑터 병합 →
[`scripts/eval_ckpt.sh`](scripts/eval_ckpt.sh)로 4개 벤치마크 평가 → 여기와 BENCHMARKS.md에 기록 →
HF org [`LLM-OS-Models2`](https://huggingface.co/LLM-OS-Models2)에 업로드.

---

## 3. 방법: 원본 JustGRPO에서 바뀐 것

원본 코드는 LLaDA-8B 하드코딩(mask 126336, full-bidirectional attention 가정)이라 그대로 안 돌아간다.
검증 과정에서 나온 핵심 변경 세 가지 (상세: [ADAPTATION.md](ADAPTATION.md)):

1. **Rollout = 네이티브 블록 확산 샘플링.** 이 0.6B 모델은 AR(블록=1) 생성이 붕괴한다
   (첫 위치 EOS 확률 49% — SFT 시 EOS 블록 패딩의 아티팩트, 이후 숫자 반복 루프. 전부 실측).
   같은 GSM8K 문제를 블록 확산 샘플링은 8/8 정답 → rollout은 dllm의 `BD3LMSampler`(block 32) 사용.
2. **손실 = AR one-pass logprob.** BD3LM의 `[x0 ∥ xt]` 학습 마스크 트릭(블록=1 특수화)으로
   원본의 **256회 forward 루프를 1회 forward로 대체**. fp32에서 순차 루프 대비 최대 오차 2e-5로
   동일성 검증(`tests/test_adaptation.py`). GRPO ratio는 on-policy 첫 업데이트에서 1이므로
   advantage-가중 AR 학습으로 성립 — "학습은 AR 순서" 철학 유지.
3. **멀티 도메인 믹싱 로더** (`data/mixed.py`). 배치 단위 라운드로빈 + 도메인별 reward 라우팅.
   GRPO가 그룹 단위로 advantage를 정규화하므로 수학(±1)/코드(0~2) 보상 스케일 차이는 자동 흡수 —
   Nemotron식 멀티 환경 RLVR과 같은 원리.

End-to-end 스모크 테스트: 실제 GSM8K 문제에서 rewards `[-1,1,1,1,-1,1,-1,1]` (5/8 정답),
advantage/LoRA gradient 정상 확인.

---

## 4. 재현 가이드

```bash
# 환경: 1× H100 80GB, uv venv 2개
#  - 학습: /home/ubuntu/justgrpo-venv (torch 2.13, transformers 4.57, peft, dllm editable)
#  - 평가: /home/ubuntu/dllm/.venv (+ lm-evaluation-harness)
# 모델/데이터: ~/data/models/Qwen3-0.6B-diffusion-bd3lm-v0.1, HF_HOME=~/data/hf_cache

bash scripts/run1_gsm8k_lora.sh                       # 학습 (Run 1)
python scripts/merge_lora.py \
  --adapter ~/data/runs/run1-gsm8k-lora/ckpt-000200 \
  --out ~/data/models/run1-gsm8k-lora-merged          # LoRA 병합
bash scripts/eval_ckpt.sh ~/data/models/run1-gsm8k-lora-merged run1-gsm8k-lora  # 4태스크 평가
```

주의: vLLM 사용 불가(확산 디노이징 루프 ↔ AR 전용 엔진 비호환). 평가 배치는 16 권장
(vocab 152k 로짓 텐서가 커서 배치 64는 OOM — 실측).

---

## 5. 진행 로그 & 예상 일정 (2026-07-24)

- 15:28 ✅ 모델 + 데이터셋(GSM8K/MATH-500/AceCode-Hard 21K) 다운로드
- 16:03 ✅ 베이스 4태스크 평가 시작 (배치 64 OOM → 16으로 재실행)
- 16:29 ✅ GSM8K 45.72 / HumanEval 46.95 / MBPP 38.20 — 3개 재현 성공
- 16:30 ✅ 적용 코드 완성 + 동등성/스모크 테스트 통과, AR 붕괴 발견 및 우회 설계
- 16:50 ✅ 믹싱 로더·런 스크립트·병합/평가 스크립트 완성
- 17:07 ✅ MATH 평가 완료 (13.60) → **베이스 4태스크 재현 전부 성공**
- 17:10 ✅ **Run 1 (GSM8K 단독 LoRA, 200스텝) 학습 시작** — 스텝 20에서 평균 보상 -0.08(베이스 기대치) → +0.39 상승 확인
- 18:35 ✅ **Run 2 (믹싱, 300스텝) 동시 시작** — GPU 사용률 32%·VRAM 20GB로 여유가 커서 병렬 실행
  (rollout이 256회 순차 디노이징 + CPU 채점이라 단일 런으로는 H100을 못 채움; 동시 실행으로 처리량 ~2배)
- 18:38 ✅ 동시 실행 후 GPU 99% 도달 (VRAM 40GB/80GB) — 유휴 시간 제거 확인

**운영 원칙**: GPU를 실시간 감시하며 빈 슬롯이 생기는 즉시 다음 작업 투입
(런 종료 → 즉시 병합·평가 → 남는 GPU에 다음 런 시작).

예상 일정 (18:38 기준, 30분 단위로 실측 갱신):
- **7/25 09~10시** Run 1 종료 → 즉시 병합·4태스크 평가, 동시에 Run 3 (코드) 학습 시작
- **7/25 밤~7/26 새벽** Run 2 종료 → ckpt-200/300 평가 → **수학 단독 vs 믹싱 비교표 1차 완성**
- **7/26 낮** Run 3 종료·평가
- **7/26 밤 목표** 전체 비교표 완성, 모델 HF([`LLM-OS-Models2`](https://huggingface.co/LLM-OS-Models2)) 업로드
