# Temporal KV Cache Reuse for LLaVA-OneVision on NExT-QA

이 프로젝트는 `llava-hf/llava-onevision-qwen2-7b-ov-hf`를 대상으로, 연속 프레임 사이의 temporal redundancy를 이용해 **video visual token의 decoder KV cache를 부분 재사용**하는 실험 코드를 제공합니다.

핵심 아이디어:
- 프레임 `t-1`와 `t`의 시각 토큰이 많이 겹친다고 가정합니다.
- 프레임별 visual token 중 픽셀 변화량이 작은 토큰은 **이전 프레임의 decoder KV를 재사용**합니다.
- 변화량이 큰 토큰만 **재계산(recompute)** 합니다.
- baseline(재사용 없음)과 reuse(부분 재사용)를 모두 NExT-QA multiple-choice 정확도로 비교합니다.

## 디렉토리 구조

```text
kv_reuse_nextqa/
├── README.md
├── requirements.txt
├── configs/
│   └── default.yaml
├── scripts/
│   ├── run_eval_baseline.sh
│   ├── run_eval_kvreuse.sh
│   └── compare_results.sh
├── src/
│   ├── run_eval.py
│   ├── hf_patch/
│   │   ├── apply_runtime_patch.py
│   │   └── transformers_llava_onevision_kvreuse.patch
│   ├── kv_reuse/
│   │   ├── cache_ops.py
│   │   ├── pixel_change.py
│   │   ├── prefill.py
│   │   ├── scoring.py
│   │   └── video_prompt.py
│   ├── data/
│   │   ├── nextqa_dataset.py
│   │   └── video_io.py
│   └── eval/
│       └── metrics.py
└── results/
```

## 환경

권장 버전:
- Python 3.10+
- PyTorch 2.3+
- CUDA 환경
- `transformers==4.46.3`

이 코드는 `transformers>=4.45.0`에서 동작하도록 작성했지만, **런타임 monkey patch는 `4.46.x` 기준**으로 맞춰 두었습니다.

## 설치

```bash
cd kv_reuse_nextqa
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

CUDA GPU를 쓸 때는 **PyTorch wheel의 CUDA 버전이 시스템 드라이버와 맞아야** 합니다.
예를 들어 드라이버가 CUDA 12.6 계열이면, 공식 PyTorch index에서 cu126 wheel을 설치하세요.

```bash
pip install --force-reinstall torch==2.6.0 torchvision==0.21.0 \
  --index-url https://download.pytorch.org/whl/cu126
```

GPU가 잡히지 않는 환경에서는 디버깅용으로 `--device cpu`를 줄 수 있습니다.

## 데이터셋

기본값은 `lmms-lab/NExTQA`의 `MC` config입니다.

코드는 다음 두 경우를 모두 처리하려고 시도합니다.
1. dataset row 안에 실제 video column/path가 있는 경우
2. dataset row에 `video`가 없고 `video_id`만 있는 경우 → `--video_root`에서 mp4를 찾음

즉, dataset 저장 형식에 따라 아래 둘 중 하나로 실행하세요.

### A. Hugging Face dataset이 비디오를 직접 제공하는 경우
그냥 실행하면 됩니다.

### B. 로컬에 NExT-QA video를 내려받아 둔 경우
예를 들어:
```bash
export NEXTQA_VIDEO_ROOT=/data/NExT-QA/videos
```

## 실행

### 1) Baseline
```bash
bash scripts/run_eval_baseline.sh
```

또는 CPU 디버깅:
```bash
python src/run_eval.py --mode baseline --config configs/default.yaml --device cpu
```

### 2) KV reuse
```bash
bash scripts/run_eval_kvreuse.sh
```

### 3) 결과 비교
```bash
bash scripts/compare_results.sh
```


## 중요한 구현 포인트

### 1. token-level reuse / recompute 함수 분리
`src/kv_reuse/pixel_change.py`
- 이전 프레임과 현재 프레임의 픽셀 차이를 14x14 grid patch 단위로 평균냅니다.
- LLaVA-OneVision video feature가 프레임당 196개 spatial token으로 pooling된다는 문서 설명을 따릅니다.
- threshold 이상이면 recompute, 미만이면 reuse로 둡니다.
- 마지막 newline token 1개를 붙여 총 197개 토큰 기준 마스크를 반환합니다.

LLaVA-OneVision docs에는 비디오가 프레임당 196 tokens로 pool된다고 명시되어 있습니다. citeturn950196view0turn652483search0

### 2. decoder KV cache reuse
`src/kv_reuse/prefill.py`
- visual token을 프레임별로 순차 prefill합니다.
- frame `t`에서 reuse 대상 토큰은 frame `t-1`의 layer-wise KV slice를 현재 cache 뒤에 직접 append합니다.
- recompute 대상 토큰은 language model forward를 다시 태웁니다.
- 따라서 “이전 frame의 decoder KV cache를 현재 frame에서 reuse”하는 요구를 직접 반영합니다.

### 3. Hugging Face 코드 수정 포함
`src/hf_patch/transformers_llava_onevision_kvreuse.patch`
- 설치된 `transformers`의 `modeling_llava_onevision.py`에 넣을 수 있는 예시 patch입니다.
- 실험 실행은 `apply_runtime_patch.py`의 monkey patch만으로도 가능하게 해두었습니다.

### 4. 정확도 비교
`src/run_eval.py`
- multiple-choice 각 선택지를 continuation log-likelihood로 점수화합니다.
- baseline / reuse 각각 결과 jsonl을 저장합니다.
- `src/eval/metrics.py`가 accuracy를 계산합니다.

## 주의

이 코드는 연구용 프로토타입입니다.
- 수학적으로 exact한 cache-consistency를 보장하지는 않습니다.
- 특히 reused token과 recomputed token이 섞일 때 attention dependency를 완전히 재현하려면 더 정교한 block-level 스케줄링이 필요합니다.
- 여기서는 **실제 decoder KV를 부분 재사용하는 실행 코드**를 우선으로 둡니다.

## 참고 구현 근거

- LLaVA-OneVision은 SigLIP vision encoder + Qwen2 language backbone 구조입니다. citeturn950196view0turn950196view1
- 비디오는 frame별 vision feature를 뽑은 뒤 projector와 pooling을 거쳐 프레임당 196 tokens가 되며, 구현 상 newline embedding을 추가해 실제 placeholder 길이는 프레임당 197이 됩니다. citeturn251274view0turn960315view1
- `prepare_inputs_for_generation`는 첫 iteration에서만 pixel/video inputs를 forward하고, 이후에는 cache만 이용합니다. citeturn470931view0
- Qwen2는 `past_key_values` 기반 incremental decoding을 지원합니다. citeturn950196view2
