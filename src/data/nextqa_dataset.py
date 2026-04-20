from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from datasets import load_dataset

from .video_io import resolve_video_path


@dataclass
class NextQASample:
    sample_id: str
    video_path: Optional[str]
    question: str
    choices: List[str]
    answer_idx: int
    raw_example: Dict


CHOICE_KEYS_CANDIDATES = [
    ["a0", "a1", "a2", "a3", "a4"],
    ["A0", "A1", "A2", "A3", "A4"],
    ["option_0", "option_1", "option_2", "option_3", "option_4"],
    ["choice0", "choice1", "choice2", "choice3", "choice4"],
]

ANSWER_KEYS = ["answer", "label", "gt", "ans"]
QUESTION_KEYS = ["question", "query", "q"]
ID_KEYS = ["qid", "question_id", "id", "uid"]


def _pick_first(example: Dict, keys: List[str]):
    for k in keys:
        if k in example:
            return example[k]
    return None


def _extract_choices(example: Dict) -> List[str]:
    for keys in CHOICE_KEYS_CANDIDATES:
        if all(k in example for k in keys):
            return [str(example[k]) for k in keys]

    if "choices" in example and isinstance(example["choices"], list):
        return [str(x) for x in example["choices"]]

    if "options" in example and isinstance(example["options"], list):
        return [str(x) for x in example["options"]]

    raise KeyError(f"Could not find 5 answer choices. Available keys: {list(example.keys())}")


def _extract_answer_idx(example: Dict) -> int:
    answer = _pick_first(example, ANSWER_KEYS)
    if answer is None:
        raise KeyError(f"Could not find answer key in example keys={list(example.keys())}")
    return int(answer)


def _default_dataset_config_name(dataset_name: str) -> Optional[str]:
    normalized = dataset_name.strip().lower()
    if normalized == "lmms-lab/nextqa":
        return "MC"
    return None


def load_nextqa_samples(
    dataset_name: str,
    split: str,
    dataset_config_name: Optional[str] = None,
    video_root: Optional[str] = None,
    max_samples: Optional[int] = None,
) -> List[NextQASample]:
    config_name = dataset_config_name or _default_dataset_config_name(dataset_name)
    if config_name is None:
        ds = load_dataset(dataset_name, split=split)
    else:
        ds = load_dataset(dataset_name, name=config_name, split=split)
    samples: List[NextQASample] = []

    for idx, ex in enumerate(ds):
        question = _pick_first(ex, QUESTION_KEYS)
        if question is None:
            continue

        try:
            choices = _extract_choices(ex)
            answer_idx = _extract_answer_idx(ex)
        except Exception:
            continue

        sample_id = str(_pick_first(ex, ID_KEYS) or idx)
        video_path = resolve_video_path(ex, video_root=video_root)

        samples.append(
            NextQASample(
                sample_id=sample_id,
                video_path=video_path,
                question=str(question),
                choices=choices,
                answer_idx=answer_idx,
                raw_example=ex,
            )
        )

        if max_samples is not None and len(samples) >= max_samples:
            break

    return samples
