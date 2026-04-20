from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import torch
import torch.nn.functional as F

from .cache_ops import clone_cache


@dataclass
class ChoiceScore:
    text: str
    token_ids: List[int]
    score: float
    avg_score: float


@torch.inference_mode()
def score_choice_continuation(
    model,
    tokenizer,
    past_key_values,
    initial_logits: torch.Tensor,
    choice_text: str,
    device: torch.device,
    normalize_by_length: bool = True,
) -> ChoiceScore:
    """
    Score a choice string as continuation log-probability conditioned on the prompt cache.
    """
    cache = clone_cache(past_key_values)
    enc = tokenizer(choice_text, add_special_tokens=False, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)

    total_logprob = 0.0
    token_ids = input_ids[0].tolist()
    next_logits = initial_logits.to(device)

    for tok in token_ids:
        logprobs = F.log_softmax(next_logits, dim=-1)
        total_logprob += float(logprobs[0, tok].item())

        step_ids = torch.tensor([[tok]], device=device, dtype=torch.long)
        out = model.language_model(
            input_ids=step_ids,
            past_key_values=cache,
            use_cache=True,
            return_dict=True,
        )
        cache = out.past_key_values
        next_logits = out.logits[:, -1, :]

    avg = total_logprob / max(1, len(token_ids)) if normalize_by_length else total_logprob
    return ChoiceScore(
        text=choice_text,
        token_ids=token_ids,
        score=total_logprob,
        avg_score=avg,
    )


@torch.inference_mode()
def rank_multiple_choices(
    model,
    tokenizer,
    past_key_values,
    initial_logits: torch.Tensor,
    choice_letter_prefixes: Sequence[Tuple[str, str]],
    device: torch.device,
    normalize_by_length: bool = True,
):
    """
    choice_letter_prefixes: [("A", " A"), ("B", " B"), ...]
    We intentionally score the option letters as continuation because the prompt instructs the model
    to answer with only the option letter.
    """
    scored = []
    for letter, continuation_text in choice_letter_prefixes:
        s = score_choice_continuation(
            model=model,
            tokenizer=tokenizer,
            past_key_values=past_key_values,
            initial_logits=initial_logits,
            choice_text=continuation_text,
            device=device,
            normalize_by_length=normalize_by_length,
        )
        scored.append((letter, s))

    best_letter, best_obj = max(scored, key=lambda x: x[1].avg_score)
    return best_letter, scored
