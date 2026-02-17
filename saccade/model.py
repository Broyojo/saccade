"""
Qwen3-VL input formatting for gaze policy SFT.

We treat one trial as an interleaved chat:
  user:      <patch image at fixation t>
  assistant: "<x_{t+1}> <y_{t+1}>"
and on the final turn:
  assistant: "FOUND" | "NOT_FOUND"  (optionally boxed)

The key pieces you need later for SFTTrainer/Trainer are:
  1) `messages` list in the right multimodal chat format
  2) `processor.apply_chat_template(messages, ...)` to get the prompt string
  3) `processor(text=..., images=..., return_tensors="pt")` to get model inputs

This file intentionally does not include training code; it just makes the
message format + processor plumbing explicit and testable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

QWEN3_VL_MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"

# Our dataset currently emits ImageNet-normalized patch tensors.
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)

FINAL_FOUND = "FOUND"
FINAL_NOT_FOUND = "NOT_FOUND"


def patch_tensor_to_pil(patch_chw: torch.Tensor) -> Image.Image:
    """
    Convert a dataset patch tensor to PIL for the Qwen processor.

    Expected input: float [3,H,W] normalized with ImageNet mean/std (like our dataset).
    """
    p = patch_chw.detach().cpu()
    if p.dtype == torch.uint8:
        arr = p.permute(1, 2, 0).numpy()
        return Image.fromarray(arr)

    # Unnormalize ImageNet -> [0,1] and convert to uint8 RGB.
    p = (p * IMAGENET_STD + IMAGENET_MEAN).clamp(0.0, 1.0)
    arr = (p.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(arr)


def format_final_answer(*, found: bool, boxed: bool = False) -> str:
    if boxed:
        # Keep it machine-parsable: box the same tokens we parse.
        return f"\\boxed{{{FINAL_FOUND if found else FINAL_NOT_FOUND}}}"
    return FINAL_FOUND if found else FINAL_NOT_FOUND


def build_rollout_messages(
    patch_images: Sequence[Image.Image],
    fixation_xy: Sequence[tuple[int, int]],
    *,
    target: str,
    found: bool | None = None,
    final_text: str | None = None,
    boxed_final: bool = False,
    system_prompt: str | None = None,
) -> list[dict[str, Any]]:
    """
    Build a full rollout chat:

      user:      patch at fixation t
      assistant: x_{t+1} y_{t+1}      for t < T-1
      assistant: FOUND/NOT_FOUND      for t == T-1
    """
    if len(patch_images) != len(fixation_xy):
        raise ValueError(
            "patch_images and fixation_xy must have the same length (T fixations)."
        )
    if len(patch_images) < 2:
        raise ValueError("Need at least 2 fixations to produce one action.")

    if final_text is None:
        final_text = format_final_answer(found=bool(found), boxed=boxed_final)

    if system_prompt is None:
        system_prompt = (
            "You are a gaze policy for visual search.\n"
            f"Target category: {target}\n"
            "On each turn, reply with either:\n"
            '  - the next fixation as two integers: "x y"\n'
            f"  - or {FINAL_FOUND}/{FINAL_NOT_FOUND} if you want to stop.\n"
            "Do not output anything else."
        )

    T = len(patch_images)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]}
    ]

    for t in range(T):
        messages.append(
            {"role": "user", "content": [{"type": "image", "image": patch_images[t]}]}
        )
        if t < T - 1:
            x, y = fixation_xy[t + 1]
            messages.append(
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": f"{int(x)} {int(y)}"}],
                }
            )
        else:
            messages.append(
                {"role": "assistant", "content": [{"type": "text", "text": final_text}]}
            )

    return messages


def build_interleaved_messages(
    patch_images: Sequence[Image.Image],
    next_fixation_xy: Sequence[tuple[int, int]],
    *,
    target: str,
    system_prompt: str | None = None,
) -> list[dict[str, Any]]:
    """
    Build the multi-turn (image -> "x y") message list for one trial.

    `patch_images[i]` is the patch at fixation i, and `next_fixation_xy[i]` is the
    coordinate of fixation i+1. So len(next_fixation_xy) should be <= len(patch_images),
    usually exactly len(patch_images) - 1.
    """
    if system_prompt is None:
        system_prompt = (
            "You are a gaze policy for visual search.\n"
            f"Target category: {target}\n"
            "Reply with the next fixation as two integers: \"x y\".\n"
            "Do not output anything else."
        )

    n = min(len(patch_images), len(next_fixation_xy))
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]}
    ]

    for t in range(n):
        x, y = next_fixation_xy[t]
        messages.append({"role": "user", "content": [{"type": "image", "image": patch_images[t]}]})
        messages.append(
            {"role": "assistant", "content": [{"type": "text", "text": f"{int(x)} {int(y)}"}]}
        )

    return messages


def extract_images(messages: Iterable[dict[str, Any]]) -> list[Image.Image]:
    """Return images in the exact order they appear in the message list."""
    out: list[Image.Image] = []
    for m in messages:
        content = m.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, dict) and part.get("type") == "image":
                out.append(part["image"])
    return out


@dataclass(frozen=True)
class Qwen3VLBatch:
    """Convenience wrapper for inputs + the rendered chat string."""

    inputs: Any  # transformers.BatchFeature
    chat: str


def encode_messages(
    processor,
    messages: list[dict[str, Any]],
    *,
    add_generation_prompt: bool = False,
    device: torch.device | str | None = None,
) -> Qwen3VLBatch:
    """
    Render chat -> tokenize -> build vision tensors.

    If you pass `add_generation_prompt=True`, the chat string will end with the
    assistant role marker and no assistant content, which is convenient for `.generate()`.
    """
    chat = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=add_generation_prompt
    )
    images = extract_images(messages)
    inputs = processor(
        text=chat,
        images=images if images else None,
        padding=True,
        return_tensors="pt",
    )
    if device is not None:
        inputs = inputs.to(device)
    return Qwen3VLBatch(inputs=inputs, chat=chat)


def load_processor(model_id: str = QWEN3_VL_MODEL_ID):
    return AutoProcessor.from_pretrained(model_id)


def load_model(
    model_id: str = QWEN3_VL_MODEL_ID,
    *,
    torch_dtype: torch.dtype | str = "auto",
):
    # Keep this simple: no device_map here (requires accelerate). Move to device manually.
    return Qwen3VLForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch_dtype)


if __name__ == "__main__":
    # Minimal, non-training demo: build one trial's rollout messages and encode them.
    from saccade.coco_search18_dataset import COCOSearch18Dataset

    ds = COCOSearch18Dataset("data/coco_search18", split="train", condition="TP")
    item = ds[0]

    # Full rollout: patches 0..T-1 as inputs, next fixation coords 1..T-1 as actions,
    # and final FOUND/NOT_FOUND as the terminal message.
    patches = [patch_tensor_to_pil(p) for p in item["patches"]]
    fix_xy = [tuple(xy.tolist()) for xy in item["fixation_xy"]]
    messages = build_rollout_messages(
        patches,
        fix_xy,
        target=item["target"],
        found=bool(item["found"]),
        boxed_final=True,
    )

    processor = load_processor()
    batch = encode_messages(processor, messages)
    print(batch.chat[:500] + ("..." if len(batch.chat) > 500 else ""))
    for k, v in batch.inputs.items():
        if hasattr(v, "shape"):
            print(k, tuple(v.shape), v.dtype)
