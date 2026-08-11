"""Automatic Grounding DINO prompt tuning for a meal's foods.

Why this exists
---------------
Grounding DINO is very sensitive to *how* a food is described, and the right
wording is not guessable from the food's name. Hand-tuned examples we measured:

    "orange chicken piece"            -> one box over the whole cluster
    "glazed fried chicken piece"      -> every piece, conf 0.4-0.5

    "small cut up strawberry piece"   -> 1 of 10 berries, conf 0.31
    "fresh strawberry"                -> 10 of 10,        conf 0.72

    "small cut up pancake piece" + "small cut up sausage piece"
                                      -> 23 pancake, 0 sausage (total collapse)
    "golden brown pancake piece" + "dark brown cylindrical sausage link"
                                      -> 16 pancake + 8 sausage, correctly split

Two failure modes show up over and over:

1. *Wording mismatch* -- the phrase describes something the food is not
   ("small cut up ... piece" for whole strawberries), so nothing clears the box
   threshold and only the single strongest blob survives.
2. *Multi-class collapse* -- with two foods on one plate, predict_with_classes
   joins the phrases into ONE caption and assigns each box to whichever phrase
   its tokens match best. Two phrases that share boilerplate are nearly
   indistinguishable, so every box lands on one class and the other gets zero.

This module searches for good phrases automatically, once per meal, on the
picture taken right after the plate is set on the table. It needs no ground
truth: it scores a candidate wording by how well the resulting detections hold
together (see `score_detection`).

Design notes
------------
* The search is *joint*. Phrases are always evaluated together in one
  predict_with_classes call, exactly as production runs them, because collapse
  is a property of the phrase *set*, not of any single phrase.
* The search is coordinate ascent (vary one food's phrase, hold the rest), which
  costs K*N*passes model calls instead of the N**K of a full grid.
* Nothing here changes detection behaviour on its own. The tuner produces a
  {food_label: phrase} mapping; `BiteAcquisitionInference.PROMPT_OVERRIDES`
  starts empty, so unless a caller installs a result the hardcoded prompts are
  used exactly as before.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np

try:
    import cv2
except ModuleNotFoundError:  # pragma: no cover - cv2 is present in deployment
    cv2 = None

try:
    import torch
    import torchvision
    _TORCH = True
except ModuleNotFoundError:  # pragma: no cover
    _TORCH = False


# --------------------------------------------------------------------------
# Candidate wordings
# --------------------------------------------------------------------------

# Fallback templates, used when no LLM is available. Ordered roughly by how
# often each shape won in our manual sweeps. "{name}" is the raw food label.
#
# The bare name and "<name> piece" are here because they *win* for whole foods
# (strawberry); the "small cut up" form is included only so the current
# production default stays in the running and can be beaten on merit.
SOLID_TEMPLATES: tuple[str, ...] = (
    "{name}",
    "{name} piece",
    "fresh {name}",
    "cooked {name} piece",
    "individual {name} piece",
    "small cut up {name} piece",
)

DIP_TEMPLATES: tuple[str, ...] = (
    "{name}",
    "{name} dip",
    "bowl of {name}",
    "{name} in a small container",
)

# Asking the model for these specific things is what makes the proposals useful:
# appearance over dish-name, and phrases that do not share wording across foods
# (the direct fix for multi-class collapse).
_LLM_INSTRUCTIONS = """\
You are helping an open-vocabulary object detector (Grounding DINO) find \
individual pieces of food on a plate, so a robot can pick up ONE piece at a time.

The plate in the image contains these foods: {foods}.

For EACH food, propose {n} short noun phrases to use as the detector's text query.

Rules that matter (learned from failures):
- Describe what the food LOOKS LIKE (colour, shape, texture, preparation), not \
what the dish is called. Restaurant/dish names ground poorly.
- Match the food's ACTUAL form in the image. Do not say "cut up" or "piece" for \
food that is whole; do say it when the food really is in pieces.
- The phrases for DIFFERENT foods must not share wording. They are concatenated \
into one caption, and near-identical phrases make the detector assign every box \
to a single food. Give each food its own distinctive colour/shape words.
- Each phrase should name ONE piece, not the pile.
- Keep each phrase under 6 words.

Reply with ONLY a JSON object mapping each food name exactly as given to a list \
of {n} phrase strings. No other text.
"""


def _template_candidates(name: str, category: str, extra: Sequence[str] = ()) -> list[str]:
    """Candidate phrases for one food, de-duplicated, order preserved."""
    templates = DIP_TEMPLATES if category == "dip" else SOLID_TEMPLATES
    out: list[str] = []
    for phrase in list(extra) + [t.format(name=name) for t in templates]:
        phrase = " ".join(str(phrase).split()).strip()
        if phrase and phrase.lower() not in {o.lower() for o in out}:
            out.append(phrase)
    return out


def phrases_conflict(phrases: Sequence[str]) -> bool:
    """True if this phrase set is unsafe to hand to detect_items.

    Two hazards, both fatal downstream rather than merely suboptimal:
    * duplicates -- predict_with_classes cannot tell the foods apart at all;
    * one phrase contained in another -- detect_items maps labels back to food
      names with a plain str.replace per entry, so "strawberry" would also
      rewrite the inside of "fresh strawberry" and mangle the other food's label.
    """
    lowered = [p.lower().strip() for p in phrases]
    if len(set(lowered)) != len(lowered):
        return True
    return any(a != b and a in b for a in lowered for b in lowered)


def propose_with_llm(
    llm_call: Callable[[str, Any], str],
    image,
    food_names: Sequence[str],
    n_per_food: int = 4,
) -> dict[str, list[str]]:
    """Ask a vision LLM for candidate phrases, looking at the actual plate.

    ``llm_call(prompt, image) -> str`` must return the model's raw text. Any
    failure (no API key, malformed reply, network) returns {} so the caller
    silently falls back to templates -- prompt tuning must never break a meal.
    """
    prompt = _LLM_INSTRUCTIONS.format(foods=", ".join(food_names), n=n_per_food)
    try:
        raw = llm_call(prompt, image)
    except Exception as exc:  # noqa: BLE001
        print(f"[prompt_autotune] LLM proposal failed ({exc}); using templates only")
        return {}
    if not raw:
        return {}
    # Models sometimes wrap JSON in prose or a ```json fence; take the outermost object.
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        print("[prompt_autotune] LLM reply had no JSON object; using templates only")
        return {}
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        print("[prompt_autotune] LLM reply was not valid JSON; using templates only")
        return {}
    out: dict[str, list[str]] = {}
    for food in food_names:
        vals = parsed.get(food)
        if isinstance(vals, str):
            vals = [vals]
        if isinstance(vals, list):
            phrases = [" ".join(str(v).split()) for v in vals if str(v).strip()]
            if phrases:
                out[food] = phrases
    return out


# --------------------------------------------------------------------------
# Scoring a candidate wording, without ground truth
# --------------------------------------------------------------------------

# A box bigger than plate_area/UNION_AREA_DIVISOR is treated as a "union" box --
# one span covering a whole cluster rather than a single piece. Matches the
# MAX_AREA_THRESHOLD convention already used in detect_items.
UNION_AREA_DIVISOR = 15.0

# Where the per-food recall term saturates: at the start of a meal each food on
# the plate is present in roughly this many pieces (measured on the logged
# frames: strawberry 15, pancake 16, sausage 8, steak 20, mozzarella 7).
# Finding this many of a food is "found that food"; finding more is not scored
# as better, so the score cannot be farmed by over-segmenting.
PIECES_PER_FOOD = 8


def _median_lab(crop, box) -> np.ndarray:
    """Median CIELAB colour inside a box; the appearance signature of a piece."""
    x1, y1, x2, y2 = (max(0, int(v)) for v in box)
    patch = crop[y1:y2, x1:x2]
    if patch.size == 0:
        return np.zeros(3, dtype=np.float32)
    lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB).astype(np.float32)
    return np.median(lab.reshape(-1, 3), axis=0)


def _colour_separability(per_class_lab: Sequence[np.ndarray]) -> float:
    """How distinct the classes look, 0..1 (0 when there is only one class).

    Fisher-style: distance between class colour means over the average spread
    within classes. This is the signal that catches a *wrong* class assignment
    even when the box count looks plausible -- if the boxes labelled "sausage"
    are the same colour as the ones labelled "pancake", the split is fiction.
    """
    groups = [g for g in per_class_lab if len(g) > 0]
    if len(groups) < 2:
        return 0.0
    means = [g.mean(axis=0) for g in groups]
    # Average within-class spread (a single box has no spread; treat as 0).
    spreads = [float(np.mean(np.linalg.norm(g - m, axis=1))) if len(g) > 1 else 0.0
               for g, m in zip(groups, means)]
    within = float(np.mean(spreads))
    between = min(float(np.linalg.norm(a - b))
                  for i, a in enumerate(means) for b in means[i + 1:])
    ratio = between / (within + 1e-6)
    return float(np.clip(ratio / 4.0, 0.0, 1.0))


def _size_consistency(areas: Sequence[float]) -> float:
    """1 for uniformly sized boxes, falling to 0 as sizes scatter.

    Pieces of one food are roughly the same size, so a wild spread means the
    wording is picking up a mixture of single pieces and clumps.
    """
    arr = np.asarray([a for a in areas if a > 0], dtype=np.float64)
    if arr.size < 2:
        return 1.0
    cv_ = float(arr.std() / (arr.mean() + 1e-9))
    return float(np.clip(1.0 - cv_, 0.0, 1.0))


def score_detection(
    boxes_per_class: Sequence[np.ndarray],
    confs_per_class: Sequence[np.ndarray],
    crop,
    plate_area: float,
    n_union_boxes: int = 0,
    n_raw_boxes: int = 0,
) -> tuple[float, dict]:
    """Score one candidate wording. Higher is better; -1 means unusable.

    No ground truth is involved. The components, in order of how much they
    mattered in the cases we validated against:

    ``conf``   mean confidence of the surviving boxes. The single best predictor
               of a good phrase -- a well-grounded wording clears the threshold
               with margin instead of scraping it.
    ``recall`` how much of the WORST-found food was found, saturating per food.
               Without any recall term the other terms are all maximised by
               finding *few* very clean boxes -- on a real steak plate an early
               version of this function preferred a wording that found 1 piece
               over one that found 20, because the single box was tidier. Taking
               the worst food rather than the total is the second half of that
               lesson: see the comment at the computation below.
    ``sep``    colour separability between classes (multi-class only), which is
               what distinguishes a real split from a mislabelled one.
    ``size``   size consistency, penalising a mix of pieces and clumps.
    ``union``  fraction of raw boxes that spanned a whole cluster; a wording
               that needs union boxes is describing the pile, not a piece.

    A class with zero boxes is disqualifying (score -1): at the moment we tune,
    every food of the meal is still on the plate, so an empty class means the
    phrase set collapsed. Total failures are still ordered among themselves by
    box count so the search can climb out of them.
    """
    counts = [len(b) for b in boxes_per_class]
    if not counts or min(counts) == 0:
        return -1.0 + 1e-3 * float(sum(counts)), {
            "reason": "class_collapse", "counts": counts,
        }

    all_conf = np.concatenate([np.asarray(c, dtype=np.float64) for c in confs_per_class])
    conf = float(all_conf.mean())

    per_class_lab = [np.stack([_median_lab(crop, b) for b in boxes]) if len(boxes) else
                     np.empty((0, 3), dtype=np.float32) for boxes in boxes_per_class]
    sep = _colour_separability(per_class_lab)

    areas = [float((b[2] - b[0]) * (b[3] - b[1])) for boxes in boxes_per_class for b in boxes]
    size = _size_consistency(areas)

    union_rate = (n_union_boxes / n_raw_boxes) if n_raw_boxes else 0.0

    # Per-food, and the WORST food decides -- every food on the plate has to be
    # found. Scoring recall on the total instead lets one abundant food mask
    # another's collapse: on a real pancake+sausage plate, 16 pancakes alone
    # saturated a total-based term, so dropping sausage from 8 pieces to 3 cost
    # the score nothing while gaining on separability. Saturating per food also
    # keeps over-segmentation from farming the term.
    recall = min(min(1.0, c / float(PIECES_PER_FOOD)) for c in counts)

    score = (1.0 * conf + 0.6 * recall + 0.35 * sep + 0.15 * size
             - 0.5 * union_rate)
    return float(score), {
        "counts": counts, "conf": round(conf, 4), "recall": round(recall, 4),
        "sep": round(sep, 4), "size": round(size, 4),
        "union_rate": round(union_rate, 4), "plate_area": plate_area,
    }


# --------------------------------------------------------------------------
# The tuner
# --------------------------------------------------------------------------

class PromptAutoTuner:
    """Searches for the phrase set that grounds a meal's foods best.

    Usage (once per meal, on the plate picture):

        tuner = PromptAutoTuner(dino_model, box_threshold=0.30,
                                text_threshold=0.20, nms_threshold=0.40)
        best = tuner.tune(image, plate_bounds, ["pancake", "sausage"],
                          ["solid", "solid"])
        inference_server.PROMPT_OVERRIDES.update(best)
    """

    def __init__(
        self,
        grounding_dino_model,
        box_threshold: float = 0.30,
        text_threshold: float = 0.20,
        nms_threshold: float = 0.40,
        llm_call: Callable[[str, Any], str] | None = None,
        max_candidates_per_food: int = 6,
        passes: int = 2,
    ) -> None:
        self.model = grounding_dino_model
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.nms_threshold = nms_threshold
        self.llm_call = llm_call
        self.max_candidates_per_food = max_candidates_per_food
        self.passes = passes

    # -- one joint evaluation -------------------------------------------------

    def evaluate(self, crop, phrases: Sequence[str]) -> tuple[float, dict]:
        """Run one joint detection for `phrases` and score it."""
        det = self.model.predict_with_classes(
            image=crop, classes=list(phrases),
            box_threshold=self.box_threshold, text_threshold=self.text_threshold,
        )
        n_raw = len(det.xyxy)
        if n_raw == 0:
            return -1.0, {"reason": "no_detections", "counts": [0] * len(phrases)}

        keep = torchvision.ops.nms(
            torch.from_numpy(det.xyxy), torch.from_numpy(det.confidence), self.nms_threshold
        ).numpy().tolist()
        xyxy, conf, cid = det.xyxy[keep], det.confidence[keep], det.class_id[keep]

        plate_area = float(crop.shape[0] * crop.shape[1])
        max_area = plate_area / UNION_AREA_DIVISOR
        areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
        small = areas <= max_area
        n_union = int((~small).sum())
        xyxy, conf, cid = xyxy[small], conf[small], cid[small]

        boxes_per_class, confs_per_class = [], []
        for k in range(len(phrases)):
            sel = np.array([c is not None and int(c) == k for c in cid], dtype=bool)
            boxes_per_class.append(xyxy[sel])
            confs_per_class.append(conf[sel])

        return score_detection(boxes_per_class, confs_per_class, crop, plate_area,
                               n_union_boxes=n_union, n_raw_boxes=n_raw)

    # -- the search -----------------------------------------------------------

    def tune(
        self,
        image,
        plate_bounds: Sequence[int],
        food_names: Sequence[str],
        food_categories: Sequence[str],
        current_phrases: Sequence[str] | None = None,
        report: Callable[[str], None] | None = None,
    ) -> dict[str, str]:
        """Return {food_name: best phrase} for the meal's SOLID foods.

        `plate_bounds` is [x, y, w, h] as detect_items computes it. Solids are
        searched on the plate crop, matching production. `current_phrases`, when
        given, seeds the search with what production would use today, so the
        result can only be a phrase that beat the status quo on this plate.

        Dips are deliberately left alone. detect_items runs them over the whole
        camera frame (a sauce usually sits in its own container beside the
        plate), so scoring them against the plate crop would model the wrong
        thing entirely -- an off-plate dip is simply not in this image. They also
        have not needed tuning: "<name> dip" grounds a sauce container at
        0.6-0.7 already. Tuning them would be a separate full-frame pass.
        """
        solid_idx = [i for i, c in enumerate(food_categories) if c != "dip"]
        if not solid_idx:
            return {}
        food_names = [food_names[i] for i in solid_idx]
        food_categories = [food_categories[i] for i in solid_idx]
        if current_phrases is not None:
            current_phrases = [current_phrases[i] for i in solid_idx]

        x, y, w, h = plate_bounds
        crop = image[y:y + h, x:x + w].copy()

        # 1. candidates: LLM proposals (if available) ahead of the templates
        llm_extra: dict[str, list[str]] = {}
        if self.llm_call is not None:
            if report:
                report("Looking at the plate to describe the food")
            llm_extra = propose_with_llm(self.llm_call, image, food_names)

        candidates: dict[str, list[str]] = {}
        for name, category in zip(food_names, food_categories):
            cand = _template_candidates(name, category, extra=llm_extra.get(name, ()))
            candidates[name] = cand[: self.max_candidates_per_food]

        # 2. seed with what production uses now, so we never regress
        if current_phrases is not None and len(current_phrases) == len(food_names):
            best = list(current_phrases)
            for name, phrase in zip(food_names, current_phrases):
                if phrase.lower() not in {c.lower() for c in candidates[name]}:
                    candidates[name].insert(0, phrase)
        else:
            best = [candidates[n][0] for n in food_names]

        best_score, best_info = self.evaluate(crop, best)
        print(f"[prompt_autotune] seed {best} -> {best_score:.3f} {best_info}")

        # 3. coordinate ascent: vary one food at a time, keep what improves
        evaluated: dict[tuple[str, ...], float] = {tuple(best): best_score}
        for p in range(self.passes):
            improved = False
            for i, name in enumerate(food_names):
                if report:
                    report(f"Finding the best way to spot the {name}")
                for phrase in candidates[name]:
                    trial = list(best)
                    trial[i] = phrase
                    key = tuple(trial)
                    if key in evaluated:
                        continue
                    if phrases_conflict(trial):
                        # Would break the label round-trip in detect_items, so it
                        # is not a usable answer however well it might score.
                        evaluated[key] = -2.0
                        continue
                    score, info = self.evaluate(crop, trial)
                    evaluated[key] = score
                    print(f"[prompt_autotune]   {phrase!r} -> {score:.3f} {info}")
                    if score > best_score + 1e-6:
                        best_score, best_info, best = score, info, trial
                        improved = True
            if not improved:
                break  # converged; a second pass would repeat the same calls

        print(f"[prompt_autotune] BEST {dict(zip(food_names, best))} "
              f"score={best_score:.3f} {best_info}")
        return dict(zip(food_names, best))


# --------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------

def load_prompt_cache(path: str | Path) -> dict[str, str]:
    """Previously tuned phrases, {food_label: phrase}. Missing/corrupt -> {}."""
    p = Path(path)
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        print(f"[prompt_autotune] could not read prompt cache {p}: {exc}")
        return {}
    return {str(k): str(v) for k, v in data.items()} if isinstance(data, dict) else {}


def save_prompt_cache(path: str | Path, prompts: dict[str, str]) -> None:
    """Merge `prompts` into the cache at `path`. Never raises."""
    p = Path(path)
    try:
        merged = load_prompt_cache(p)
        merged.update(prompts)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(merged, indent=2, sort_keys=True) + "\n")
        print(f"[prompt_autotune] saved {len(prompts)} prompt(s) to {p}")
    except OSError as exc:
        print(f"[prompt_autotune] could not write prompt cache {p}: {exc}")
