"""Tests for the prompt auto-tuner. CPU only -- no GPU, no Grounding DINO.

Run:  python -m feeding_deployment.perception.test_prompt_autotune
      (or: pytest src/feeding_deployment/perception/test_prompt_autotune.py)

Covers the two things that can silently go wrong:
  1. the detect_items refactor changed a prompt for some food, and
  2. the unsupervised score prefers a wording we know to be worse.
"""

import re
import sys
from pathlib import Path

import cv2
import numpy as np

from feeding_deployment.perception.prompt_autotune import (
    UNION_AREA_DIVISOR,
    _template_candidates,
    load_prompt_cache,
    phrases_conflict,
    propose_with_llm,
    save_prompt_cache,
    score_detection,
)

REPO = Path(__file__).resolve().parents[2]
INFERENCE_PY = REPO / "feeding_deployment" / "actions" / "flair" / "inference_class.py"

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {name}{(' -- ' + detail) if detail and not ok else ''}")
    if not ok:
        FAILURES.append(name)


# ---------------------------------------------------------------------------
# 1. build_detection_phrases produces the prompts the hardcoded rules always did
# ---------------------------------------------------------------------------

def _load_build_detection_phrases():
    """Exec just that method out of inference_class.py, avoiding its heavy imports."""
    src = INFERENCE_PY.read_text()
    start = src.index("    def build_detection_phrases(self):")
    end = src.index("    def detect_items(self,")
    body = "\n".join(l[4:] if l.startswith("    ") else l for l in src[start:end].split("\n"))
    ns: dict = {}
    exec(body, ns)
    return ns["build_detection_phrases"]


class _Stub:
    def __init__(self, classes, categories, overrides=None):
        self.FOOD_CLASSES = classes
        self.FOOD_CATEGORIES = categories
        self.PROMPT_OVERRIDES = overrides or {}


def test_phrases():
    build = _load_build_detection_phrases()

    # Every hardcoded mapping we rely on, including the ones tuned by hand.
    cases = {
        "orange chicken": "glazed fried chicken piece",
        "chicken kebab": "cubed chicken piece",
        "strawberry": "fresh strawberry",
        "pancake": "golden brown pancake piece",
        "sausage": "dark brown cylindrical sausage link",
        "chicken nugget": "chicken nugget piece",
        "broccoli": "green broccoli floret piece",
        "potato wedge": "fried potato wedge",
        "chicken popcorn": "round popcorn chicken nugget",
        "hash brown": "round hash brown piece",
        "apple": "cut white apple slice piece",
        "cantaloupe": "small cut up cantaloupe piece",   # generic fallback
        # Added on main in "Day 16" and merged through the refactor -- these
        # assertions are what proves the merge put them back in the right place.
        "steak": "red cooked steak piece",
        "mozzarella stick": "golden breaded fried cheese stick piece",
    }
    for food, expected in cases.items():
        phrases, repl = build(_Stub([food], ["solid"]))
        check(f"phrase[{food}]", phrases == [expected] and repl[food] == expected,
              f"got {phrases}")

    # Dips keep the "<name> dip" form.
    phrases, _ = build(_Stub(["whipped cream"], ["dip"]))
    check("phrase[whipped cream/dip]", phrases == ["whipped cream dip"], f"got {phrases}")

    # Mixed meal keeps per-index correspondence between classes and categories.
    phrases, repl = build(_Stub(["pancake", "sausage", "maple syrup"],
                                ["solid", "solid", "dip"]))
    check("phrase[mixed meal]",
          phrases == ["golden brown pancake piece",
                      "dark brown cylindrical sausage link",
                      "maple syrup dip"], f"got {phrases}")

    # An override wins, and only for the food it names.
    phrases, repl = build(_Stub(["pancake", "sausage"], ["solid", "solid"],
                                overrides={"sausage": "browned breakfast link"}))
    check("override applies to named food only",
          phrases == ["golden brown pancake piece", "browned breakfast link"],
          f"got {phrases}")
    check("override round-trips via replacement_dict",
          repl["sausage"] == "browned breakfast link")

    # An empty override dict must leave everything untouched (the safety property).
    a, _ = build(_Stub(["orange chicken"], ["solid"], overrides={}))
    b, _ = build(_Stub(["orange chicken"], ["solid"]))
    check("empty overrides == default behaviour", a == b == ["glazed fried chicken piece"])


# ---------------------------------------------------------------------------
# 2. the score prefers the wordings we established by hand
# ---------------------------------------------------------------------------

def _synthetic_plate():
    """A plate crop with two visually distinct foods: pale-yellow and dark-red."""
    crop = np.full((600, 600, 3), 40, np.uint8)          # dark plate
    pale, dark = (150, 220, 240), (60, 60, 150)          # BGR: pale golden / dark red
    boxes_pale, boxes_dark = [], []
    for i in range(16):                                   # 16 pale pieces
        x, y = 30 + (i % 4) * 90, 30 + (i // 4) * 60
        cv2.rectangle(crop, (x, y), (x + 55, y + 45), pale, -1)
        boxes_pale.append([x, y, x + 55, y + 45])
    for i in range(8):                                    # 8 dark pieces
        x, y = 30 + (i % 4) * 90, 330 + (i // 4) * 70
        cv2.rectangle(crop, (x, y), (x + 55, y + 45), dark, -1)
        boxes_dark.append([x, y, x + 55, y + 45])
    return crop, np.array(boxes_pale, float), np.array(boxes_dark, float)


def test_scoring():
    crop, pale, dark = _synthetic_plate()
    plate_area = float(crop.shape[0] * crop.shape[1])

    def conf(n, v):
        return np.full(n, v, dtype=float)

    # (a) Multi-class collapse -- all boxes on one class -- must be disqualified.
    #     This is the pancake/sausage and steak/mozzarella failure.
    collapsed, _ = score_detection(
        [np.vstack([pale, dark]), np.empty((0, 4))],
        [conf(24, 0.36), conf(0, 0)], crop, plate_area, 0, 24)
    split, split_info = score_detection(
        [pale, dark], [conf(16, 0.47), conf(8, 0.47)], crop, plate_area, 0, 24)
    check("collapse scores below a correct split", collapsed < split,
          f"collapse={collapsed:.3f} split={split:.3f}")
    check("collapse is flagged as unusable", collapsed < 0, f"{collapsed:.3f}")

    # (b) Same split, higher confidence wins -- the strawberry case
    #     ("small cut up strawberry piece" 0.31 vs "fresh strawberry" 0.72).
    lo, _ = score_detection([pale], [conf(16, 0.31)], crop, plate_area, 0, 16)
    hi, _ = score_detection([pale], [conf(16, 0.72)], crop, plate_area, 0, 16)
    check("higher confidence wins", hi > lo, f"lo={lo:.3f} hi={hi:.3f}")

    # (c) A wording that needs whole-cluster boxes is penalised -- the orange
    #     chicken case, where the only surviving box spanned the entire plate.
    clean, _ = score_detection([pale], [conf(16, 0.45)], crop, plate_area, 0, 16)
    unions, _ = score_detection([pale], [conf(16, 0.45)], crop, plate_area, 8, 24)
    check("union boxes are penalised", unions < clean, f"clean={clean:.3f} union={unions:.3f}")

    # (d) A *mislabelled* split (each class given a colour-mixed set of boxes)
    #     must score below the true split, even at identical confidence and count.
    mixed_a = np.vstack([pale[:8], dark[:4]])
    mixed_b = np.vstack([pale[8:], dark[4:]])
    mixed, _ = score_detection([mixed_a, mixed_b], [conf(12, 0.47), conf(12, 0.47)],
                               crop, plate_area, 0, 24)
    check("colour-incoherent split scores below the true split", mixed < split,
          f"mixed={mixed:.3f} split={split:.3f}")

    # (e) Sanity: the reported diagnostics match what was passed in.
    check("score reports per-class counts", split_info["counts"] == [16, 8], str(split_info))

    # (f) REGRESSION: finding one tidy piece must not beat finding the plateful.
    #     Observed for real on the steak frame -- an earlier version of this
    #     scorer preferred a wording that found 1 steak (clean, uniform, well
    #     separated) over one that found 20, because nothing rewarded recall.
    #     The sparse case is given *better* conf/size to make the test strict.
    sparse, sparse_info = score_detection(
        [pale[:1], dark[:7]], [conf(1, 0.55), conf(7, 0.55)], crop, plate_area, 0, 8)
    full, full_info = score_detection(
        [pale, dark], [conf(16, 0.42), conf(8, 0.42)], crop, plate_area, 0, 24)
    check("a full plate beats one tidy piece", full > sparse,
          f"full={full:.3f} {full_info} sparse={sparse:.3f} {sparse_info}")

    # (f2) REGRESSION: one abundant food must not mask another's collapse.
    #      Observed for real on the pancake+sausage frame -- 16 pancakes alone
    #      saturated a total-based recall term, so the search happily traded
    #      sausage down from 8 boxes to 3 for a separability gain. The collapsed
    #      side is given *better* conf here to make the test strict.
    balanced, bal_info = score_detection(
        [pale, dark], [conf(16, 0.42), conf(8, 0.42)], crop, plate_area, 0, 24)
    lopsided, lop_info = score_detection(
        [pale, dark[:3]], [conf(16, 0.55), conf(3, 0.55)], crop, plate_area, 0, 19)
    check("an abundant food cannot mask another's collapse", balanced > lopsided,
          f"balanced={balanced:.3f} {bal_info} lopsided={lopsided:.3f} {lop_info}")

    # (g) ...but recall saturates, so shattering food into boxes is not rewarded.
    many = np.repeat(pale, 4, axis=0)   # 64 boxes -- over-segmentation
    over, _ = score_detection([many], [conf(len(many), 0.42)], crop, plate_area, 0, len(many))
    normal, _ = score_detection([pale], [conf(16, 0.42)], crop, plate_area, 0, 16)
    check("recall saturates (over-segmentation gains nothing)", over <= normal + 1e-9,
          f"over={over:.3f} normal={normal:.3f}")


# ---------------------------------------------------------------------------
# 3. candidate generation, conflict guard, cache
# ---------------------------------------------------------------------------

def test_candidates_and_cache(tmp: Path):
    cand = _template_candidates("strawberry", "solid")
    check("candidates include the bare name", "strawberry" in cand, str(cand))
    check("candidates include the current production form",
          "small cut up strawberry piece" in cand, str(cand))
    check("candidates are unique", len(cand) == len(set(c.lower() for c in cand)))

    dip = _template_candidates("whipped cream", "dip")
    check("dip candidates use dip wording", "whipped cream dip" in dip, str(dip))

    extra = _template_candidates("steak", "solid", extra=["red cooked steak piece"])
    check("LLM proposals are ranked first", extra[0] == "red cooked steak piece", str(extra[:2]))

    # The conflict guard protects detect_items' str.replace label round-trip.
    check("duplicate phrases conflict", phrases_conflict(["a piece", "a piece"]))
    check("substring phrases conflict", phrases_conflict(["strawberry", "fresh strawberry"]))
    check("distinct phrases do not conflict",
          not phrases_conflict(["golden brown pancake piece",
                                "dark brown cylindrical sausage link"]))

    # LLM parsing tolerates prose/fences around the JSON, and fails soft.
    good = propose_with_llm(
        lambda p, i: 'sure!\n```json\n{"steak": ["red steak cube", "seared beef piece"]}\n```',
        None, ["steak"])
    check("LLM JSON is extracted from prose", good.get("steak") == ["red steak cube", "seared beef piece"],
          str(good))
    check("LLM garbage -> {}", propose_with_llm(lambda p, i: "no json here", None, ["steak"]) == {})
    def _boom(p, i):
        raise RuntimeError("no api key")
    check("LLM exception -> {} (never breaks a meal)",
          propose_with_llm(_boom, None, ["steak"]) == {})

    # Cache round-trip and merge.
    path = tmp / "prompts.json"
    check("missing cache -> {}", load_prompt_cache(path) == {})
    save_prompt_cache(path, {"strawberry": "fresh strawberry"})
    save_prompt_cache(path, {"pancake": "golden brown pancake piece"})
    loaded = load_prompt_cache(path)
    check("cache merges across calls",
          loaded == {"strawberry": "fresh strawberry",
                     "pancake": "golden brown pancake piece"}, str(loaded))
    (tmp / "bad.json").write_text("{not json")
    check("corrupt cache -> {}", load_prompt_cache(tmp / "bad.json") == {})


# ---------------------------------------------------------------------------
# 4. the tuner must query in the SAME class order production uses
# ---------------------------------------------------------------------------

class _RecordingModel:
    """Stands in for Grounding DINO; records the class order it was handed."""

    def __init__(self):
        self.seen: list[list[str]] = []

    def predict_with_classes(self, image, classes, box_threshold, text_threshold):
        self.seen.append(list(classes))
        import types
        # One tiny box per class, so scoring runs without a collapse.
        n = len(classes)
        return types.SimpleNamespace(
            xyxy=np.array([[10 * i, 10 * i, 10 * i + 8, 10 * i + 8] for i in range(n)], dtype=np.float32),
            confidence=np.full(n, 0.4, dtype=np.float32),
            class_id=np.arange(n),
        )


def test_query_order():
    """Regression: Grounding DINO's attribution depends on class ORDER.

    Measured on a potato-wedge + popcorn-chicken plate: the identical phrase
    pair split 3/7 when queried [potato, popcorn] and collapsed to 10/0 when
    queried [popcorn, potato] -- and production queries in FOOD_CLASSES order.
    A tuner that searched in a different order would recommend prompts that
    fail live, which is exactly what happened when tuning this by hand.
    """
    from feeding_deployment.perception.prompt_autotune import PromptAutoTuner

    model = _RecordingModel()
    tuner = PromptAutoTuner(model, passes=1, max_candidates_per_food=2)
    # Deliberately NOT alphabetical, and a dip in the middle to check filtering.
    foods = ["chicken popcorn", "ranch dressing", "potato wedge"]
    cats = ["solid", "dip", "solid"]
    crop = np.full((200, 200, 3), 60, np.uint8)
    tuner.tune(crop, [0, 0, 200, 200], foods, cats,
               current_phrases=["round popcorn chicken nugget", "ranch dressing dip",
                                "fried potato wedge"])

    check("tuner queried at least once", len(model.seen) > 0)
    # Every query must keep solids in their FOOD_CLASSES order: popcorn before
    # potato, with the dip excluded entirely.
    bad = [q for q in model.seen if len(q) != 2]
    check("dips are excluded from the solid query", not bad,
          f"queries with wrong arity: {bad[:2]}")
    seed = model.seen[0]
    check("solids keep production order (popcorn before potato)",
          "popcorn" in seed[0] and "potato" in seed[1], str(seed))
    swapped = [q for q in model.seen if "potato" in q[0]]
    check("no query ever reverses the class order", not swapped,
          f"reversed queries: {swapped[:2]}")


if __name__ == "__main__":
    import tempfile
    print("== build_detection_phrases ==")
    test_phrases()
    print("\n== scoring ==")
    test_scoring()
    print("\n== candidates / cache ==")
    with tempfile.TemporaryDirectory() as d:
        test_candidates_and_cache(Path(d))
    print("\n== query order ==")
    test_query_order()
    print(f"\n{'FAILED: ' + ', '.join(FAILURES) if FAILURES else 'ALL TESTS PASSED'}")
    sys.exit(1 if FAILURES else 0)
