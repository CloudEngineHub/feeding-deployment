#!/usr/bin/env python
"""Run the Grounding DINO prompt auto-tuner offline, on logged detection pkls.

This is how you check the tuner without a robot: it replays a real
food_detection_data_*.pkl through the same search that runs on the plate
picture, and prints what production detects today versus what the tuned wording
detects on that same frame.

    # one frame
    python scripts/tune_food_prompts.py <pkl>

    # the frames behind prompts we already tuned by hand -- the regression check
    python scripts/tune_food_prompts.py --suite

    # skip the vision-LLM proposals (templates only; no ANTHROPIC_API_KEY needed)
    python scripts/tune_food_prompts.py <pkl> --no-llm

Needs a free GPU: it loads Grounding DINO. Do not run it while a meal is in
progress -- the running executive already holds the model.
"""

import argparse
import os
import pickle
import subprocess
import sys

import cv2
import numpy as np
import torch
import torchvision

# groundingdino is imported inside load_dino(), not here, so the
# meal-is-running guard below runs before anything touches the GPU.

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from feeding_deployment.perception.prompt_autotune import (  # noqa: E402
    UNION_AREA_DIVISOR, PromptAutoTuner,
)

G = "/home/isacc/Grounded-Segment-Anything"
LOG = ("/home/isacc/deployment_ws/src/feeding-deployment/src/feeding_deployment/"
       "integration/log")

# Frames whose right answer we already established by hand, with the prompt we
# landed on. The tuner should reach these (or beat them) on its own.
SUITE = [
    (f"{LOG}/aimee_test/food_detection_log/food_detection_data_61.pkl",
     {"strawberry": "fresh strawberry"}),
    (f"{LOG}/aimee/food_detection_log/food_detection_data_231.pkl",
     {"pancake": "golden brown pancake piece",
      "sausage": "dark brown cylindrical sausage link"}),
    (f"{LOG}/aimee_test/food_detection_log/food_detection_data_259.pkl",
     {"steak": "red cooked steak piece"}),
]

INFERENCE_PY = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..",
    "src", "feeding_deployment", "actions", "flair", "inference_class.py")


def production_phrases(food_names, categories):
    """The phrases detect_items would use right now, from the live source.

    Read out of inference_class.py rather than reimplemented, so this stays the
    real baseline as prompts are added on main (Day 16 added steak and
    mozzarella; a hardcoded generic baseline here would be a strawman).
    """
    src = open(INFERENCE_PY).read()
    start = src.index("    def build_detection_phrases(self):")
    end = src.index("    def detect_items(self,")
    body = "\n".join(l[4:] if l.startswith("    ") else l
                     for l in src[start:end].split("\n"))
    ns: dict = {}
    exec(body, ns)

    class _Stub:
        FOOD_CLASSES = list(food_names)
        FOOD_CATEGORIES = list(categories)
        PROMPT_OVERRIDES: dict = {}

    phrases, _ = ns["build_detection_phrases"](_Stub())
    return phrases


def load_dino():
    from groundingdino.util.inference import Model  # deferred: see note at top
    print("Loading Grounding DINO (Swin-B) ...")
    return Model(
        model_config_path=G + "/GroundingDINO/groundingdino/config/GroundingDINO_SwinB.py",
        model_checkpoint_path=G + "/groundingdino_swinb_cogcoor.pth",
    )


def count_per_class(dino, crop, phrases, box_thr, text_thr, nms_thr):
    """Boxes per class after NMS and the union-area filter -- what a bite sees."""
    det = dino.predict_with_classes(image=crop, classes=list(phrases),
                                    box_threshold=box_thr, text_threshold=text_thr)
    if len(det.xyxy) == 0:
        return [0] * len(phrases), []
    keep = torchvision.ops.nms(torch.from_numpy(det.xyxy),
                               torch.from_numpy(det.confidence), nms_thr).numpy().tolist()
    xyxy, conf, cid = det.xyxy[keep], det.confidence[keep], det.class_id[keep]
    max_area = (crop.shape[0] * crop.shape[1]) / UNION_AREA_DIVISOR
    areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
    ok = areas <= max_area
    conf, cid = conf[ok], cid[ok]
    counts, confs = [], []
    for k in range(len(phrases)):
        sel = np.array([c is not None and int(c) == k for c in cid], dtype=bool)
        counts.append(int(sel.sum()))
        confs.append(round(float(conf[sel].mean()), 2) if sel.any() else 0.0)
    return counts, confs


def run_one(dino, pkl_path, expected=None, use_llm=True):
    d = pickle.load(open(pkl_path, "rb"))
    frame = d["camera_color_data"]
    it = d["items_detection"]
    bounds = it["plate_bounds"]
    solids = [s for s in (d.get("food_items", {}) or {}).get("solid", []) or []]
    if not solids:
        print(f"  {os.path.basename(pkl_path)}: no solid foods; skipping")
        return None
    categories = ["solid"] * len(solids)

    print(f"\n=== {os.path.basename(pkl_path)}  foods={solids} ===")

    llm_call = None
    if use_llm:
        try:
            from feeding_deployment.actions.flair.inference_class import BiteAcquisitionInference
            server = BiteAcquisitionInference.__new__(BiteAcquisitionInference)
            server.client = None
            llm_call = lambda p, i: BiteAcquisitionInference.ask_claude_about_image(server, p, i)
        except Exception as e:  # noqa: BLE001
            print(f"  (vision-LLM proposals unavailable: {e}; templates only)")

    tuner = PromptAutoTuner(dino, llm_call=llm_call)
    x, y, w, h = bounds
    crop = frame[y:y + h, x:x + w].copy()

    # What production does today for these foods (read from the live source).
    baseline = production_phrases(solids, categories)
    b_counts, b_confs = count_per_class(dino, crop, baseline, 0.30, 0.20, 0.40)
    print(f"  production : {dict(zip(solids, baseline))}")
    print(f"               counts={dict(zip(solids, b_counts))} conf={dict(zip(solids, b_confs))}")

    best = tuner.tune(frame, bounds, solids, categories, current_phrases=baseline)
    tuned = [best[s] for s in solids]
    t_counts, t_confs = count_per_class(dino, crop, tuned, 0.30, 0.20, 0.40)
    print(f"  tuned      : {best}")
    print(f"               counts={dict(zip(solids, t_counts))} conf={dict(zip(solids, t_confs))}")

    if expected:
        for food, want in expected.items():
            got = best.get(food)
            verdict = "MATCHES hand-tuned" if got == want else f"differs (hand-tuned: {want!r})"
            print(f"  [{food}] -> {got!r} : {verdict}")
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pkl", nargs="*", help="food_detection_data_*.pkl to tune against")
    ap.add_argument("--suite", action="store_true",
                    help="run the frames whose prompts we tuned by hand")
    ap.add_argument("--no-llm", action="store_true", help="templates only, no vision LLM")
    ap.add_argument("--force", action="store_true",
                    help="load the model even if a meal appears to be running (don't)")
    args = ap.parse_args()

    targets = [(p, None) for p in args.pkl]
    if args.suite:
        targets += SUITE
    if not targets:
        ap.error("give a pkl path, or --suite")

    # Refuse to load the model while a meal is running. Free VRAM is NOT a safe
    # proxy: a meal holding ~7GB still leaves ~6GB free, enough for this script
    # to load and still starve the meal's next detection. Check for the process.
    if not args.force:
        try:
            # Enumerate with ps and filter in Python. Two earlier versions of
            # this guard silently matched nothing and let the script run during
            # a live meal: once because the executive's cmdline is a bare
            # "python run.py ..." (no path to match on), and once because pgrep
            # parses a leading "--run_on_robot" as a flag rather than a pattern.
            # ps + substring checks has neither failure mode.
            running = subprocess.run(
                ["ps", "-eo", "pid,args"], capture_output=True, text=True,
            ).stdout
            meals = [l.strip() for l in running.splitlines()
                     if "run.py" in l and "--run_on_robot" in l
                     and "ps -eo" not in l and "bash -c" not in l]
            if meals:
                print("A meal is running on the robot:")
                for line in meals:
                    print(f"  {line}")
                print("Refusing to load Grounding DINO -- it would compete for the GPU "
                      "with the meal's own detection. Re-run when the meal is over "
                      "(or pass --force if you are certain).")
                return 1
        except FileNotFoundError:
            print("(could not check for a running meal; pgrep unavailable)")

    if torch.cuda.is_available():
        free_gb = torch.cuda.mem_get_info()[0] / 1e9
        print(f"GPU free: {free_gb:.1f} GB")
        if free_gb < 4.0:
            print("Under 4 GB free -- something else is using the GPU. Refusing to load.")
            return 1

    dino = load_dino()
    for pkl_path, expected in targets:
        if not os.path.exists(pkl_path):
            print(f"\n!! missing: {pkl_path}")
            continue
        run_one(dino, pkl_path, expected=expected, use_llm=not args.no_llm)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
