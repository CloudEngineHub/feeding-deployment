import json

available_detectors = {
    "mouth open": "mouth_open_detector",
    "head shake": "head_shake_detector",
    "head still": "head_still_detector"
}

# save in json
with open("available_detectors.json", "w") as f:
    json.dump(available_detectors, f, indent=4)