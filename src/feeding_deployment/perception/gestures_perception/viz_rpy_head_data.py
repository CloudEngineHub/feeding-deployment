
from pathlib import Path
import matplotlib.pyplot as plt
import pickle
import numpy as np

gesture_data_path = Path(__file__).parent.parent.parent / "integration" / "log" / "rajat" / "default" / "gesture_detectors" / "gesture_examples" / "head_still.pkl"

with open(gesture_data_path, 'rb') as f:
    gesture_data = pickle.load(f)
language_description = gesture_data['gesture_description']
function_label = gesture_data['gesture_label']
positive_examples = gesture_data['positive_examples']
negative_examples = gesture_data['negative_examples']

fig, axes = plt.subplots(3, 1, figsize=(12, 8))
plt.suptitle(language_description)

for i, label in enumerate(["roll", "pitch", "yaw"]):
    ax = axes[i]
    for pos_idx in range(len(positive_examples)):
        pos_ex = np.array(positive_examples[pos_idx]['head_pose'])
        ax.plot(pos_ex[..., 3 + i], label="Positive", color="blue")
    for neg_idx in range(len(negative_examples)):
        neg_ex = np.array(negative_examples[neg_idx]['head_pose'])
        ax.plot(neg_ex[..., 3 + i], label="Negative", color="orange")
    ax.set_ylabel(f"{label} (deg)")

plt.legend()

plt.tight_layout()
plt.show()
