from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class PhysicalCapability:
    label: str
    description: str


PHYSICAL_CAPABILITY_PROFILES: List[PhysicalCapability] = [
    PhysicalCapability(
        label="severe_paralysis_clear_speech",
        description="Severe upper-limb paralysis with clear speech. This user has very limited voluntary control of their arms and hands and cannot reliably press buttons or perform gestures. However, they have clear and consistent speech, allowing them to communicate intentions verbally. They can open and close their mouth reliably and tolerate moderate feeding speeds, but require the robot to handle nearly all physical aspects of the task. Fatigue is present but not extreme, so they can sustain interaction through an entire meal with consistent pacing.",
    ),
    PhysicalCapability(
        label="moderate_motor_unreliable_speech",
        description="Moderate motor control with unreliable speech. This user retains some arm and hand movement, enough to press a large accessible button and make small adjustments in posture. Their speech is slurred or inconsistent, making voice commands unreliable for precise interaction. They show stable mouth control and can safely receive food, but fatigue appears quickly, requiring slower pacing and occasional pauses. Because speech is unreliable, they depend heavily on simple physical interfaces for control and confirmation.",
    ),
    PhysicalCapability(
        label="high_fatigue_swallowing_risk",
        description="High fatigue with elevated swallowing risk. This user demonstrates limited endurance and becomes fatigued quickly during meals. Their mouth opening is delayed and sometimes inconsistent, requiring careful timing for safe transfer. They are sensitive to fast movements and large bite sizes, and there is heightened concern for choking or aspiration, so feeding must proceed slowly and cautiously. Although they may have some speech or motor ability, safety considerations dominate all interaction design.",
    ),
    PhysicalCapability(
        label="inconsistent_mouth_control",
        description="Inconsistent mouth control with cognitive or timing variability. This user has some physical movement but limited precision and coordination. Speech is present but slow, and gestures are minimal or absent. The most significant challenge is inconsistent mouth timing—they may open too early, too late, or unpredictably—making perception-based bite detection unreliable. They benefit from explicit confirmations and predictable pacing, even when fatigue is not severe.",
    ),
]