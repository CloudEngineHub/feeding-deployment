from dataclasses import dataclass
from typing import List, Optional, Tuple, Tuple

@dataclass(frozen=True)
class PreferenceDim:
    field: str
    label: str
    options: List[str]
    description: str

PREFERENCE_BUNDLE: List[PreferenceDim] = [
    PreferenceDim(
        field="microwave_time",
        label="Microwave time",
        options=["no microwave", "1 min", "2 min", "3 min"],
        description="How long food should be reheated before being served. Some users may prefer hotter food while others prefer food closer to room temperature."
    ),
    PreferenceDim(
        field="occlusion_relevance",
        label="Occlusion relevance",
        options=["minimize left occlusion", "minimize front occlusion", "do not consider occlusion"],
        description="How important it is that the robot avoids blocking the user’s view (for example of a TV or a social partner depending on where they are seated relative to the user)."
    ),
    PreferenceDim(
        field="robot_speed",
        label="Robot speed",
        options=["slow", "medium", "fast"],
        description="The speed at which the robot moves."
    ),
    PreferenceDim(
        field="skewering_axis",
        label="Skewering axis selection",
        options=["parallel to major axis", "perpendicular to major axis"],
        description="The direction in which the robot inserts the fork into food when acquiring a bite. Parallel skewering can produce narrower bites that may be easier to eat for users with limited mouth opening. Perpendicular skewering can increase bite acquisition success."
    ),
    PreferenceDim(
        field="web_interface_confirmation",
        label="Web interface confirmation",
        options=["yes", "no"],
        description="Whether the system requires explicit confirmation from the interface that bite acquisition has succeeded."
    ),
    PreferenceDim(
        field="transfer_mode",
        label="Outside mouth vs inside mouth transfer",
        options=["outside mouth transfer", "inside mouth transfer"],
        description="How food is delivered to the mouth: outside-mouth transfer (the robot stops outside the mouth and the user leans forward to take the bite) vs inside-mouth transfer (the robot inserts the food into the mouth)."
    ),
    PreferenceDim(
        field="outside_mouth_distance",
        label="For outside-mouth transfer: distance from the mouth",
        options=["near", "medium", "far"],
        description="If using outside-mouth transfer, how far from the mouth the robot stops before the user takes the bite."
    ),
    PreferenceDim(
        field="robot_ready_cue",
        label="[Robot→Human] Convey ready for initiating transfer",
        options=["speech", "LED", "speech + LED", "no cue"],
        description="How the robot signals that it is ready to initiate transfer."
    ),
    PreferenceDim(
        field="bite_initiation_feeding",
        label="[Human→Robot] Bite initiation for FEEDING",
        options=["open mouth", "button", "autocontinue"],
        description="How the robot determines that the user is ready for bite transfer. open mouth: readiness is detected from the user opening their mouth; button: the user explicitly presses a button or interface control; autocontinue: the robot proceeds automatically after waiting for a timeout."
    ),
    PreferenceDim(
        field="bite_initiation_drinking",
        label="[Human→Robot] Bite initiation for DRINKING",
        options=["open mouth", "button", "autocontinue"],
        description="How the robot determines that the user is ready for drink transfer."
    ),
    PreferenceDim(
        field="bite_initiation_wiping",
        label="[Human→Robot] Bite initiation for MOUTH WIPING",
        options=["open mouth", "button", "autocontinue"],
        description="How the robot determines that the user is ready for wipe transfer."
    ),
    PreferenceDim(
        field="robot_bite_available_cue",
        label="[Robot→Human] Convey user can take a bite",
        options=["speech", "LED", "speech + LED", "no cue"],
        description="How the robot signals that the tool has reached the transfer location and the user can complete the transfer."
    ),
    PreferenceDim(
        field="bite_completion_feeding",
        label="[Human→Robot] Bite completion for FEEDING",
        options=["perception", "button", "autocontinue"],
        description="How the robot determines that the user has finished taking a bite. perception: the robot detects completion using sensors; button: the user explicitly signals completion; autocontinue: the robot proceeds automatically after a timeout."
    ),
    PreferenceDim(
        field="bite_completion_drinking",
        label="[Human→Robot] Bite completion for DRINKING",
        options=["perception", "button", "autocontinue"],
        description="How the robot determines that the user has finished drinking."
    ),
    PreferenceDim(
        field="bite_completion_wiping",
        label="[Human→Robot] Bite completion for MOUTH WIPING",
        options=["perception", "button", "autocontinue"],
        description="How the robot determines that wiping is finished."
    ),
    PreferenceDim(
        field="retract_between_bites",
        label="Retract between bites",
        options=["yes", "no"],
        description="Whether the robot moves to a retract position between tasks to reduce occlusion and avoid obstructing the user."
    ),
    PreferenceDim(
        field="bite_dipping_preference",
        label="Bite dipping preference (X/Y in Z)",
        options=["dip food in sauce", "do not dip"],
        description="Whether the robot dips food into sauces before feeding."
    ),
    PreferenceDim(
        field="amount_to_dip",
        label="Amount to dip",
        options=["not applicable", "less", "more"],
        description="How much sauce should be applied when dipping."
    ),
    PreferenceDim(
        field="wait_before_autocontinue_seconds",
        label="Time to wait before autocontinue",
        options=["10", "100", "1000"],
        description="How long the robot waits before automatically continuing the next task if the user does not intervene."
    ),
]

# Note: bite_ordering_preference is not in the user's list, but keeping it for compatibility
PREFERENCE_BUNDLE_WITH_ORDERING = PREFERENCE_BUNDLE + [
    PreferenceDim(
        field="bite_ordering_preference",
        label="Bite ordering preference (X/Y)",
        options=["alternate X and Y", "start with X then Y", "start with Y then X"],
        description="TODO"
    )
]

