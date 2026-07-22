### Categorical (20)

| Dimension | Model predicted | User's actual preference |  | Reason |
| --- | --- | --- | --- | --- |
| robot_speed | medium | **fast** | ✗ | Medium was stable across both prior meals. |
| microwave_time | 1 min | **(unresolved)** | ✗ | Day 2 (matching TV/afternoon context) used 1 min for a similar chicken meal. |
| skewering_axis | perpendicular to major axis | **(unresolved)** | ✗ | Perpendicular was used in both prior meals for bite success. |
| confirm_feeding_pickup | yes (with auto-continue countdown) | **(unresolved)** | ✗ | Both prior meals used yes with auto-continue countdown. |
| confirm_navigation_arrival | yes (with auto-continue countdown) | **(unresolved)** | ✗ | Both prior meals used yes with auto-continue countdown. |
| confirm_manipulation | yes (with auto-continue countdown) | **(unresolved)** | ✗ | Both prior meals used yes with auto-continue countdown. |
| transfer_mode | outside mouth transfer | **(unresolved)** | ✗ | Deployment only performs outside-mouth and user has not requested inside. |
| outside_mouth_distance | near | **(unresolved)** | ✗ | Near was consistent across both prior meals. |
| convey_robot_ready_for_initiating_transfer | no cue | **(unresolved)** | ✗ | Day 2's matching TV context used no cue. |
| detect_user_ready_for_initiating_transfer_feeding | open mouth | **(unresolved)** | ✗ | Open mouth used in both prior meals and user can open mouth wide. |
| detect_user_ready_for_initiating_transfer_drinking | open mouth | **(unresolved)** | ✗ | Open mouth used in both prior meals. |
| detect_user_ready_for_initiating_transfer_wiping | open mouth | **(unresolved)** | ✗ | Open mouth used in both prior meals. |
| convey_robot_ready_for_completing_transfer | no cue | **(unresolved)** | ✗ | Day 2's matching TV context used no cue. |
| detect_user_completed_transfer_feeding | button | **(unresolved)** | ✗ | Day 2's matching TV context used button; user can press buttons. |
| detect_user_completed_transfer_drinking | button | **(unresolved)** | ✗ | Day 2's matching TV context used button (head-nod may feel unnatural). |
| detect_user_completed_transfer_wiping | button | **(unresolved)** | ✗ | Button used in both prior meals. |
| retract_between_bites | yes | **(unresolved)** | ✗ | Day 2's matching TV context used yes to avoid blocking the TV view. |
| bite_dipping_preference | do not dip | **(unresolved)** | ✗ | No dips in this meal, so do not dip is required. |
| wait_before_autocontinue_bite_selection | None | **(unresolved)** | ✗ |  |
| wait_before_autocontinue_task_selection | 60 sec | **(unresolved)** | ✗ | Day 2's matching TV context used 60 sec (distraction-prone). |

### Correction walkthrough

| Step (file) | Direct correction (event) | Correlated prediction changes | Acc |
| --- | --- | --- | --- |
| `105120` init | — initial prediction — | — | **0** |
| `105203` | `robot_speed` medium→**fast** ✓ | none | 0→**1** |
| `105235` | `wait_before_autocontinue_mealprep` *(?)* | none | 1→**1** |
