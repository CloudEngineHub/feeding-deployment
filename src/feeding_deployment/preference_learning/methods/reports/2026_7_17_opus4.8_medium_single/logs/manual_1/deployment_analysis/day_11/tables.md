### Categorical (20)

| Dimension | Model predicted | User's actual preference |  | Reason |
| --- | --- | --- | --- | --- |
| robot_speed | fast | **(unresolved)** | ✗ | Day 3 same-meal TV context used fast; user tolerates faster speed while watching TV. |
| microwave_time | 1 min | **(unresolved)** | ✗ | Day 3 (same meal, watching TV) used 1 min for this exact food. |
| skewering_axis | perpendicular to major axis | **(unresolved)** | ✗ | Perpendicular is the user's consistent choice across all meals. |
| confirm_feeding_pickup | yes (with auto-continue countdown) | **(unresolved)** | ✗ | TV/personal meals consistently use yes with auto-continue countdown. |
| confirm_navigation_arrival | yes (with auto-continue countdown) | **(unresolved)** | ✗ | Matches TV-context meals (yes with countdown); no social pressure to wait indefinitely. |
| confirm_manipulation | yes (with auto-continue countdown) | **(unresolved)** | ✗ | Consistent with TV-context meals using yes with countdown. |
| transfer_mode | outside mouth transfer | **(unresolved)** | ✗ | Deployment only performs outside mouth transfer and user has never requested otherwise. |
| outside_mouth_distance | near | **(unresolved)** | ✗ | Near is the user's invariant preference in every prior meal. |
| convey_robot_ready_for_initiating_transfer | no cue | **(unresolved)** | ✗ | TV-watching meals consistently use no cue. |
| detect_user_ready_for_initiating_transfer_feeding | open mouth | **(unresolved)** | ✗ | TV/personal meals use open mouth; user can open mouth wide. |
| detect_user_ready_for_initiating_transfer_drinking | open mouth | **(unresolved)** | ✗ | Matches open-mouth pattern in non-social TV meals. |
| detect_user_ready_for_initiating_transfer_wiping | open mouth | **(unresolved)** | ✗ | Matches open-mouth pattern in non-social TV meals. |
| convey_robot_ready_for_completing_transfer | no cue | **(unresolved)** | ✗ | No cue matches TV-watching context meals. |
| detect_user_completed_transfer_feeding | button | **(unresolved)** | ✗ | Button used consistently in TV meals including day 3. |
| detect_user_completed_transfer_drinking | button | **(unresolved)** | ✗ | Button consistent across TV-context meals. |
| detect_user_completed_transfer_wiping | button | **(unresolved)** | ✗ | Button used across nearly all meals. |
| retract_between_bites | yes | **(unresolved)** | ✗ | TV-watching meals use retract yes to avoid obstructing view. |
| bite_dipping_preference | do not dip | **(unresolved)** | ✗ | No dips in this meal, so do not dip is required. |
| wait_before_autocontinue_bite_selection | None | **(unresolved)** | ✗ |  |
| wait_before_autocontinue_task_selection | 60 sec | **(unresolved)** | ✗ | Day 3 same-meal TV context used 60 sec. |

### Correction walkthrough

| Step (file) | Direct correction (event) | Correlated prediction changes | Acc |
| --- | --- | --- | --- |
| `133635` init | — initial prediction — | — | **0** |
