### Categorical (20)

| Dimension | Model predicted | User's actual preference |  | Reason |
| --- | --- | --- | --- | --- |
| robot_speed | medium | **(unresolved)** | ✗ | Medium is the user's baseline in most prior meals including the personal-setting day 1. |
| microwave_time | no microwave | **(unresolved)** | ✗ | Cold fruit meal (melon, banana) is served cold, so no microwaving. |
| skewering_axis | perpendicular to major axis | **(unresolved)** | ✗ | Perpendicular to major axis is consistent across all prior meals. |
| confirm_feeding_pickup | yes (with auto-continue countdown) | **(unresolved)** | ✗ | Consistently 'yes (with auto-continue countdown)' across all prior meals. |
| confirm_navigation_arrival | yes (with auto-continue countdown) | **(unresolved)** | ✗ | Consistently 'yes (with auto-continue countdown)' across all prior meals. |
| confirm_manipulation | yes (with auto-continue countdown) | **(unresolved)** | ✗ | Consistently 'yes (with auto-continue countdown)' across all prior meals. |
| transfer_mode | outside mouth transfer | **(unresolved)** | ✗ | Deployment only performs outside-mouth transfer and user never requested inside. |
| outside_mouth_distance | near | **(unresolved)** | ✗ | Near is used across all prior meals. |
| convey_robot_ready_for_initiating_transfer | speech + LED | **no cue** | ✗ | Personal setting matches day 1 where speech + LED cues were used. |
| detect_user_ready_for_initiating_transfer_feeding | open mouth | **(unresolved)** | ✗ | Open mouth used across all meals and user can open mouth wide. |
| detect_user_ready_for_initiating_transfer_drinking | open mouth | **(unresolved)** | ✗ | Open mouth used across all meals. |
| detect_user_ready_for_initiating_transfer_wiping | open mouth | **(unresolved)** | ✗ | Open mouth used across all meals. |
| convey_robot_ready_for_completing_transfer | speech + LED | **(unresolved)** | ✗ | Matches day 1 personal-setting speech + LED cues. |
| detect_user_completed_transfer_feeding | perception | **(unresolved)** | ✗ | Personal-setting day 1 used perception for feeding completion. |
| detect_user_completed_transfer_drinking | perception | **button** | ✗ | Personal-setting day 1 used perception for drinking; user has good head control. |
| detect_user_completed_transfer_wiping | button | **(unresolved)** | ✗ | Button used for wiping completion in all prior meals. |
| retract_between_bites | no | **yes** | ✗ | Personal-setting day 1 used no retract; alone means view obstruction is fine. |
| bite_dipping_preference | do not dip | **(unresolved)** | ✗ | Meal has no dips, so do not dip (hard rule). |
| wait_before_autocontinue_bite_selection | None | **(unresolved)** | ✗ |  |
| wait_before_autocontinue_task_selection | no autocontinue | **(unresolved)** | ✗ | Personal-setting day 1 used no autocontinue for a relaxed private meal. |

### Correction walkthrough

| Step (file) | Direct correction (event) | Correlated prediction changes | Acc |
| --- | --- | --- | --- |
| `111156` init | — initial prediction — | — | **0** |
| `111234` | `wait_before_autocontinue_feeding_pickup` *(?)* | none | 0→**0** |
| `111323` | `detect_user_completed_transfer_drinking` perception→**button** ✓ | none | 0→**1** |
| `111400` | `convey_init` speech+LED→**no cue** ✓ | none | 1→**2** |
| `111435` | `retract_between_bites` no→**yes** ✓ | none | 2→**3** |
