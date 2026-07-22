### Categorical (20)

| Dimension | Model predicted | User's actual preference |  | Reason |
| --- | --- | --- | --- | --- |
| robot_speed | medium | **fast** | ✗ | Morning fruit meal (day 4) and many TV meals used medium. |
| microwave_time | no microwave | **(unresolved)** | ✗ | Cold fruit meal; day 4 identical fruit meal used no microwave. |
| skewering_axis | perpendicular to major axis | **(unresolved)** | ✗ | Perpendicular in every prior meal. |
| confirm_feeding_pickup | yes (with auto-continue countdown) | **(unresolved)** | ✗ | Consistently 'yes (with auto-continue countdown)' across recent meals. |
| confirm_navigation_arrival | no | **(unresolved)** | ✗ | Recent meals (day 12+) consistently set this to 'no'. |
| confirm_manipulation | no | **(unresolved)** | ✗ | Recent meals (day 12+) consistently set this to 'no'. |
| transfer_mode | outside mouth transfer | **(unresolved)** | ✗ | Deployment only does outside-mouth and user can lean forward. |
| outside_mouth_distance | near | **(unresolved)** | ✗ | 'near' in every prior meal. |
| convey_robot_ready_for_initiating_transfer | no cue | **(unresolved)** | ✗ | TV-in-front meals consistently used 'no cue'. |
| detect_user_ready_for_initiating_transfer_feeding | open mouth | **(unresolved)** | ✗ | Non-social TV meals used open mouth; user can open mouth wide. |
| detect_user_ready_for_initiating_transfer_drinking | open mouth | **(unresolved)** | ✗ | Matches feeding readiness cue in non-social TV meals. |
| detect_user_ready_for_initiating_transfer_wiping | open mouth | **(unresolved)** | ✗ | Matches feeding readiness cue in non-social TV meals. |
| convey_robot_ready_for_completing_transfer | no cue | **(unresolved)** | ✗ | TV-in-front meals consistently used 'no cue'. |
| detect_user_completed_transfer_feeding | button | **(unresolved)** | ✗ | Most TV meals used button; user can press buttons reliably. |
| detect_user_completed_transfer_drinking | button | **(unresolved)** | ✗ | Consistent with button-based completion in TV meals. |
| detect_user_completed_transfer_wiping | button | **(unresolved)** | ✗ | Consistent with button-based completion in TV meals. |
| retract_between_bites | yes | **(unresolved)** | ✗ | Most TV/non-personal meals retract to avoid obstructing view. |
| bite_dipping_preference | do not dip | **(unresolved)** | ✗ | No dips in this meal; hard rule forces do not dip. |
| wait_before_autocontinue_bite_selection | None | **(unresolved)** | ✗ |  |
| wait_before_autocontinue_task_selection | 60 sec | **(unresolved)** | ✗ | TV meals like day 15/17 used 60 sec. |

### Correction walkthrough

| Step (file) | Direct correction (event) | Correlated prediction changes | Acc |
| --- | --- | --- | --- |
| `140011` init | — initial prediction — | — | **0** |
| `140039` | `robot_speed` medium→**fast** ✓ | none | 0→**1** |
