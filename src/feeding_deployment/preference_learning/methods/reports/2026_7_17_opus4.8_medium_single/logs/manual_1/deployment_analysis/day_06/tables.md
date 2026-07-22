### Categorical (20)

| Dimension | Model predicted | User's actual preference |  | Reason |
| --- | --- | --- | --- | --- |
| robot_speed | medium | **(unresolved)** | ✗ | TV meals varied medium/fast; medium is the safe central value with no correction yet. |
| microwave_time | 2 min | **(unresolved)** | ✗ | Chicken-nugget meals (days 1, 5) consistently used 2 min. |
| skewering_axis | perpendicular to major axis | **(unresolved)** | ✗ | Perpendicular in every prior meal (long-term tendency). |
| confirm_feeding_pickup | yes (with auto-continue countdown) | **(unresolved)** | ✗ | TV-setting meals (days 2-3) used yes (with auto-continue countdown). |
| confirm_navigation_arrival | yes (with auto-continue countdown) | **(unresolved)** | ✗ | TV-setting meals used yes (with auto-continue countdown). |
| confirm_manipulation | yes (with auto-continue countdown) | **(unresolved)** | ✗ | TV-setting meals used yes (with auto-continue countdown). |
| transfer_mode | outside mouth transfer | **(unresolved)** | ✗ | This deployment only does outside-mouth and user can lean forward. |
| outside_mouth_distance | near | **(unresolved)** | ✗ | Near in every prior meal. |
| convey_robot_ready_for_initiating_transfer | no cue | **(unresolved)** | ✗ | TV meals used no cue. |
| detect_user_ready_for_initiating_transfer_feeding | open mouth | **(unresolved)** | ✗ | TV meals used open mouth and user has good head control. |
| detect_user_ready_for_initiating_transfer_drinking | open mouth | **(unresolved)** | ✗ | TV meals used open mouth. |
| detect_user_ready_for_initiating_transfer_wiping | open mouth | **(unresolved)** | ✗ | TV meals used open mouth. |
| convey_robot_ready_for_completing_transfer | no cue | **(unresolved)** | ✗ | TV meals used no cue. |
| detect_user_completed_transfer_feeding | button | **(unresolved)** | ✗ | TV meals used button; user can press buttons. |
| detect_user_completed_transfer_drinking | button | **(unresolved)** | ✗ | TV meals used button. |
| detect_user_completed_transfer_wiping | button | **(unresolved)** | ✗ | Button used in all prior meals. |
| retract_between_bites | yes | **(unresolved)** | ✗ | TV/social meals used yes (view obstruction matters when watching TV). |
| bite_dipping_preference | more | **(unresolved)** | ✗ | Ketchup present and day 5 with ketchup used more dipping. |
| wait_before_autocontinue_bite_selection | None | **(unresolved)** | ✗ |  |
| wait_before_autocontinue_task_selection | 60 sec | **(unresolved)** | ✗ | TV meals used 60 sec (distracted user). |

### Correction walkthrough

| Step (file) | Direct correction (event) | Correlated prediction changes | Acc |
| --- | --- | --- | --- |
| `112246` init | — initial prediction — | — | **0** |
| `112329` | `wait_before_autocontinue_mealprep` *(?)* | none | 0→**0** |
| `112408` | `bite_ordering` *(TEXT)* | none | 0→**0** |
| `112440` | `wait_before_autocontinue_feeding_pickup` *(?)* | none | 0→**0** |
