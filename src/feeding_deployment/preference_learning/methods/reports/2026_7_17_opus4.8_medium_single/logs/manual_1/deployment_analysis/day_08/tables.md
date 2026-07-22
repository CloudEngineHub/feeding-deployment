### Categorical (20)

| Dimension | Model predicted | User's actual preference |  | Reason |
| --- | --- | --- | --- | --- |
| robot_speed | medium | **(unresolved)** | ✗ | Day 7 dessert-while-TV used medium; relaxed dessert context. |
| microwave_time | no microwave | **(unresolved)** | ✗ | Cold fruit dessert; day 4 fruit meal used no microwave. |
| skewering_axis | perpendicular to major axis | **(unresolved)** | ✗ | Perpendicular in every prior meal. |
| confirm_feeding_pickup | yes (with auto-continue countdown) | **(unresolved)** | ✗ | TV meals (days 2,3,7) consistently used yes (with auto-continue countdown). |
| confirm_navigation_arrival | yes (with auto-continue countdown) | **(unresolved)** | ✗ | Consistent yes (with auto-continue countdown) in relaxed TV meals. |
| confirm_manipulation | yes (with auto-continue countdown) | **(unresolved)** | ✗ | Consistent yes (with auto-continue countdown) in relaxed TV meals. |
| transfer_mode | outside mouth transfer | **(unresolved)** | ✗ | Deployment default and all prior meals used outside mouth transfer. |
| outside_mouth_distance | near | **(unresolved)** | ✗ | Near used in every prior meal. |
| convey_robot_ready_for_initiating_transfer | no cue | **(unresolved)** | ✗ | TV meals consistently used no cue. |
| detect_user_ready_for_initiating_transfer_feeding | open mouth | **(unresolved)** | ✗ | TV meals used open mouth; user can open mouth wide. |
| detect_user_ready_for_initiating_transfer_drinking | open mouth | **(unresolved)** | ✗ | TV meals used open mouth. |
| detect_user_ready_for_initiating_transfer_wiping | open mouth | **(unresolved)** | ✗ | TV meals used open mouth. |
| convey_robot_ready_for_completing_transfer | no cue | **(unresolved)** | ✗ | TV meals consistently used no cue. |
| detect_user_completed_transfer_feeding | button | **(unresolved)** | ✗ | TV meals consistently used button; user can press buttons. |
| detect_user_completed_transfer_drinking | button | **(unresolved)** | ✗ | TV meals consistently used button. |
| detect_user_completed_transfer_wiping | button | **(unresolved)** | ✗ | Button used in every prior meal. |
| retract_between_bites | yes | **(unresolved)** | ✗ | TV/social meals used retract yes. |
| bite_dipping_preference | more | **(unresolved)** | ✗ | Day 7 dessert with sauce used more; whipped cream is dippable. |
| wait_before_autocontinue_bite_selection | None | **(unresolved)** | ✗ |  |
| wait_before_autocontinue_task_selection | 60 sec | **(unresolved)** | ✗ | Relaxed dessert TV meals (days 2,3,4,7) used 60 sec. |

### Correction walkthrough

| Step (file) | Direct correction (event) | Correlated prediction changes | Acc |
| --- | --- | --- | --- |
| `112716` init | — initial prediction — | — | **0** |
| `112747` | `bite_ordering` *(TEXT)* | none | 0→**0** |
| `112815` | `wait_before_autocontinue_feeding_pickup` *(?)* | none | 0→**0** |
| `112845` | `wait_before_autocontinue_mealprep` *(?)* | none | 0→**0** |
