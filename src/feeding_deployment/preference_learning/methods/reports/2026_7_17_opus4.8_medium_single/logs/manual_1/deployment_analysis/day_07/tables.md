### Categorical (20)

| Dimension | Model predicted | User's actual preference |  | Reason |
| --- | --- | --- | --- | --- |
| robot_speed | fast | **medium** | ✗ | Recent TV meals (days 3,6) used fast. |
| microwave_time | no microwave | **1 min** | ✗ | Cold dessert (fruit/brownies with sauce); day 4 fruit meal used no microwave. |
| skewering_axis | perpendicular to major axis | **(unresolved)** | ✗ | Every prior meal used perpendicular to major axis. |
| confirm_feeding_pickup | yes (with auto-continue countdown) | **(unresolved)** | ✗ | TV-setting meals (days 2,3,6) used yes with auto-continue countdown. |
| confirm_navigation_arrival | yes (with auto-continue countdown) | **(unresolved)** | ✗ | TV meals consistently used yes with auto-continue countdown. |
| confirm_manipulation | yes (with auto-continue countdown) | **(unresolved)** | ✗ | TV meals consistently used yes with auto-continue countdown. |
| transfer_mode | outside mouth transfer | **(unresolved)** | ✗ | Deployment only performs outside-mouth and user never requested inside. |
| outside_mouth_distance | near | **(unresolved)** | ✗ | User consistently prefers near across all meals. |
| convey_robot_ready_for_initiating_transfer | no cue | **(unresolved)** | ✗ | TV meals (days 2,3,6) used no cue. |
| detect_user_ready_for_initiating_transfer_feeding | open mouth | **(unresolved)** | ✗ | TV meals used open mouth; user can open mouth wide. |
| detect_user_ready_for_initiating_transfer_drinking | open mouth | **(unresolved)** | ✗ | TV meals used open mouth. |
| detect_user_ready_for_initiating_transfer_wiping | open mouth | **(unresolved)** | ✗ | TV meals used open mouth. |
| convey_robot_ready_for_completing_transfer | no cue | **(unresolved)** | ✗ | TV meals used no cue. |
| detect_user_completed_transfer_feeding | button | **(unresolved)** | ✗ | TV meals used button; user can press buttons. |
| detect_user_completed_transfer_drinking | button | **(unresolved)** | ✗ | TV meals used button. |
| detect_user_completed_transfer_wiping | button | **(unresolved)** | ✗ | Button used across all meals for wiping. |
| retract_between_bites | yes | **(unresolved)** | ✗ | TV/non-personal meals used retract yes to avoid obstructing view. |
| bite_dipping_preference | more | **(unresolved)** | ✗ | Chocolate sauce present; recent meals (days 5,6) preferred more dipping. |
| wait_before_autocontinue_bite_selection | None | **(unresolved)** | ✗ |  |
| wait_before_autocontinue_task_selection | 15 sec | **(unresolved)** | ✗ | Most recent evening/night TV meal (day 6) used 15 sec. |

### Correction walkthrough

| Step (file) | Direct correction (event) | Correlated prediction changes | Acc |
| --- | --- | --- | --- |
| `112508` init | — initial prediction — | — | **0** |
| `112540` | `robot_speed` fast→**medium** ✓ | none | 0→**1** |
| `112615` | `microwave_time` no microwave→**1 min** ✓ | none | 1→**2** |
| `112646` | `bite_ordering` *(TEXT)* | none | 2→**2** |
