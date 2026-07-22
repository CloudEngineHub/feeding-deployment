### Categorical (20)

| Dimension | Model predicted | User's actual preference |  | Reason |
| --- | --- | --- | --- | --- |
| robot_speed | medium | **(unresolved)** | ✗ | Day 15 morning TV meal used medium. |
| microwave_time | 2 min | **(unresolved)** | ✗ | Nugget meals in the morning (day 15) reheated 2 min. |
| skewering_axis | perpendicular to major axis | **(unresolved)** | ✗ | Always perpendicular across all history. |
| confirm_feeding_pickup | yes (with auto-continue countdown) | **(unresolved)** | ✗ | Consistently yes with auto-continue countdown in TV/personal settings. |
| confirm_navigation_arrival | no | **(unresolved)** | ✗ | Recent meals (days 12-16) relaxed navigation checks to no. |
| confirm_manipulation | no | **(unresolved)** | ✗ | Recent meals (days 12-16) relaxed manipulation checks to no. |
| transfer_mode | outside mouth transfer | **(unresolved)** | ✗ | Deployment only does outside mouth transfer and user never requested inside. |
| outside_mouth_distance | near | **(unresolved)** | ✗ | User always prefers near. |
| convey_robot_ready_for_initiating_transfer | no cue | **(unresolved)** | ✗ | TV-watching meals use no cue. |
| detect_user_ready_for_initiating_transfer_feeding | open mouth | **(unresolved)** | ✗ | TV-watching meals detect readiness via open mouth. |
| detect_user_ready_for_initiating_transfer_drinking | open mouth | **(unresolved)** | ✗ | Matches feeding readiness cue in TV settings (open mouth). |
| detect_user_ready_for_initiating_transfer_wiping | open mouth | **(unresolved)** | ✗ | Matches other readiness cues in TV settings (open mouth). |
| convey_robot_ready_for_completing_transfer | no cue | **(unresolved)** | ✗ |  |
| detect_user_completed_transfer_feeding | button | **(unresolved)** | ✗ | Day 15 TV meal used button for completion. |
| detect_user_completed_transfer_drinking | button | **(unresolved)** | ✗ | Consistent with button completion in TV settings. |
| detect_user_completed_transfer_wiping | button | **perception** | ✗ | Consistent with button completion in TV settings. |
| retract_between_bites | yes | **(unresolved)** | ✗ | TV-watching meals prefer retracting to avoid view obstruction. |
| bite_dipping_preference | more | **(unresolved)** | ✗ | Nugget+ketchup meals (days 5,6,15) used more dipping. |
| wait_before_autocontinue_bite_selection | None | **(unresolved)** | ✗ |  |
| wait_before_autocontinue_task_selection | 60 sec | **(unresolved)** | ✗ | Day 15 similar meal used 60 sec. |

### Correction walkthrough

| Step (file) | Direct correction (event) | Correlated prediction changes | Acc |
| --- | --- | --- | --- |
| `135210` init | — initial prediction — | — | **0** |
| `135246` | `detect_user_completed_transfer_wiping` button→**perception** ✓ | none | 0→**1** |
