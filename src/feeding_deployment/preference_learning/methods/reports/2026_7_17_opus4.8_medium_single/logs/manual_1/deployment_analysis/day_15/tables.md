### Categorical (20)

| Dimension | Model predicted | User's actual preference |  | Reason |
| --- | --- | --- | --- | --- |
| robot_speed | fast | **medium** | ✗ | TV-front meals (days 3,6,11) and same meal day 6 used fast. |
| microwave_time | 2 min | **(unresolved)** | ✗ | Same meal (chicken nuggets) on day 6 used 2 min; warm food preferred. |
| skewering_axis | perpendicular to major axis | **(unresolved)** | ✗ | Perpendicular in every prior meal. |
| confirm_feeding_pickup | yes (with auto-continue countdown) | **(unresolved)** | ✗ | Consistently yes (with auto-continue countdown) across nearly all meals. |
| confirm_navigation_arrival | no | no | ✓ |  |
| confirm_manipulation | no | **(unresolved)** | ✗ | Recent trend (days 12-14) relaxed manipulation confirmation to no. |
| transfer_mode | outside mouth transfer | **(unresolved)** | ✗ | Deployment only does outside-mouth and user never requested inside. |
| outside_mouth_distance | near | **(unresolved)** | ✗ | Near in every prior meal and consistent with leaning ability. |
| convey_robot_ready_for_initiating_transfer | no cue | **(unresolved)** | ✗ | TV-front meals consistently use no cue. |
| detect_user_ready_for_initiating_transfer_feeding | open mouth | **(unresolved)** | ✗ | TV-front meals use open mouth; user has good head control. |
| detect_user_ready_for_initiating_transfer_drinking | open mouth | **(unresolved)** | ✗ | TV-front meals use open mouth. |
| detect_user_ready_for_initiating_transfer_wiping | open mouth | **(unresolved)** | ✗ | TV-front meals use open mouth. |
| convey_robot_ready_for_completing_transfer | no cue | **(unresolved)** | ✗ | TV-front meals consistently use no cue. |
| detect_user_completed_transfer_feeding | button | **(unresolved)** | ✗ | TV-front and recent meals use button completion. |
| detect_user_completed_transfer_drinking | button | **(unresolved)** | ✗ | TV-front and recent meals use button completion. |
| detect_user_completed_transfer_wiping | button | **(unresolved)** | ✗ | Button used consistently including recent meals. |
| retract_between_bites | yes | **(unresolved)** | ✗ | TV-front meals (days 2,3,7,8,11,12) use retract yes to avoid blocking view. |
| bite_dipping_preference | more | **(unresolved)** | ✗ | Ketchup present and same meal (days 5,6) used more dipping. |
| wait_before_autocontinue_bite_selection | None | **(unresolved)** | ✗ |  |
| wait_before_autocontinue_task_selection | 60 sec | **(unresolved)** | ✗ | Most TV-front meals used 60 sec. |

### Correction walkthrough

| Step (file) | Direct correction (event) | Correlated prediction changes | Acc |
| --- | --- | --- | --- |
| `134759` init | — initial prediction — | — | **1** |
| `134837` | `bite_ordering` *(TEXT)* | none | 1→**1** |
| `134912` | `robot_speed` fast→**medium** ✓ | **−** `confirm_navigation_arrival` no→**yes (with auto-continue countdown)** ✗ | 1→**1** |
| `134952` | `confirm_navigation_arrival` yes (with auto-continue countdown)→**no** ✓ *(undoing drift)* | none | 1→**2** |
