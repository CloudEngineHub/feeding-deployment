### Categorical (20)

| Dimension | Model predicted | User's actual preference |  | Reason |
| --- | --- | --- | --- | --- |
| robot_speed | medium | **fast** | ✗ | General tso's and social meals cluster at medium (days 9, 12, 16). |
| microwave_time | 1 min | **(unresolved)** | ✗ | General tso's chicken was reheated 1 min on days 2, 9, and 12. |
| skewering_axis | perpendicular to major axis | **(unresolved)** | ✗ | Perpendicular in every prior meal. |
| confirm_feeding_pickup | yes (with auto-continue countdown) | **(unresolved)** | ✗ | Recent and social meals use yes with auto-continue countdown (day 16). |
| confirm_navigation_arrival | no | **(unresolved)** | ✗ | Recent meals (days 12-17) skip this as parking proved reliable. |
| confirm_manipulation | no | **(unresolved)** | ✗ | Recent meals (days 12-17) skip this as perception proved reliable. |
| transfer_mode | outside mouth transfer | **(unresolved)** | ✗ | Deployment only does outside-mouth transfer and user can lean forward. |
| outside_mouth_distance | near | **(unresolved)** | ✗ | Near in every prior meal. |
| convey_robot_ready_for_initiating_transfer | LED | **(unresolved)** | ✗ | Social meals consistently use LED (days 5, 9, 14, 16). |
| detect_user_ready_for_initiating_transfer_feeding | button | **(unresolved)** | ✗ | Social meals use button to avoid mouth-open confusion while talking. |
| detect_user_ready_for_initiating_transfer_drinking | button | **(unresolved)** | ✗ | Social meals use button. |
| detect_user_ready_for_initiating_transfer_wiping | button | **(unresolved)** | ✗ | Social meals use button. |
| convey_robot_ready_for_completing_transfer | LED | **(unresolved)** | ✗ | Social meals consistently use LED. |
| detect_user_completed_transfer_feeding | button | **(unresolved)** | ✗ | Social meals use button; user can press buttons. |
| detect_user_completed_transfer_drinking | button | **(unresolved)** | ✗ | Social meals use button; head-nod feels unnatural socially. |
| detect_user_completed_transfer_wiping | button | **(unresolved)** | ✗ | Social meals use button. |
| retract_between_bites | yes | **(unresolved)** | ✗ | Social meals retract to avoid obstructing the partner's view. |
| bite_dipping_preference | do not dip | **(unresolved)** | ✗ | Meal has no dips, so hard rule forces do not dip. |
| wait_before_autocontinue_bite_selection | None | **(unresolved)** | ✗ |  |
| wait_before_autocontinue_task_selection | no autocontinue | **(unresolved)** | ✗ | Social meals use no autocontinue to allow chatting between bites. |

### Correction walkthrough

| Step (file) | Direct correction (event) | Correlated prediction changes | Acc |
| --- | --- | --- | --- |
| `135314` init | — initial prediction — | — | **0** |
| `135344` | `robot_speed` medium→**fast** ✓ | none | 0→**1** |
| `135417` | `wait_before_autocontinue_mealprep` *(?)* | none | 1→**1** |
