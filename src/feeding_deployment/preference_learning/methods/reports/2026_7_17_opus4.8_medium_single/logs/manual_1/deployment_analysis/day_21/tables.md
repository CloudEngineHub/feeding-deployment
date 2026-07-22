### Categorical (20)

| Dimension | Model predicted | User's actual preference |  | Reason |
| --- | --- | --- | --- | --- |
| robot_speed | medium | **(unresolved)** | ✗ | Day=16 social pizza meal used medium; social evenings vary but medium is the mode. |
| microwave_time | 1 min | **(unresolved)** | ✗ | Matches day=16 identical pizza-and-broccoli meal and typical 1 min for warm foods. |
| skewering_axis | perpendicular to major axis | **(unresolved)** | ✗ | Always perpendicular across all meals. |
| confirm_feeding_pickup | yes (with auto-continue countdown) | **(unresolved)** | ✗ | Consistently yes (with countdown) including social meals like day=16. |
| confirm_navigation_arrival | no | **(unresolved)** | ✗ | Recent meals (day 12+) all set to no as parking proved reliable. |
| confirm_manipulation | no | **(unresolved)** | ✗ | Recent meals all set to no as perception proved reliable. |
| transfer_mode | outside mouth transfer | **(unresolved)** | ✗ | Always outside mouth transfer; user can lean forward. |
| outside_mouth_distance | near | **(unresolved)** | ✗ | Always near across all meals. |
| convey_robot_ready_for_initiating_transfer | LED | **(unresolved)** | ✗ | Social settings consistently use LED (days 5,9,14,16,18,19,20). |
| detect_user_ready_for_initiating_transfer_feeding | button | **(unresolved)** | ✗ | Social settings use button to avoid mouth-open confusion while talking. |
| detect_user_ready_for_initiating_transfer_drinking | button | **(unresolved)** | ✗ | Social settings use button consistently. |
| detect_user_ready_for_initiating_transfer_wiping | button | **(unresolved)** | ✗ | Social settings use button consistently. |
| convey_robot_ready_for_completing_transfer | LED | **(unresolved)** | ✗ | Social settings consistently use LED. |
| detect_user_completed_transfer_feeding | button | **(unresolved)** | ✗ | Button used across nearly all meals and all social ones. |
| detect_user_completed_transfer_drinking | button | **(unresolved)** | ✗ | Button used in social settings to avoid head-nod awkwardness. |
| detect_user_completed_transfer_wiping | button | **(unresolved)** | ✗ | Button consistently used. |
| retract_between_bites | yes | **(unresolved)** | ✗ | Social/watching settings prefer retract yes as in day=16. |
| bite_dipping_preference | do not dip | **(unresolved)** | ✗ | No dips in this meal, so do not dip (hard rule). |
| wait_before_autocontinue_bite_selection | None | **(unresolved)** | ✗ |  |
| wait_before_autocontinue_task_selection | no autocontinue | **(unresolved)** | ✗ | Social settings use no autocontinue (days 5,9,14,16,18,19,20). |

### Correction walkthrough

| Step (file) | Direct correction (event) | Correlated prediction changes | Acc |
| --- | --- | --- | --- |
| `135939` init | — initial prediction — | — | **0** |
