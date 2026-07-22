### Categorical (20)

| Dimension | Model predicted | User's actual preference |  | Reason |
| --- | --- | --- | --- | --- |
| robot_speed | medium | medium | ✓ |  |
| microwave_time | 1 min | **(unresolved)** | ✗ | Warm savory pizza+broccoli; prior social/savory meals ranged 1-2 min, choosing 1 min as a typical reheat. |
| skewering_axis | perpendicular to major axis | **(unresolved)** | ✗ | Perpendicular in every prior meal, a stable long-term tendency. |
| confirm_feeding_pickup | yes (without any auto-continue) | **(unresolved)** | ✗ | Social setting latent factor drives explicit no-auto-continue confirmation as in days 5 and 9. |
| confirm_navigation_arrival | yes (without any auto-continue) | **no** | ✗ | Social setting consistently used no-auto-continue confirmation (days 5, 9). |
| confirm_manipulation | yes (without any auto-continue) | **(unresolved)** | ✗ | Social setting consistently used no-auto-continue confirmation (days 5, 9). |
| transfer_mode | outside mouth transfer | **(unresolved)** | ✗ | Deployment only does outside-mouth transfer and user never requested inside. |
| outside_mouth_distance | near | **(unresolved)** | ✗ | Near in every prior meal, a stable tendency. |
| convey_robot_ready_for_initiating_transfer | LED | **(unresolved)** | ✗ | Social meals used LED cue (days 5, 9). |
| detect_user_ready_for_initiating_transfer_feeding | button | **(unresolved)** | ✗ | Social setting favors button over open-mouth to avoid confusion while talking (days 5, 9). |
| detect_user_ready_for_initiating_transfer_drinking | button | **(unresolved)** | ✗ | Social setting favors button (days 5, 9). |
| detect_user_ready_for_initiating_transfer_wiping | button | **(unresolved)** | ✗ | Social setting favors button (days 5, 9). |
| convey_robot_ready_for_completing_transfer | LED | **(unresolved)** | ✗ | Social meals used LED cue (days 5, 9). |
| detect_user_completed_transfer_feeding | button | **(unresolved)** | ✗ | Button used consistently and especially in social meals to avoid head-nod/force ambiguity. |
| detect_user_completed_transfer_drinking | button | **(unresolved)** | ✗ | Social setting favors button over head-nod perception (days 5, 9). |
| detect_user_completed_transfer_wiping | button | **(unresolved)** | ✗ | Button used across nearly all meals including social ones. |
| retract_between_bites | yes | **(unresolved)** | ✗ | Social settings prefer retracting to avoid obstructing the partner's view (days 5, 9). |
| bite_dipping_preference | do not dip | **(unresolved)** | ✗ | No dips in this meal, so hard rule forces do not dip. |
| wait_before_autocontinue_bite_selection | None | **(unresolved)** | ✗ |  |
| wait_before_autocontinue_task_selection | no autocontinue | **(unresolved)** | ✗ | Social meals used no autocontinue to accommodate conversation (days 5, 9). |

### Correction walkthrough

| Step (file) | Direct correction (event) | Correlated prediction changes | Acc |
| --- | --- | --- | --- |
| `135022` init | — initial prediction — | — | **1** |
| `135105` | `confirm_navigation_arrival` yes (without any auto-continue)→**no** ✓ | **−** `robot_speed` medium→**fast** ✗ | 1→**1** |
| `135143` | `robot_speed` fast→**medium** ✓ *(undoing drift)* | none | 1→**2** |
