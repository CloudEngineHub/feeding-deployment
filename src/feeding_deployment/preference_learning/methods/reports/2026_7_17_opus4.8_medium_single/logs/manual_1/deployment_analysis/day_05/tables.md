### Categorical (20)

| Dimension | Model predicted | User's actual preference |  | Reason |
| --- | --- | --- | --- | --- |
| robot_speed | medium | **fast** | ✗ | Medium is the user's most common speed across history absent a state cue. |
| microwave_time | 2 min | **(unresolved)** | ✗ | Chicken nugget meal on day 1 used 2 min; warm meal expected. |
| skewering_axis | perpendicular to major axis | **(unresolved)** | ✗ | Perpendicular in every prior meal. |
| confirm_feeding_pickup | yes (with auto-continue countdown) | **(unresolved)** | ✗ | Kept on across history and especially useful socially to avoid awkward empty-fork transfers. |
| confirm_navigation_arrival | yes (with auto-continue countdown) | **yes (without any auto-continue)** | ✗ | Consistently on with auto-continue countdown across all prior meals. |
| confirm_manipulation | yes (with auto-continue countdown) | **yes (without any auto-continue)** | ✗ | Consistently on with auto-continue countdown across all prior meals. |
| transfer_mode | outside mouth transfer | **(unresolved)** | ✗ | Deployment only does outside-mouth and user can lean forward; no request for inside. |
| outside_mouth_distance | near | **(unresolved)** | ✗ | Near in every prior meal and user is comfortable close. |
| convey_robot_ready_for_initiating_transfer | no cue | **(unresolved)** | ✗ | Recent meals (days 2-4) used no cue; subtle in a social setting. |
| detect_user_ready_for_initiating_transfer_feeding | button | **(unresolved)** | ✗ | Social setting makes open-mouth detection unreliable while talking; user can press buttons. |
| detect_user_ready_for_initiating_transfer_drinking | button | **(unresolved)** | ✗ | Same social-latent factor favors button over open mouth. |
| detect_user_ready_for_initiating_transfer_wiping | button | **(unresolved)** | ✗ | Same social-latent factor favors button over open mouth. |
| convey_robot_ready_for_completing_transfer | no cue | **LED** | ✗ | Recent meals used no cue; keeps interaction subtle in company. |
| detect_user_completed_transfer_feeding | perception | **(unresolved)** | ✗ | Force-torque perception is reliable and unobtrusive; used on day 1. |
| detect_user_completed_transfer_drinking | button | **(unresolved)** | ✗ | Head-nod perception feels unnatural socially, so button as in days 2-4. |
| detect_user_completed_transfer_wiping | button | **(unresolved)** | ✗ | Button in every prior meal and avoids awkward head-nod socially. |
| retract_between_bites | yes | **(unresolved)** | ✗ | Social setting favors retracting to avoid obstructing the partner's view, as on days 2-4. |
| bite_dipping_preference | less | **more** | ✗ | Ketchup is present and dippable, but social setting favors less to reduce messiness. |
| wait_before_autocontinue_bite_selection | None | **(unresolved)** | ✗ |  |
| wait_before_autocontinue_task_selection | no autocontinue | **(unresolved)** | ✗ | Social setting with chatting favors no autocontinue between tasks, as on day 1. |

### Correction walkthrough

| Step (file) | Direct correction (event) | Correlated prediction changes | Acc |
| --- | --- | --- | --- |
| `111525` init | — initial prediction — | — | **0** |
| `111623` | `bite_ordering` *(TEXT)* | none | 0→**0** |
| `111719` | `bite_dipping_preference` less→**more** ✓ | none | 0→**1** |
| `111807` | `convey_complete` no cue→**LED** ✓ | none | 1→**2** |
| `111859` | `wait_before_autocontinue_feeding_pickup` *(?)* | none | 2→**2** |
| `111948` | `confirm_manipulation` yes (with auto-continue countdown)→**yes (without any auto-continue)** ✓ | **+** `confirm_navigation_arrival` yes (with auto-continue countdown)→yes (without any auto-continue) ✓ | 2→**4** |
| `112042` | `wait_before_autocontinue_mealprep` *(?)* | **−** `confirm_navigation_arrival` yes (without any auto-continue)→**yes (with auto-continue countdown)** ✗ | 4→**3** |
| `112130` | `confirm_navigation_arrival` yes (with auto-continue countdown)→**yes (without any auto-continue)** ✓ | none | 3→**4** |
| `112214` | `robot_speed` medium→**fast** ✓ | none | 4→**5** |
