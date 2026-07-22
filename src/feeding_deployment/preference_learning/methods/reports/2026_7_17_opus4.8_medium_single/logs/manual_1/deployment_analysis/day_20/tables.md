### Categorical (20)

| Dimension | Model predicted | User's actual preference |  | Reason |
| --- | --- | --- | --- | --- |
| robot_speed | medium | **fast** | ✗ | Afternoon social meals (days 9, 19) consistently used medium speed. |
| microwave_time | no microwave | **(unresolved)** | ✗ | Cold dessert; day 8 strawberries with whipped cream used no microwave. |
| skewering_axis | perpendicular to major axis | **(unresolved)** | ✗ | Perpendicular in every prior meal, a stable long-term tendency. |
| confirm_feeding_pickup | yes (with auto-continue countdown) | **(unresolved)** | ✗ | Recent social meals (days 18, 19) used yes with auto-continue countdown. |
| confirm_navigation_arrival | no | **(unresolved)** | ✗ | All recent meals (days 12-19) relaxed this to no. |
| confirm_manipulation | no | **(unresolved)** | ✗ | All recent meals (days 12-19) relaxed this to no. |
| transfer_mode | outside mouth transfer | **(unresolved)** | ✗ | Deployment only does outside mouth transfer and user never requested inside. |
| outside_mouth_distance | near | **(unresolved)** | ✗ | Near in every prior meal. |
| convey_robot_ready_for_initiating_transfer | LED | **(unresolved)** | ✗ | Social settings consistently use LED cues (days 5,9,14,16,18,19). |
| detect_user_ready_for_initiating_transfer_feeding | button | **(unresolved)** | ✗ | Social settings use button to avoid mouth-open ambiguity while talking. |
| detect_user_ready_for_initiating_transfer_drinking | button | **(unresolved)** | ✗ | Social settings use button, matching feeding detection. |
| detect_user_ready_for_initiating_transfer_wiping | button | **(unresolved)** | ✗ | Social settings use button, matching feeding detection. |
| convey_robot_ready_for_completing_transfer | LED | **(unresolved)** | ✗ | Social settings consistently use LED cues. |
| detect_user_completed_transfer_feeding | button | **(unresolved)** | ✗ | Social settings use button for completion (days 9,14,16,18,19). |
| detect_user_completed_transfer_drinking | button | **(unresolved)** | ✗ | Social settings use button, matching feeding. |
| detect_user_completed_transfer_wiping | button | **(unresolved)** | ✗ | Social settings use button, matching feeding. |
| retract_between_bites | yes | **(unresolved)** | ✗ | Social settings favor retracting to avoid obstructing view (yes in all social meals). |
| bite_dipping_preference | less | **more** | ✗ | Day 8 used more dipping for this dessert, but social setting reduces messiness, so less. |
| wait_before_autocontinue_bite_selection | None | **(unresolved)** | ✗ |  |
| wait_before_autocontinue_task_selection | no autocontinue | **(unresolved)** | ✗ | Social settings use no autocontinue for chatting between tasks. |

### Correction walkthrough

| Step (file) | Direct correction (event) | Correlated prediction changes | Acc |
| --- | --- | --- | --- |
| `135644` init | — initial prediction — | — | **0** |
| `135722` | `robot_speed` medium→**fast** ✓ | none | 0→**1** |
| `135805` | `wait_before_autocontinue_mealprep` *(?)* | **+** `bite_dipping_preference` less→more ✓ | 1→**2** |
| `135845` | `bite_ordering` *(TEXT)* | **−** `bite_dipping_preference` more→**less** ✗ | 2→**1** |
| `135916` | `bite_dipping_preference` less→**more** ✓ | none | 1→**2** |
