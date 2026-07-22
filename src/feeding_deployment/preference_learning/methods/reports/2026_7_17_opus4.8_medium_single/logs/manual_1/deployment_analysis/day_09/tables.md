### Categorical (20)

| Dimension | Model predicted | User's actual preference |  | Reason |
| --- | --- | --- | --- | --- |
| robot_speed | fast | **medium** | ✗ | Day=5 social setting used fast; afternoon meals also trended faster. |
| microwave_time | 1 min | **(unresolved)** | ✗ | Day=2 (same general tso's chicken meal) used 1 min. |
| skewering_axis | perpendicular to major axis | **(unresolved)** | ✗ | Perpendicular in every prior meal. |
| confirm_feeding_pickup | yes (without any auto-continue) | **(unresolved)** | ✗ | Day=5 social used yes (without any auto-continue) for careful pickup checking with a partner present. |
| confirm_navigation_arrival | yes (without any auto-continue) | yes (without any auto-continue) | ✓ |  |
| confirm_manipulation | yes (without any auto-continue) | **(unresolved)** | ✗ | Day=5 social used no-auto-continue confirmation. |
| transfer_mode | outside mouth transfer | **(unresolved)** | ✗ | Deployment does outside-mouth and user never requested inside. |
| outside_mouth_distance | near | **(unresolved)** | ✗ | Near in all prior meals. |
| convey_robot_ready_for_initiating_transfer | LED | **(unresolved)** | ✗ | Day=5 social used discreet LED cue. |
| detect_user_ready_for_initiating_transfer_feeding | button | **(unresolved)** | ✗ | Day=5 social used button since open-mouth is awkward while talking. |
| detect_user_ready_for_initiating_transfer_drinking | button | **(unresolved)** | ✗ | Day=5 social used button for the same social reason. |
| detect_user_ready_for_initiating_transfer_wiping | button | **(unresolved)** | ✗ | Day=5 social used button. |
| convey_robot_ready_for_completing_transfer | LED | **(unresolved)** | ✗ | Day=5 social used LED. |
| detect_user_completed_transfer_feeding | button | **(unresolved)** | ✗ | Day=5 social used button; user can press buttons. |
| detect_user_completed_transfer_drinking | button | **(unresolved)** | ✗ | Day=5 social used button (head-nod feels unnatural socially). |
| detect_user_completed_transfer_wiping | button | **(unresolved)** | ✗ | Button used consistently, including social meals. |
| retract_between_bites | yes | **(unresolved)** | ✗ | Social settings consistently used retract yes to avoid obstructing the partner. |
| bite_dipping_preference | do not dip | **(unresolved)** | ✗ | Meal has no dips, so hard rule forces do not dip. |
| wait_before_autocontinue_bite_selection | None | **(unresolved)** | ✗ |  |
| wait_before_autocontinue_task_selection | no autocontinue | no autocontinue | ✓ |  |

### Correction walkthrough

| Step (file) | Direct correction (event) | Correlated prediction changes | Acc |
| --- | --- | --- | --- |
| `112915` init | — initial prediction — | — | **2** |
| `112951` | `robot_speed` fast→**medium** ✓ | none | 2→**3** |
| `113035` | `wait_before_autocontinue_mealprep` *(?)* | **−** `confirm_navigation_arrival` yes (without any auto-continue)→**yes (with auto-continue countdown)** ✗ · **−** `wait_task` no autocontinue→**60 sec** ✗ | 3→**1** |
| `113137` | `wait_task` 60 sec→**no autocontinue** ✓ *(undoing drift)* | none | 1→**2** |
| `113216` | `confirm_navigation_arrival` yes (with auto-continue countdown)→**yes (without any auto-continue)** ✓ *(undoing drift)* | none | 2→**3** |
