### Categorical (20)

| Dimension | Model predicted | User's actual preference |  | Reason |
| --- | --- | --- | --- | --- |
| robot_speed | medium | **(unresolved)** | ✗ | Medium matches both personal-setting meals (day 1, day 10). |
| microwave_time | 1 min | **(unresolved)** | ✗ | Day 10 personal night used 1 min for a warm savory meal; sandwiches are similarly served warm. |
| skewering_axis | perpendicular to major axis | **(unresolved)** | ✗ | Perpendicular is the user's constant preference across all meals. |
| confirm_feeding_pickup | yes (with auto-continue countdown) | **(unresolved)** | ✗ | Personal relaxed setting (day 10) uses the auto-continue countdown rather than indefinite waits. |
| confirm_navigation_arrival | yes (with auto-continue countdown) | **no** | ✗ | Matches day 10 personal-night auto-continue countdown. |
| confirm_manipulation | yes (with auto-continue countdown) | **(unresolved)** | ✗ | Matches day 10 personal-night auto-continue countdown. |
| transfer_mode | outside mouth transfer | **(unresolved)** | ✗ | Deployment only performs outside-mouth transfer and user never requested inside. |
| outside_mouth_distance | near | **(unresolved)** | ✗ | Near is the user's invariant distance preference across every meal. |
| convey_robot_ready_for_initiating_transfer | speech + LED | **(unresolved)** | ✗ | Personal-setting meals (day 1, day 10) used speech + LED cues. |
| detect_user_ready_for_initiating_transfer_feeding | open mouth | **(unresolved)** | ✗ | Open mouth used in personal/private settings (day 1, day 10); user has good mouth control. |
| detect_user_ready_for_initiating_transfer_drinking | open mouth | **(unresolved)** | ✗ | Open mouth matches personal-setting pattern in memory. |
| detect_user_ready_for_initiating_transfer_wiping | open mouth | **(unresolved)** | ✗ | Open mouth matches personal-setting pattern in memory. |
| convey_robot_ready_for_completing_transfer | speech + LED | **(unresolved)** | ✗ | Speech + LED used in personal-setting meals (day 1, day 10). |
| detect_user_completed_transfer_feeding | button | **(unresolved)** | ✗ | Day 10 personal night used button; user can press buttons reliably. |
| detect_user_completed_transfer_drinking | button | **(unresolved)** | ✗ | Day 10 personal night used button. |
| detect_user_completed_transfer_wiping | button | button | ✓ |  |
| retract_between_bites | no | **(unresolved)** | ✗ | Personal-setting meals (day 1, day 10) skipped retracting to save time. |
| bite_dipping_preference | do not dip | **(unresolved)** | ✗ | Meal has no dips or sauces, so do not dip is required. |
| wait_before_autocontinue_bite_selection | None | **(unresolved)** | ✗ |  |
| wait_before_autocontinue_task_selection | 60 sec | **(unresolved)** | ✗ | Day 10 personal night used 60 sec. |

### Correction walkthrough

| Step (file) | Direct correction (event) | Correlated prediction changes | Acc |
| --- | --- | --- | --- |
| `134148` init | — initial prediction — | — | **1** |
| `134225` | `confirm_navigation_arrival` yes (with auto-continue countdown)→**no** ✓ | **−** `detect_user_completed_transfer_wiping` button→**perception** ✗ | 1→**1** |
| `134301` | `detect_user_completed_transfer_wiping` perception→**button** ✓ *(undoing drift)* | none | 1→**2** |
