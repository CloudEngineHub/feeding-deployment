### Categorical (20)

| Dimension | Model predicted | User's actual preference |  | Reason |
| --- | --- | --- | --- | --- |
| robot_speed | medium | medium | ✓ |  |
| microwave_time | 1 min | **(unresolved)** | ✗ | General tso's chicken was reheated 1 min on days 2 and 9, a warm savory dish. |
| skewering_axis | perpendicular to major axis | **(unresolved)** | ✗ | Perpendicular in every prior meal, a stable long-term tendency. |
| confirm_feeding_pickup | yes (with auto-continue countdown) | yes (with auto-continue countdown) | ✓ |  |
| confirm_navigation_arrival | yes (with auto-continue countdown) | **no** | ✗ | TV meals used yes with auto-continue countdown; not a social setting. |
| confirm_manipulation | yes (with auto-continue countdown) | **no** | ✗ | TV meals used yes with auto-continue countdown. |
| transfer_mode | outside mouth transfer | **(unresolved)** | ✗ | Deployment only performs outside-mouth transfer and user never requested inside. |
| outside_mouth_distance | near | **(unresolved)** | ✗ | Near in every prior meal; user can lean forward. |
| convey_robot_ready_for_initiating_transfer | no cue | **(unresolved)** | ✗ | TV-watching meals consistently used no cue. |
| detect_user_ready_for_initiating_transfer_feeding | open mouth | **(unresolved)** | ✗ | TV meals used open mouth; user has good mouth control. |
| detect_user_ready_for_initiating_transfer_drinking | open mouth | **(unresolved)** | ✗ | TV meals used open mouth readiness. |
| detect_user_ready_for_initiating_transfer_wiping | open mouth | **(unresolved)** | ✗ | TV meals used open mouth readiness. |
| convey_robot_ready_for_completing_transfer | no cue | **(unresolved)** | ✗ | TV-watching meals consistently used no cue. |
| detect_user_completed_transfer_feeding | button | **perception** | ✗ | All TV meals used button completion. |
| detect_user_completed_transfer_drinking | button | **(unresolved)** | ✗ | All TV meals used button completion. |
| detect_user_completed_transfer_wiping | button | **perception** | ✗ | Button used consistently across meals for wiping completion. |
| retract_between_bites | yes | **(unresolved)** | ✗ | TV-watching meals preferred retracting to avoid obstructing view. |
| bite_dipping_preference | do not dip | **(unresolved)** | ✗ | Meal has no dips, so must be do not dip. |
| wait_before_autocontinue_bite_selection | None | **(unresolved)** | ✗ |  |
| wait_before_autocontinue_task_selection | 60 sec | **(unresolved)** | ✗ | TV meals with countdown pattern used 60 sec. |

### Correction walkthrough

| Step (file) | Direct correction (event) | Correlated prediction changes | Acc |
| --- | --- | --- | --- |
| `133701` init | — initial prediction — | — | **2** |
| `133740` | `detect_completed_feeding` button→**perception** ✓ | none | 2→**3** |
| `133814` | `wait_before_autocontinue_mealprep` *(?)* | none | 3→**3** |
| `133859` | `confirm_manipulation` yes (with auto-continue countdown)→**no** ✓ | none | 3→**4** |
| `133937` | `confirm_navigation_arrival` yes (with auto-continue countdown)→**no** ✓ | **−** `confirm_feeding_pickup` yes (with auto-continue countdown)→**no** ✗ · **+** `detect_user_completed_transfer_wiping` button→perception ✓ | 4→**5** |
| `134020` | `confirm_feeding_pickup` no→**yes (with auto-continue countdown)** ✓ *(undoing drift)* | **−** `detect_user_completed_transfer_wiping` perception→**button** ✗ | 5→**5** |
| `134050` | `detect_user_completed_transfer_wiping` button→**perception** ✓ | **−** `robot_speed` medium→**fast** ✗ | 5→**5** |
| `134121` | `robot_speed` fast→**medium** ✓ *(undoing drift)* | none | 5→**6** |
