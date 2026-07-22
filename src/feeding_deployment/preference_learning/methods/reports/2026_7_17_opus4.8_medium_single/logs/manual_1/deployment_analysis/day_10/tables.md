### Categorical (20)

| Dimension | Model predicted | User's actual preference |  | Reason |
| --- | --- | --- | --- | --- |
| robot_speed | medium | **(unresolved)** | ✗ | Personal/night meals (days 1,8) used medium; long-term default is medium. |
| microwave_time | 2 min | **1 min** | ✗ | Hot fried chicken/wedges match prior chicken-nugget meals reheated 2 min (days 1,6). |
| skewering_axis | perpendicular to major axis | **(unresolved)** | ✗ | Perpendicular in every prior meal. |
| confirm_feeding_pickup | yes (with auto-continue countdown) | **(unresolved)** | ✗ | Non-social settings consistently used countdown confirmation (days 1-4,6-8). |
| confirm_navigation_arrival | yes (with auto-continue countdown) | **(unresolved)** | ✗ | Non-social settings kept countdown confirmation on. |
| confirm_manipulation | yes (with auto-continue countdown) | **(unresolved)** | ✗ | Non-social settings kept countdown confirmation on. |
| transfer_mode | outside mouth transfer | **(unresolved)** | ✗ | Deployment only does outside-mouth and user can lean forward. |
| outside_mouth_distance | near | **(unresolved)** | ✗ | Near in every prior meal. |
| convey_robot_ready_for_initiating_transfer | no cue | **(unresolved)** | ✗ | Non-social recent meals used no cue (days 2-8). |
| detect_user_ready_for_initiating_transfer_feeding | open mouth | **(unresolved)** | ✗ | Non-social meals used open mouth; user can open mouth wide. |
| detect_user_ready_for_initiating_transfer_drinking | open mouth | **(unresolved)** | ✗ | Non-social meals used open mouth readiness. |
| detect_user_ready_for_initiating_transfer_wiping | open mouth | **(unresolved)** | ✗ | Non-social meals used open mouth readiness. |
| convey_robot_ready_for_completing_transfer | no cue | **speech + LED** | ✗ | Non-social recent meals used no cue. |
| detect_user_completed_transfer_feeding | button | **(unresolved)** | ✗ | Recent meals (days 2-9) used button completion. |
| detect_user_completed_transfer_drinking | button | button | ✓ |  |
| detect_user_completed_transfer_wiping | button | **(unresolved)** | ✗ | Button used for wiping completion throughout. |
| retract_between_bites | no | no | ✓ |  |
| bite_dipping_preference | more | **(unresolved)** | ✗ | Ranch is dippable and personal/non-social meals with sauce preferred more dipping (days 6-8). |
| wait_before_autocontinue_bite_selection | None | **(unresolved)** | ✗ |  |
| wait_before_autocontinue_task_selection | 60 sec | 60 sec | ✓ |  |

### Correction walkthrough

| Step (file) | Direct correction (event) | Correlated prediction changes | Acc |
| --- | --- | --- | --- |
| `133138` init | — initial prediction — | — | **3** |
| `133222` | `bite_ordering` *(TEXT)* | **−** `retract_between_bites` no→**yes** ✗ | 3→**2** |
| `133300` | `retract_between_bites` yes→**no** ✓ *(undoing drift)* | **+** `convey_complete` no cue→speech+LED ✓ · **−** `detect_user_completed_transfer_drinking` button→**perception** ✗ · **−** `wait_task` 60 sec→**no autocontinue** ✗ | 2→**2** |
| `133356` | `microwave_time` 2 min→**1 min** ✓ | none | 2→**3** |
| `133427` | `wait_task` no autocontinue→**60 sec** ✓ *(undoing drift)* | **−** `convey_complete` speech+LED→**no cue** ✗ · **+** `detect_user_completed_transfer_drinking` perception→button ✓ | 3→**4** |
| `133459` | `wait_before_autocontinue_mealprep` *(?)* | none | 4→**4** |
| `133533` | `convey_complete` no cue→**speech+LED** ✓ | **−** `detect_user_completed_transfer_drinking` button→**perception** ✗ | 4→**4** |
| `133608` | `detect_user_completed_transfer_drinking` perception→**button** ✓ *(undoing drift)* | none | 4→**5** |
