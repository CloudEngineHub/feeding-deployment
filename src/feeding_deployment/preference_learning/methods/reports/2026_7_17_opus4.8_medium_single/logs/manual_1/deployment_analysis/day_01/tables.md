### Categorical (20)

| Dimension | Model predicted | User's actual preference |  | Reason |
| --- | --- | --- | --- | --- |
| robot_speed | medium | medium | ✓ |  |
| microwave_time | 1 min | **(unresolved)** | ✗ | Chicken nuggets are served warm, so a short reheat is the default. |
| skewering_axis | perpendicular to major axis | **(unresolved)** | ✗ | User can open mouth wide, so perpendicular skewering for higher acquisition success is the default. |
| confirm_feeding_pickup | yes (with auto-continue countdown) | yes (with auto-continue countdown) | ✓ |  |
| confirm_navigation_arrival | yes (with auto-continue countdown) | **(unresolved)** | ✗ | Users start with navigation confirmation on to refine parking; countdown default. |
| confirm_manipulation | yes (with auto-continue countdown) | **(unresolved)** | ✗ | Users start with manipulation confirmation on to refine perception; countdown default. |
| transfer_mode | outside mouth transfer | **(unresolved)** | ✗ | Deployment only performs outside-mouth transfer and no explicit request otherwise. |
| outside_mouth_distance | medium | **near** | ✗ | User can lean forward comfortably; medium is a neutral default. |
| convey_robot_ready_for_initiating_transfer | speech + LED | **(unresolved)** | ✗ | Redundant speech + LED cue is a safe default with no contrary evidence. |
| detect_user_ready_for_initiating_transfer_feeding | open mouth | **(unresolved)** | ✗ | User can open mouth wide; in personal (non-social) setting open mouth is reliable. |
| detect_user_ready_for_initiating_transfer_drinking | open mouth | **(unresolved)** | ✗ | Open mouth is reliable in a personal setting given good head/mouth control. |
| detect_user_ready_for_initiating_transfer_wiping | button | **open mouth** | ✗ | Mouth-opening is ambiguous for wiping, so button is the safer default given user can press buttons. |
| convey_robot_ready_for_completing_transfer | speech + LED | **(unresolved)** | ✗ | Redundant speech + LED cue default with no contrary evidence. |
| detect_user_completed_transfer_feeding | perception | **(unresolved)** | ✗ | Force-torque perception is very reliable and the default for bites. |
| detect_user_completed_transfer_drinking | perception | perception | ✓ |  |
| detect_user_completed_transfer_wiping | perception | **button** | ✗ | User has good head control for head-nod perception in a personal setting. |
| retract_between_bites | no | **(unresolved)** | ✗ | Eating alone in a personal setting, skipping retraction saves time. |
| bite_dipping_preference | do not dip | **(unresolved)** | ✗ | Meal has no dips/sauces, so hard rule forces do not dip. |
| wait_before_autocontinue_bite_selection | None | **(unresolved)** | ✗ |  |
| wait_before_autocontinue_task_selection | 30 sec | **no autocontinue** | ✗ | Neutral default with no evidence of chatting or rushing. |

### Correction walkthrough

| Step (file) | Direct correction (event) | Correlated prediction changes | Acc |
| --- | --- | --- | --- |
| `104018` init | — initial prediction — | — | **3** |
| `104052` | `wait_before_autocontinue_mealprep` *(?)* | ~ `wait_task` 30 sec→60 sec (✗→✗) | 3→**3** |
| `104130` | `wait_task` 60 sec→**no autocontinue** ✓ | **−** `confirm_feeding_pickup` yes (with auto-continue countdown)→**no** ✗ · **+** `detect_user_ready_for_initiating_transfer_wiping` button→open mouth ✓ | 3→**4** |
| `104204` | `outside_mouth_distance` medium→**near** ✓ | **−** `robot_speed` medium→**slow** ✗ | 4→**4** |
| `104246` | `robot_speed` slow→**medium** ✓ *(undoing drift)* | **+** `confirm_feeding_pickup` no→yes (with auto-continue countdown) ✓ | 4→**6** |
| `104324` | `wait_before_autocontinue_feeding_pickup` *(?)* | **−** `confirm_feeding_pickup` yes (with auto-continue countdown)→**yes (without any auto-continue)** ✗ | 6→**5** |
| `104403` | `detect_user_completed_transfer_wiping` perception→**button** ✓ | **−** `detect_user_ready_for_initiating_transfer_wiping` open mouth→**button** ✗ · **−** `detect_user_completed_transfer_drinking` perception→**button** ✗ | 5→**4** |
| `104441` | `detect_user_ready_for_initiating_transfer_wiping` button→**open mouth** ✓ | none | 4→**5** |
| `104514` | `detect_user_completed_transfer_drinking` button→**perception** ✓ *(undoing drift)* | none | 5→**6** |
| `104544` | `confirm_feeding_pickup` yes (without any auto-continue)→**yes (with auto-continue countdown)** ✓ *(undoing drift)* | none | 6→**7** |
