### Categorical (20)

| Dimension | Model predicted | User's actual preference |  | Reason |
| --- | --- | --- | --- | --- |
| robot_speed | medium | **(unresolved)** | ✗ | Prior meal set medium and no correction suggests otherwise. |
| microwave_time | 2 min | **1 min** | ✗ | Warm entree; prior meal used 2 min for hot food. |
| skewering_axis | perpendicular to major axis | **(unresolved)** | ✗ | Perpendicular maximizes acquisition success for chicken/broccoli, matching prior meal. |
| confirm_feeding_pickup | yes (with auto-continue countdown) | **(unresolved)** | ✗ | Prior meal kept confirmations on with auto-continue countdown. |
| confirm_navigation_arrival | yes (with auto-continue countdown) | **(unresolved)** | ✗ | Prior meal used yes with auto-continue countdown. |
| confirm_manipulation | yes (with auto-continue countdown) | **(unresolved)** | ✗ | Prior meal used yes with auto-continue countdown. |
| transfer_mode | outside mouth transfer | **(unresolved)** | ✗ | Deployment only performs outside-mouth transfer and user can lean forward. |
| outside_mouth_distance | near | **(unresolved)** | ✗ | Prior meal used near and user is comfortable with the robot. |
| convey_robot_ready_for_initiating_transfer | speech + LED | **no cue** | ✗ | Prior meal used speech + LED. |
| detect_user_ready_for_initiating_transfer_feeding | open mouth | **(unresolved)** | ✗ | User has good mouth control and prior meal used open mouth; eating alone reduces talking-conflict risk. |
| detect_user_ready_for_initiating_transfer_drinking | open mouth | **(unresolved)** | ✗ | Consistent with feeding readiness and prior open-mouth preference. |
| detect_user_ready_for_initiating_transfer_wiping | open mouth | open mouth | ✓ |  |
| convey_robot_ready_for_completing_transfer | speech + LED | **(unresolved)** | ✗ | Prior meal used speech + LED. |
| detect_user_completed_transfer_feeding | perception | **button** | ✗ | Perception is reliable and matches prior meal. |
| detect_user_completed_transfer_drinking | perception | **button** | ✗ | User has good head control for nod gesture; matches prior meal. |
| detect_user_completed_transfer_wiping | button | **(unresolved)** | ✗ | Prior meal used button for wiping completion. |
| retract_between_bites | yes | **(unresolved)** | ✗ | Watching-TV context favors retracting to avoid obstructing the TV view, overriding prior personal-setting 'no'. |
| bite_dipping_preference | do not dip | **(unresolved)** | ✗ | Meal has no dips, so do not dip is required. |
| wait_before_autocontinue_bite_selection | None | **(unresolved)** | ✗ |  |
| wait_before_autocontinue_task_selection | no autocontinue | **60 sec** | ✗ | Prior meal used no autocontinue and TV-watching distraction supports not rushing. |

### Correction walkthrough

| Step (file) | Direct correction (event) | Correlated prediction changes | Acc |
| --- | --- | --- | --- |
| `104613` init | — initial prediction — | — | **1** |
| `104649` | `wait_task` no autocontinue→**60 sec** ✓ | none | 1→**2** |
| `104727` | `detect_user_completed_transfer_drinking` perception→**button** ✓ | none | 2→**3** |
| `104816` | `convey_init` speech+LED→**no cue** ✓ | none | 3→**4** |
| `104856` | `wait_before_autocontinue_mealprep` *(?)* | none | 4→**4** |
| `104938` | `microwave_time` 2 min→**1 min** ✓ | **−** `detect_user_ready_for_initiating_transfer_wiping` open mouth→**button** ✗ | 4→**4** |
| `105022` | `detect_user_ready_for_initiating_transfer_wiping` button→**open mouth** ✓ *(undoing drift)* | none | 4→**5** |
| `105058` | `detect_completed_feeding` perception→**button** ✓ | none | 5→**6** |
