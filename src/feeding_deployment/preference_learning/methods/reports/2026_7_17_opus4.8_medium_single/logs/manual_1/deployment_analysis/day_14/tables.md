### Categorical (20)

| Dimension | Model predicted | User's actual preference |  | Reason |
| --- | --- | --- | --- | --- |
| robot_speed | medium | **fast** | ✗ | Social meal day 9 and sandwich meal day 13 both used medium. |
| microwave_time | 1 min | **(unresolved)** | ✗ | Bite-sized sandwiches on day 13 used 1 min, consistent with warm-not-hot sandwich servings. |
| skewering_axis | perpendicular to major axis | **(unresolved)** | ✗ | Perpendicular is the user's invariant choice across all meals. |
| confirm_feeding_pickup | yes (without any auto-continue) | **yes (with auto-continue countdown)** | ✗ | Social settings (days 5, 9) consistently used yes (without any auto-continue) to avoid awkward empty-fork transfers. |
| confirm_navigation_arrival | yes (without any auto-continue) | **(unresolved)** | ✗ | Social meals (days 5, 9) used full no-autocontinue confirmation, driven by the social latent factor. |
| confirm_manipulation | yes (without any auto-continue) | **no** | ✗ | Social meals (days 5, 9) used full no-autocontinue confirmation in front of others. |
| transfer_mode | outside mouth transfer | **(unresolved)** | ✗ | Deployment only performs outside mouth transfer and user never requested inside. |
| outside_mouth_distance | near | **(unresolved)** | ✗ | User consistently prefers near across all meals. |
| convey_robot_ready_for_initiating_transfer | LED | **(unresolved)** | ✗ | Social meals (days 5, 9) used discreet LED cues. |
| detect_user_ready_for_initiating_transfer_feeding | button | **(unresolved)** | ✗ | Social settings use button since open-mouth is unreliable while talking. |
| detect_user_ready_for_initiating_transfer_drinking | button | **(unresolved)** | ✗ | Social settings (days 5, 9) used button. |
| detect_user_ready_for_initiating_transfer_wiping | button | **(unresolved)** | ✗ | Social settings (days 5, 9) used button. |
| convey_robot_ready_for_completing_transfer | LED | **(unresolved)** | ✗ | Social meals (days 5, 9) used LED. |
| detect_user_completed_transfer_feeding | button | **(unresolved)** | ✗ | User predominantly uses button, and social meals used button. |
| detect_user_completed_transfer_drinking | button | **(unresolved)** | ✗ | Social meals (days 5, 9) used button; head-nod feels unnatural socially. |
| detect_user_completed_transfer_wiping | button | **(unresolved)** | ✗ | User consistently uses button for wiping completion. |
| retract_between_bites | yes | **(unresolved)** | ✗ | Social settings (days 5, 9) used retract=yes to avoid obstructing the companion's view. |
| bite_dipping_preference | do not dip | **(unresolved)** | ✗ | Meal has no dips, so do not dip is required. |
| wait_before_autocontinue_bite_selection | None | **(unresolved)** | ✗ |  |
| wait_before_autocontinue_task_selection | no autocontinue | no autocontinue | ✓ |  |

### Correction walkthrough

| Step (file) | Direct correction (event) | Correlated prediction changes | Acc |
| --- | --- | --- | --- |
| `134335` init | — initial prediction — | — | **1** |
| `134428` | `confirm_feeding_pickup` yes (without any auto-continue)→**yes (with auto-continue countdown)** ✓ | ~ `confirm_manipulation` yes (without any auto-continue)→yes (with auto-continue countdown) (✗→✗) · **−** `wait_task` no autocontinue→**60 sec** ✗ | 1→**1** |
| `134520` | `confirm_manipulation` yes (with auto-continue countdown)→**no** ✓ | **+** `wait_task` 60 sec→no autocontinue ✓ | 1→**3** |
| `134607` | `robot_speed` medium→**fast** ✓ | **−** `wait_task` no autocontinue→**15 sec** ✗ | 3→**3** |
| `134653` | `wait_before_autocontinue_mealprep` *(?)* | ~ `wait_task` 15 sec→30 sec (✗→✗) | 3→**3** |
| `134725` | `wait_task` 30 sec→**no autocontinue** ✓ *(undoing drift)* | none | 3→**4** |
