### Categorical (20)

| Dimension | Model predicted | User's actual preference |  | Reason |
| --- | --- | --- | --- | --- |
| robot_speed | fast | **medium** | ✗ | This meal (chicken strips/hash brown) and recent social day 18 both used fast. |
| microwave_time | 1 min | **(unresolved)** | ✗ | Days 3 and 11 (same meal) both used 1 min, matching recent social meals. |
| skewering_axis | perpendicular to major axis | **(unresolved)** | ✗ | Perpendicular is used in every prior meal. |
| confirm_feeding_pickup | yes (with auto-continue countdown) | **(unresolved)** | ✗ | Recent social meals (days 14, 16, 18) use auto-continue countdown. |
| confirm_navigation_arrival | no | **(unresolved)** | ✗ | Trust has relaxed to 'no' in all recent meals (days 12-18). |
| confirm_manipulation | no | **(unresolved)** | ✗ | Trust has relaxed to 'no' in all recent meals (days 12-18). |
| transfer_mode | outside mouth transfer | **(unresolved)** | ✗ | Deployment only performs outside-mouth transfer and user can lean forward. |
| outside_mouth_distance | near | **(unresolved)** | ✗ | Consistently 'near' across all meals. |
| convey_robot_ready_for_initiating_transfer | LED | **(unresolved)** | ✗ | Social meals consistently use LED cues (days 5, 9, 14, 16, 18). |
| detect_user_ready_for_initiating_transfer_feeding | button | **(unresolved)** | ✗ | Social settings use button since open-mouth conflicts with talking. |
| detect_user_ready_for_initiating_transfer_drinking | button | **(unresolved)** | ✗ | Social settings use button per days 5, 9, 14, 16, 18. |
| detect_user_ready_for_initiating_transfer_wiping | button | **(unresolved)** | ✗ | Social settings use button per days 5, 9, 14, 16, 18. |
| convey_robot_ready_for_completing_transfer | LED | **(unresolved)** | ✗ | Social meals consistently use LED cues. |
| detect_user_completed_transfer_feeding | button | **(unresolved)** | ✗ | Button used in nearly all social meals; force-torque not preferred socially. |
| detect_user_completed_transfer_drinking | button | **(unresolved)** | ✗ | Head-nod feels unnatural socially, so button used in social meals. |
| detect_user_completed_transfer_wiping | button | **(unresolved)** | ✗ | Button used consistently for wiping across meals. |
| retract_between_bites | yes | **(unresolved)** | ✗ | Social settings retract to avoid obstructing the partner in front. |
| bite_dipping_preference | do not dip | **(unresolved)** | ✗ | Meal has no dips, so must be do not dip. |
| wait_before_autocontinue_bite_selection | None | **(unresolved)** | ✗ |  |
| wait_before_autocontinue_task_selection | no autocontinue | **(unresolved)** | ✗ | Social meals consistently use no autocontinue to avoid rushing during conversation. |

### Correction walkthrough

| Step (file) | Direct correction (event) | Correlated prediction changes | Acc |
| --- | --- | --- | --- |
| `135448` init | — initial prediction — | — | **0** |
| `135529` | `wait_before_autocontinue_mealprep` *(?)* | none | 0→**0** |
| `135605` | `robot_speed` fast→**medium** ✓ | none | 0→**1** |
