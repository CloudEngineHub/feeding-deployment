### Categorical (20)

| Dimension | Model predicted | User's actual preference |  | Reason |
| --- | --- | --- | --- | --- |
| robot_speed | medium | **fast** | ✗ | No corrections yet; medium is the neutral default. |
| microwave_time | 1 min | **(unresolved)** | ✗ | Chicken nuggets are served warm, so a short reheat is the default. |
| skewering_axis | perpendicular to major axis | **(unresolved)** | ✗ | User can open mouth wide, so perpendicular skewering for higher acquisition success is the default. |
| confirm_feeding_pickup | yes (with auto-continue countdown) | **(unresolved)** | ✗ | No history; users typically start with confirmation on with countdown, personal setting allows auto-continue. |
| confirm_navigation_arrival | yes (with auto-continue countdown) | **(unresolved)** | ✗ | Users start with navigation confirmation on to refine parking; countdown default. |
| confirm_manipulation | yes (with auto-continue countdown) | **(unresolved)** | ✗ | Users start with manipulation confirmation on to refine perception; countdown default. |
| transfer_mode | outside mouth transfer | **(unresolved)** | ✗ | Deployment only performs outside-mouth transfer and no explicit request otherwise. |
| outside_mouth_distance | medium | **(unresolved)** | ✗ | User can lean forward comfortably; medium is a neutral default. |
| convey_robot_ready_for_initiating_transfer | speech + LED | **(unresolved)** | ✗ | Redundant speech + LED cue is a safe default with no contrary evidence. |
| detect_user_ready_for_initiating_transfer_feeding | open mouth | **(unresolved)** | ✗ | User can open mouth wide; in personal (non-social) setting open mouth is reliable. |
| detect_user_ready_for_initiating_transfer_drinking | open mouth | **(unresolved)** | ✗ | Open mouth is reliable in a personal setting given good head/mouth control. |
| detect_user_ready_for_initiating_transfer_wiping | button | **(unresolved)** | ✗ | Mouth-opening is ambiguous for wiping, so button is the safer default given user can press buttons. |
| convey_robot_ready_for_completing_transfer | speech + LED | **(unresolved)** | ✗ | Redundant speech + LED cue default with no contrary evidence. |
| detect_user_completed_transfer_feeding | perception | **(unresolved)** | ✗ | Force-torque perception is very reliable and the default for bites. |
| detect_user_completed_transfer_drinking | perception | **(unresolved)** | ✗ | User has good head control for the head-nod perception default in a personal setting. |
| detect_user_completed_transfer_wiping | perception | **(unresolved)** | ✗ | User has good head control for head-nod perception in a personal setting. |
| retract_between_bites | no | **(unresolved)** | ✗ | Eating alone in a personal setting, skipping retraction saves time. |
| bite_dipping_preference | do not dip | **(unresolved)** | ✗ | Meal has no dips/sauces, so hard rule forces do not dip. |
| wait_before_autocontinue_bite_selection | None | **(unresolved)** | ✗ |  |
| wait_before_autocontinue_task_selection | 30 sec | **(unresolved)** | ✗ | Neutral default with no evidence of chatting or rushing. |

### Correction walkthrough

| Step (file) | Direct correction (event) | Correlated prediction changes | Acc |
| --- | --- | --- | --- |
| `104018` init | — initial prediction — | — | **0** |
| `104052` | `wait_before_autocontinue_mealprep` *(?)* | none | 0→**0** |
| `104130` | `wait_task` 60 sec→**no autocontinue** ✓ | none | 0→**0** |
| `104204` | `outside_mouth_distance` medium→**near** ✓ | ~ `robot_speed` medium→slow (✗→✗) | 0→**0** |
| `104246` | `robot_speed` slow→**medium** ✓ | none | 0→**0** |
| `104324` | `wait_before_autocontinue_feeding_pickup` *(?)* | none | 0→**0** |
| `104403` | `detect_user_completed_transfer_wiping` perception→**button** ✓ | none | 0→**0** |
| `104441` | `detect_user_ready_for_initiating_transfer_wiping` button→**open mouth** ✓ | none | 0→**0** |
| `104514` | `detect_user_completed_transfer_drinking` button→**perception** ✓ | none | 0→**0** |
| `104544` | `confirm_feeding_pickup` yes (without any auto-continue)→**yes (with auto-continue countdown)** ✓ | none | 0→**0** |
| `104613` | — | none | 0→**0** |
| `104649` | `wait_task` no autocontinue→**60 sec** ✓ | none | 0→**0** |
| `104727` | `detect_user_completed_transfer_drinking` perception→**button** ✓ | none | 0→**0** |
| `104816` | `convey_init` speech+LED→**no cue** ✓ | none | 0→**0** |
| `104856` | `wait_before_autocontinue_mealprep` *(?)* | none | 0→**0** |
| `104938` | `microwave_time` 2 min→**1 min** ✓ | none | 0→**0** |
| `105022` | `detect_user_ready_for_initiating_transfer_wiping` button→**open mouth** ✓ | none | 0→**0** |
| `105058` | `detect_completed_feeding` perception→**button** ✓ | none | 0→**0** |
| `105120` | — | none | 0→**0** |
| `105203` | `robot_speed` medium→**fast** ✓ | none | 0→**1** |
| `105235` | `wait_before_autocontinue_mealprep` *(?)* | none | 1→**1** |
| `111156` | — | none | 1→**0** |
| `111234` | `wait_before_autocontinue_feeding_pickup` *(?)* | none | 0→**0** |
| `111323` | `detect_user_completed_transfer_drinking` perception→**button** ✓ | none | 0→**0** |
| `111400` | `convey_init` speech+LED→**no cue** ✓ | none | 0→**0** |
| `111435` | `retract_between_bites` no→**yes** ✓ | none | 0→**0** |
| `111525` | — | none | 0→**0** |
| `111623` | `bite_ordering` *(TEXT)* | none | 0→**0** |
| `111719` | `bite_dipping_preference` less→**more** ✓ | none | 0→**0** |
| `111807` | `convey_complete` no cue→**LED** ✓ | none | 0→**0** |
| `111859` | `wait_before_autocontinue_feeding_pickup` *(?)* | none | 0→**0** |
| `111948` | `confirm_manipulation` yes (with auto-continue countdown)→**yes (without any auto-continue)** ✓ | none | 0→**0** |
| `112042` | `wait_before_autocontinue_mealprep` *(?)* | none | 0→**0** |
| `112130` | `confirm_navigation_arrival` yes (with auto-continue countdown)→**yes (without any auto-continue)** ✓ | none | 0→**0** |
| `112214` | `robot_speed` medium→**fast** ✓ | none | 0→**1** |
| `112246` | — | none | 1→**0** |
| `112329` | `wait_before_autocontinue_mealprep` *(?)* | **+** `robot_speed` medium→fast ✓ | 0→**1** |
| `112408` | `bite_ordering` *(TEXT)* | none | 1→**1** |
| `112440` | `wait_before_autocontinue_feeding_pickup` *(?)* | none | 1→**1** |
| `112508` | — | none | 1→**1** |
| `112540` | `robot_speed` fast→**medium** ✓ | none | 1→**0** |
| `112615` | `microwave_time` no microwave→**1 min** ✓ | none | 0→**0** |
| `112646` | `bite_ordering` *(TEXT)* | none | 0→**0** |
| `112716` | — | none | 0→**0** |
| `112747` | `bite_ordering` *(TEXT)* | none | 0→**0** |
| `112815` | `wait_before_autocontinue_feeding_pickup` *(?)* | none | 0→**0** |
| `112845` | `wait_before_autocontinue_mealprep` *(?)* | none | 0→**0** |
| `112915` | — | **+** `robot_speed` medium→fast ✓ | 0→**1** |
| `112951` | `robot_speed` fast→**medium** ✓ | none | 1→**0** |
| `113035` | `wait_before_autocontinue_mealprep` *(?)* | none | 0→**0** |
| `113137` | `wait_task` 60 sec→**no autocontinue** ✓ | none | 0→**0** |
| `113216` | `confirm_navigation_arrival` yes (with auto-continue countdown)→**yes (without any auto-continue)** ✓ | none | 0→**0** |
| `113258` | — | none | 0→**0** |
| `113344` | `detect_user_completed_transfer_drinking` perception→**button** ✓ | none | 0→**0** |
| `113429` | `retract_between_bites` yes→**no** ✓ | none | 0→**0** |
| `113504` | `convey_complete` no cue→**speech+LED** ✓ | none | 0→**0** |
| `133138` | — | none | 0→**0** |
| `133222` | `bite_ordering` *(TEXT)* | none | 0→**0** |
| `133300` | `retract_between_bites` yes→**no** ✓ | none | 0→**0** |
| `133356` | `microwave_time` 2 min→**1 min** ✓ | none | 0→**0** |
| `133427` | `wait_task` no autocontinue→**60 sec** ✓ | none | 0→**0** |
| `133459` | `wait_before_autocontinue_mealprep` *(?)* | none | 0→**0** |
| `133533` | `convey_complete` no cue→**speech+LED** ✓ | none | 0→**0** |
| `133608` | `detect_user_completed_transfer_drinking` perception→**button** ✓ | none | 0→**0** |
| `133635` | — | **+** `robot_speed` medium→fast ✓ | 0→**1** |
| `133701` | — | **−** `robot_speed` fast→**medium** ✗ | 1→**0** |
| `133740` | `detect_completed_feeding` button→**perception** ✓ | none | 0→**0** |
| `133814` | `wait_before_autocontinue_mealprep` *(?)* | none | 0→**0** |
| `133859` | `confirm_manipulation` yes (with auto-continue countdown)→**no** ✓ | none | 0→**0** |
| `133937` | `confirm_navigation_arrival` yes (with auto-continue countdown)→**no** ✓ | none | 0→**0** |
| `134020` | `confirm_feeding_pickup` no→**yes (with auto-continue countdown)** ✓ | none | 0→**0** |
| `134050` | `detect_user_completed_transfer_wiping` button→**perception** ✓ | **+** `robot_speed` medium→fast ✓ | 0→**1** |
| `134121` | `robot_speed` fast→**medium** ✓ | none | 1→**0** |
| `134148` | — | none | 0→**0** |
| `134225` | `confirm_navigation_arrival` yes (with auto-continue countdown)→**no** ✓ | none | 0→**0** |
| `134301` | `detect_user_completed_transfer_wiping` perception→**button** ✓ | none | 0→**0** |
| `134335` | — | none | 0→**0** |
| `134428` | `confirm_feeding_pickup` yes (without any auto-continue)→**yes (with auto-continue countdown)** ✓ | none | 0→**0** |
| `134520` | `confirm_manipulation` yes (with auto-continue countdown)→**no** ✓ | none | 0→**0** |
| `134607` | `robot_speed` medium→**fast** ✓ | none | 0→**1** |
| `134653` | `wait_before_autocontinue_mealprep` *(?)* | none | 1→**1** |
| `134725` | `wait_task` 30 sec→**no autocontinue** ✓ | none | 1→**1** |
| `134759` | — | none | 1→**1** |
| `134837` | `bite_ordering` *(TEXT)* | none | 1→**1** |
| `134912` | `robot_speed` fast→**medium** ✓ | none | 1→**0** |
| `134952` | `confirm_navigation_arrival` yes (with auto-continue countdown)→**no** ✓ | none | 0→**0** |
| `135022` | — | none | 0→**0** |
| `135105` | `confirm_navigation_arrival` yes (without any auto-continue)→**no** ✓ | **+** `robot_speed` medium→fast ✓ | 0→**1** |
| `135143` | `robot_speed` fast→**medium** ✓ | none | 1→**0** |
| `135210` | — | none | 0→**0** |
| `135246` | `detect_user_completed_transfer_wiping` button→**perception** ✓ | none | 0→**0** |
| `135314` | — | none | 0→**0** |
| `135344` | `robot_speed` medium→**fast** ✓ | none | 0→**1** |
| `135417` | `wait_before_autocontinue_mealprep` *(?)* | none | 1→**1** |
| `135448` | — | none | 1→**1** |
| `135529` | `wait_before_autocontinue_mealprep` *(?)* | none | 1→**1** |
| `135605` | `robot_speed` fast→**medium** ✓ | none | 1→**0** |
| `135644` | — | none | 0→**0** |
| `135722` | `robot_speed` medium→**fast** ✓ | none | 0→**1** |
| `135805` | `wait_before_autocontinue_mealprep` *(?)* | none | 1→**1** |
| `135845` | `bite_ordering` *(TEXT)* | none | 1→**1** |
| `135916` | `bite_dipping_preference` less→**more** ✓ | none | 1→**1** |
| `135939` | — | none | 1→**0** |
| `140011` | — | none | 0→**0** |
| `140039` | `robot_speed` medium→**fast** ✓ | none | 0→**1** |
