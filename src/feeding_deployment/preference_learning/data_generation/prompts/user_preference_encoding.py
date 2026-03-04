from pathlib import Path
from typing import List

from feeding_deployment.preference_learning.config.physical_capabilities import PHYSICAL_CAPABILITY_PROFILES
from feeding_deployment.preference_learning.data_generation.prompts.system_description import get_system_description_prompt

USER_PREFERENCE_ENCODING_PROMPT_PATH = Path(__file__).parent / "user_preference_encoding.txt"

def get_user_preference_encoding_prompt(physical_profile_label: str) -> str:
    template = USER_PREFERENCE_ENCODING_PROMPT_PATH.read_text(encoding="utf-8")
    system_description = get_system_description_prompt()
    
    PHYSICAL_CAPABILITY_BY_LABEL = {p.label: p for p in PHYSICAL_CAPABILITY_PROFILES}
    physical_profile_description = PHYSICAL_CAPABILITY_BY_LABEL[physical_profile_label].description

    return template.format(
        system_description=system_description,
        physical_profile=physical_profile_description
    )
    
if __name__ == "__main__":
    physical_profile = PHYSICAL_CAPABILITY_PROFILES[0]  # just take the first one as an example
    prompt = get_user_preference_encoding_prompt(physical_profile.label)
    print(prompt)