from pathlib import Path
from typing import List, Optional

from feeding_deployment.preference_learning.config.preference_bundle import PreferenceDim, PREFERENCE_BUNDLE

SYSTEM_DESCRIPTION_PATH = Path(__file__).parent / "system_description.txt"

def render_preference_dimensions(bundle: List[PreferenceDim]) -> str:
    lines = []

    for i, dim in enumerate(bundle, start=1):
        lines.append(f"{i}. {dim.label}")
        lines.append(f"Field name: {dim.field}")
        if getattr(dim, "kind", "categorical") == "color":
            lines.append(
                "Value type: HSV color. Emit an object "
                '{"h": <0-179>, "s": <0-255>, "v": <0-255>, "range": <0.0-1.0>}.'
            )
        elif getattr(dim, "kind", "categorical") == "nav_offset":
            lines.append(
                "Value type: navigation pose offset. Emit an object "
                '{"dx": <-0.5-0.5 m>, "dy": <-0.5-0.5 m>, "dyaw": <-0.785-0.785 rad>}.'
            )
        elif getattr(dim, "kind", "categorical") == "text":
            lines.append("Value type: free-text string (a single concise natural-language sentence).")
        else:
            lines.append(f"Allowed options: [{', '.join(dim.options)}]")
        lines.append(dim.description)
        lines.append("")  # blank line between dimensions

    return "\n".join(lines).strip()

def get_system_description_prompt(dims: Optional[List[PreferenceDim]] = None) -> str:
    """System description with dimension definitions. ``dims`` restricts which
    dimensions are rendered (e.g. LLM-generated dims only for data generation,
    or a single dim for per-dimension prediction); default is the full bundle."""
    template = SYSTEM_DESCRIPTION_PATH.read_text(encoding="utf-8")

    preference_dimensions = render_preference_dimensions(PREFERENCE_BUNDLE if dims is None else dims)

    return template.format(
        preference_dimensions=preference_dimensions
    )
    
if __name__ == "__main__":
    print(get_system_description_prompt())