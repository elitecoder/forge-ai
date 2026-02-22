"""Safe prompt template rendering via {{var}} substitution."""

import re

_VAR_RE = re.compile(r"\{\{(\w+)\}\}")


def render(template: str, variables: dict[str, str]) -> str:
    def replacer(m: re.Match) -> str:
        key = m.group(1)
        return variables.get(key, m.group(0))

    return _VAR_RE.sub(replacer, template)
