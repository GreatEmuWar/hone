"""Griffe extension that corrects griffe-pydantic's list of a model's fields.

griffe-pydantic takes every class attribute of a model for a field, including unannotated
aliases such as `__lt__ = __le__` on hone's arithmetic mixin, and then fails to render them
for lack of a type. Pydantic itself ignores unannotated and underscored attributes, so this
does too. It must run after griffe_pydantic.
"""
from typing import Any

import griffe


class PydanticFields(griffe.Extension):
    def on_package(self, *, pkg: griffe.Module, **kwargs: Any) -> None:
        self._fix(pkg)

    def _fix(self, obj: griffe.Object) -> None:
        for member in obj.members.values():
            if member.is_alias:
                continue
            if member.is_attribute and "pydantic-field" in member.labels:
                if member.annotation is None or member.name.startswith("_"):
                    member.labels.discard("pydantic-field")
                    member.labels.add("class-attribute")
            elif member.is_class or member.is_module:
                self._fix(member)
