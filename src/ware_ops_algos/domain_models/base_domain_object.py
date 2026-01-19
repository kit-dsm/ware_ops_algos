from abc import ABC
from dataclasses import dataclass, fields
from typing import Generic, TypeVar, Any

T = TypeVar("T")


@dataclass
class BaseDomainObject(ABC, Generic[T]):
    tpe: T

    def get_features(self) -> dict[str, Any]:
        features = {}
        for f in fields(self):
            if f.name != "tpe":
                value = getattr(self, f.name)
                if value is not None:
                    features[f.name] = value
        return features

    def get_type_value(self) -> str:
        return str(self.tpe.value)
