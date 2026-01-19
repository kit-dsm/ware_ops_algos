from dataclasses import dataclass, asdict
from enum import Enum
from typing import Optional, Any

from ware_ops_algos.domain_models.base_domain_object import BaseDomainObject


class ArticleType(str, Enum):
    STANDARD = "standard"


@dataclass
class Article:
    article_id: int
    article_name: Optional[str] = None
    weight: Optional[float] = None
    volume: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Articles(BaseDomainObject):
    tpe: ArticleType
    articles: Optional[list[Article]] = None

    def get_features(self) -> dict[str, Any]:
        features = {}
        features["n_articles"] = len(self.articles)

        if self.articles:
            features["has_weight"] = any(a.weight is not None for a in self.articles)
            features["has_volume"] = any(a.volume is not None for a in self.articles)

        return features
