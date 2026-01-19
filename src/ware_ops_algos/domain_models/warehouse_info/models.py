from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from ware_ops_algos.domain_models import BaseDomainObject


class WarehouseInfoType(str, Enum):
    ONLINE = "online"
    OFFLINE = "offline"


@dataclass
class WarehouseInfo(BaseDomainObject):
    tpe: WarehouseInfoType


