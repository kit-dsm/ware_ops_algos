from dataclasses import dataclass
from enum import Enum

from ware_ops_algos.domain_models import BaseDomainObject


class WarehouseInfoType(str, Enum):
    ONLINE = "online"
    OFFLINE = "offline"


@dataclass
class WarehouseInfo(BaseDomainObject):
    tpe: WarehouseInfoType


