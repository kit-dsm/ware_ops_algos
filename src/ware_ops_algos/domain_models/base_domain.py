from ware_ops_algos.domain_models import LayoutData, StorageLocations, OrdersDomain, Articles, Resources, WarehouseInfo, \
    DynamicInfo, PlannerInformation
from ware_ops_algos.domain_models.planner_information import PlannerInformation
from ware_ops_algos.domain_models.planning import DecisionContext, PlanView


class BaseWarehouseDomain:
    def __init__(self,
                 problem_class: str,
                 objective: str,
                 layout: LayoutData,
                 articles: Articles,
                 orders: OrdersDomain,
                 resources: Resources,
                 storage: StorageLocations,
                 warehouse_info: WarehouseInfo = None,
                 information: PlannerInformation | None = None):
        self.problem_class = problem_class
        self.objective = objective
        self.layout = layout
        self.articles = articles
        self.orders = orders
        self.resources = resources
        self.storage = storage
        self.warehouse_info = warehouse_info
        self.information = information


class SimWarehouseDomain(BaseWarehouseDomain):
    def __init__(self,
                 problem_class: str,
                 objective: str,
                 layout: LayoutData,
                 articles: Articles,
                 orders: OrdersDomain,
                 resources: Resources,
                 storage: StorageLocations,
                 dynamic_warehouse_info: DynamicInfo | None = None,
                 warehouse_info: WarehouseInfo | None = None,
                 information: PlannerInformation | None = None,
                 context: DecisionContext | None = None,
                 plan: PlanView | None = None,
                 memory: object | None = None,
                 operations: object | None = None):
        super().__init__(problem_class,
                         objective,
                         layout,
                         articles,
                         orders,
                         resources,
                         storage,
                         warehouse_info=warehouse_info,
                         information=information)
        self.dynamic_warehouse_info = dynamic_warehouse_info
        self.context = context
        self.plan = plan
        self.memory = memory
        self.operations = operations
