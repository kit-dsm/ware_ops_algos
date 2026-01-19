from ware_ops_algos.domain_models import LayoutData, StorageLocations, OrdersDomain, Articles, Resources, WarehouseInfo


class BaseWarehouseDomain:
    def __init__(self,
                 problem_class: str,
                 objective: str,
                 layout: LayoutData,
                 articles: Articles,
                 orders: OrdersDomain,
                 resources: Resources,
                 storage: StorageLocations,
                 warehouse_info: WarehouseInfo = None):
        self.problem_class = problem_class
        self.objective = objective
        self.layout = layout
        self.articles = articles
        self.orders = orders
        self.resources = resources
        self.storage = storage
        self.warehouse_info = warehouse_info
