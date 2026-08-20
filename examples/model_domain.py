from ware_ops_algos.domain_models import (
    Article,
    Articles,
    ArticleType,
    BaseWarehouseDomain,
    DimensionType,
    LayoutData,
    LayoutNetwork,
    LayoutParameters,
    LayoutType,
    Location,
    Order,
    OrderPosition,
    OrdersDomain,
    OrderType,
    PickCart,
    Resource,
    Resources,
    ResourceType,
    StorageLocations,
    StorageType,
    WarehouseInfo,
    WarehouseInfoType,
)
from ware_ops_algos.domain_models.layout.graph_generators import (
    ShelfStorageGraphGenerator,
    distance_matrix_generator,
    predecessor_matrix_generator,
)


def build_domain() -> BaseWarehouseDomain:
    layout_parameters = LayoutParameters(
        n_aisles=2,
        n_pick_locations=4,
        n_blocks=1,
        dist_top_to_pick_location=1,
        dist_bottom_to_pick_location=1,
        dist_pick_locations=1,
        dist_aisle=2,
        dist_start=1,
        dist_end=1,
        start_location=(0, 0),
        end_location=(0, 5),
        start_connection_point=(1, 0),
        end_connection_point=(1, 5),
    )

    graph_generator = ShelfStorageGraphGenerator(
        n_aisles=layout_parameters.n_aisles,
        n_pick_locations=layout_parameters.n_pick_locations,
        dist_aisle=layout_parameters.dist_aisle,
        dist_pick_locations=layout_parameters.dist_pick_locations,
        dist_aisle_location=layout_parameters.dist_bottom_to_pick_location,
        dist_start=layout_parameters.dist_start,
        dist_end=layout_parameters.dist_end,
        start_location=layout_parameters.start_location,
        end_location=layout_parameters.end_location,
        start_connection_point=layout_parameters.start_connection_point,
        end_connection_point=layout_parameters.end_connection_point,
    )
    graph_generator.populate_graph()
    graph = graph_generator.G

    distance_matrix = distance_matrix_generator(graph)
    start_node = layout_parameters.start_location
    end_node = layout_parameters.end_location
    closest_node = distance_matrix[start_node].drop([start_node, end_node]).idxmin()

    layout = LayoutData(
        tpe=LayoutType.CONVENTIONAL,
        graph_data=layout_parameters,
        layout_network=LayoutNetwork(
            graph=graph,
            distance_matrix=distance_matrix,
            predecessor_matrix=predecessor_matrix_generator(graph),
            closest_node_to_start=closest_node,
            min_aisle_position=0,
            max_aisle_position=5,
            start_node=start_node,
            end_node=end_node,
            node_list=list(graph.nodes),
        ),
    )

    articles = Articles(
        tpe=ArticleType.STANDARD,
        articles=[
            Article(article_id=1, article_name="A", weight=1, volume=1),
            Article(article_id=2, article_name="B", weight=2, volume=1),
            Article(article_id=3, article_name="C", weight=3, volume=2),
            Article(article_id=4, article_name="D", weight=1, volume=1.5),
        ],
    )

    storage = StorageLocations(
        tpe=StorageType.DEDICATED,
        locations=[
            Location(x=1, y=1, article_id=1, amount=100),
            Location(x=1, y=3, article_id=2, amount=100),
            Location(x=2, y=2, article_id=3, amount=100),
            Location(x=2, y=4, article_id=4, amount=100),
        ],
    )
    storage.build_article_location_mapping()

    demands = [
        (1, 1, 2),
        (2, 2, 3),
        (3, 3, 2),
        (4, 4, 4),
    ]
    orders = OrdersDomain(
        tpe=OrderType.GENERAL_DEMAND,
        orders=[
            Order(
                order_id=order_id,
                order_date=float(order_id),
                due_date=100.0,
                order_positions=[
                    OrderPosition(
                        order_number=order_id,
                        article_id=article_id,
                        amount=amount,
                    )
                ],
            )
            for order_id, article_id, amount in demands
        ],
    )

    pick_cart = PickCart(
        n_dimension=3,
        capacities=[6, 10, 10],
        dimensions=[
            DimensionType.ITEMS,
            DimensionType.WEIGHT,
            DimensionType.VOLUME,
        ],
        n_boxes=1,
        box_can_mix_orders=True,
    )
    resources = Resources(
        tpe=ResourceType.HUMAN,
        resources=[
            Resource(
                id=1,
                capacity=6,
                speed=1.0,
                time_per_pick=1.0,
                pick_cart=pick_cart,
            )
        ],
    )

    return BaseWarehouseDomain(
        problem_class="OBRP",
        objective="Distance",
        layout=layout,
        articles=articles,
        orders=orders,
        resources=resources,
        storage=storage,
        warehouse_info=WarehouseInfo(tpe=WarehouseInfoType.OFFLINE),
    )


def main() -> None:
    domain = build_domain()
    cart = domain.resources.resources[0].pick_cart

    print(f"Layout: {domain.layout.graph_data.n_aisles} aisles")
    print(f"Articles: {len(domain.articles.articles)}")
    print(f"Orders: {len(domain.orders.orders)}")
    print(
        "Cart capacities:",
        dict(zip((dimension.value for dimension in cart.dimensions), cart.capacities)),
    )


if __name__ == "__main__":
    main()
