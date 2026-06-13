TAXONOMY = {
    "SPRP": {
        "objectives": ["distance", "picking_time"],
        "variables": ["item_assignment", "order_selection", "routing"]
    },
    "SPRP-SS": {
            "objectives": ["distance", "picking_time"],
            "variables": ["item_assignment", "order_selection", "routing"]
        },
    "OBRP": {
        "objectives": ["distance", "picking_time"],
        "variables": ["item_assignment", "batching", "routing", "batching_routing"]
    },
    "OBSRP": {
        "objectives": ["distance", "makespan", "tardiness", "picking_time", "completion_time"],
        "variables": ["item_assignment", "batching", "scheduling", "routing"]
    }
}
