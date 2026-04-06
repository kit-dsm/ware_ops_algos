SUBPROBLEMS = {
    "SPRP": {
        "objectives": ["distance", "picking_time"],
        "variables": ["item_assignment", "order_selection", "routing"]
    },
    "SPRP-SS": {
            "objectives": ["distance", "picking_time"],
            "variables": ["item_assignment", "order_selection", "routing"]
        },
    "OBRP": {
        "objectives": ["tardiness", "picking_time", "cost", "completion_time"],
        "variables": ["item_assignment", "batching", "routing", "batching_routing"]
    },
    "OBSRP": {
        "objectives": ["tardiness", "picking_time", "cost", "completion_time"],
        "variables": ["item_assignment", "batching", "sequencing", "routing"]
    }
}
