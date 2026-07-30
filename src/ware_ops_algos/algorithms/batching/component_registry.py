from ware_ops_algos.algorithms.batching.seed_batching import (
    RandomSeed,
    MostPositionsSeed,
    FewestPositionsSeed,
    ClosestToDepotSeed,
    SharedArticlesSimilarity,
    MinDistanceSimilarity,
)
from ware_ops_algos.algorithms.batching.moves import (
    SwapNeighborhood,
    ShiftNeighborhood,
)

SEED_CRITERIA = {
    "RANDOM": RandomSeed,
    "MOST_POSITIONS": MostPositionsSeed,
    "FEWEST_POSITIONS": FewestPositionsSeed,
    "CLOSEST_TO_DEPOT": ClosestToDepotSeed,
}

SIMILARITY_MEASURES = {
    "SHARED_ARTICLES": SharedArticlesSimilarity,
    "MIN_DISTANCE": MinDistanceSimilarity,
}

NEIGHBORHOODS = {
    "SWAP": SwapNeighborhood,
    "SHIFT": ShiftNeighborhood,
}