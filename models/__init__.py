from compressai.models import (
    FactorizedPrior,
    ScaleHyperprior,
    MeanScaleHyperprior,
    JointAutoregressiveHierarchicalPriors,
    Cheng2020Anchor,
    Cheng2020Attention,
)


image_models = {
    "factorized": FactorizedPrior,
    "hyperprior": ScaleHyperprior,
    "mbt2018-mean": MeanScaleHyperprior,
    "mbt2018": JointAutoregressiveHierarchicalPriors,
    "cheng2020-anchor": Cheng2020Anchor,
    "cheng2020-attn": Cheng2020Attention,
}


models = {}
models.update(image_models)
