from compressai.zoo.image import (
    bmshj2018_factorized,
    bmshj2018_hyperprior,
    cheng2020_anchor,
    cheng2020_attn,
    mbt2018,
    mbt2018_mean,
)


image_models = {
    "bmshj2018-factorized": bmshj2018_factorized,
    "bmshj2018-hyperprior": bmshj2018_hyperprior,
    "mbt2018-mean": mbt2018_mean,
    "mbt2018": mbt2018,
    "cheng2020-anchor": cheng2020_anchor,
    "cheng2020-attn": cheng2020_attn,
}


models = {}
models.update(image_models)
