import distrax
import jax
import jax.numpy as jp


class OneHotCategoricalST(distrax.OneHotCategorical):
    def _sample_n(self, key: jax.Array, n: int):
        samples = super()._sample_n(key, n)
        probs = jp.expand_dims(self.probs, axis=0)
        return probs + jax.lax.stop_gradient(samples - probs)


def make_dist(logits: jax.Array, stoch: int, discrete: int, **kwargs) -> distrax.Distribution:
    if discrete > 0:
        logits = logits.reshape(-1, stoch, discrete)
        dist = distrax.Independent(OneHotCategoricalST(logits), reinterpreted_batch_ndims=1)
    else:
        mean, std = jp.split(logits, 2, axis=-1)
        if "min_std" in kwargs:
            std = jp.maximum(std, kwargs["min_std"])
        if "max_std" in kwargs:
            std = jp.minimum(std, kwargs["max_std"])
        dist = distrax.Normal(mean, std)

    return dist
