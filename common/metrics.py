import time
from functools import partial
from typing import Any, Dict, Tuple, Union, Sequence

import jax
import jax.numpy as jp
from flax.core import FrozenDict, freeze
from flax.struct import PyTreeNode, dataclass, field

nonpytree_node = partial(field, pytree_node=False)


def array_factory(value: Any):
    def _factory():
        return jp.array(value)
    return _factory


@dataclass
class Metric:
    def reset(self) -> "Metric":
        raise NotImplementedError
    
    def update(self, **kwargs) -> "Metric":
        raise NotImplementedError
    
    def compute(self) -> Any:
        raise NotImplementedError
    
    
@dataclass
class Max(Metric):
    argname: str = nonpytree_node(default="values")
    max_value: jax.Array = field(default_factory=array_factory(jp.finfo(jp.float32).min))
    reduce: callable = nonpytree_node(default=lambda x: x)

    def reset(self):
        return self.replace(max_value=jp.full_like(self.max_value, jp.finfo(jp.float32).min))

    def update(self, **kwargs):
        if self.argname not in kwargs:
            raise TypeError(f'Expected keyword argument "{self.argname}"')
        values: jp.ndarray = self.reduce(kwargs[self.argname])
        return self.replace(max_value=jp.maximum(self.max_value, values))

    def compute(self):
        return self.max_value


@dataclass
class Min(Metric):
    argname: str = nonpytree_node(default="values")
    min_value: jax.Array = field(default_factory=array_factory(jp.finfo(jp.float32).max))
    reduce: callable = nonpytree_node(default=lambda x: x)

    def reset(self):
        return self.replace(min_value=jp.finfo(jp.float32).max)

    def update(self, **kwargs):
        if self.argname not in kwargs:
            raise TypeError(f'Expected keyword argument "{self.argname}"')
        values: jp.ndarray = self.reduce(kwargs[self.argname])
        return self.replace(min_value=jp.minimum(self.min_value, values))

    def compute(self):
        return self.min_value


@dataclass
class Sum(Metric):
    argname: str = nonpytree_node(default="values")
    total: jp.ndarray = field(default_factory=array_factory(0))
    reduce: callable = nonpytree_node(default=lambda x: x)

    def reset(self):
        return self.replace(total=jp.zeros_like(self.total, dtype=jp.float32))

    def update(self, **kwargs):
        if self.argname not in kwargs:
            raise TypeError(f'Expected keyword argument "{self.argname}"')
        values: jp.ndarray = kwargs[self.argname]
        return self.replace(total=self.total + self.reduce(values))

    def compute(self):
        return self.total


@dataclass
class Lambda(Metric):
    argname: str = nonpytree_node(default="values")
    update_fn: callable = nonpytree_node(default=lambda v, x: x)
    compute_fn: callable = nonpytree_node(default=lambda v: v)
    values: jax.Array = field(default_factory=array_factory(0))

    def reset(self):
        return self.replace(values=jp.zeros_like(self.values, dtype=jp.float32))

    def update(self, **kwargs):
        if self.argname not in kwargs:
            raise TypeError(f'Expected keyword argument "{self.argname}"')
        values: jp.ndarray = kwargs[self.argname]
        return self.replace(values=self.update_fn(self.values, values))

    def compute(self):
        return self.values
    
    
Identity = lambda argname: Lambda(argname=argname)  # noqa: E731


@dataclass
class Timeit(Metric):
    elapsed: float = field(default=0.0)

    def reset(self) -> Metric:
        return self.replace(elapsed=time.time())

    def update(self, **kwargs):
        return self

    def compute(self):
        return time.time() - self.start
    
    
@dataclass
class Average(Metric):
    argname: str = nonpytree_node(default="values")
    total: jax.Array = field(default_factory=array_factory(0.0))
    count: jax.Array = field(default_factory=array_factory(0.0))
    
    def reset(self):
        return self.replace(
            total=jp.zeros_like(self.total, dtype=self.total.dtype),
            count=jp.zeros_like(self.count, dtype=self.count.dtype),
        )
        
    def update(self, **kwargs):
        if self.argname not in kwargs:
            raise ValueError(f"Missing argument {self.argname} for update 'Average'")
        values: Union[int, float, jax.Array] = kwargs[self.argname]
        total = self.total + (values if isinstance(values, (int, float)) else values.sum())
        count = self.count + (1 if isinstance(values, (int, float)) else values.size)
        return self.replace(total=total, count=count)
    
    def compute(self):
        return self.total / self.count
    
    
@dataclass
class Statistics:
    mean: jax.Array
    standard_error_of_mean: jax.Array
    standard_deviation: jax.Array


@dataclass
class Welford(Metric):
    argname: str = nonpytree_node(default="values")
    count: jax.Array = field(default_factory=array_factory(0))
    mean: jax.Array = field(default_factory=array_factory(0))
    m2: jax.Array = field(default_factory=array_factory(0))

    def reset(self):
        return self.replace(
            count=jp.zeros_like(self.count, dtype=jp.float32),
            mean=jp.zeros_like(self.mean, dtype=jp.float32),
            m2=jp.zeros_like(self.m2, dtype=jp.float32),
        )

    def update(self, **kwargs):
        if self.argname not in kwargs:
            raise TypeError(f'Expected keyword argument "{self.argname}"')
        values: Union[int, float, jax.Array] = kwargs[self.argname]
        count = 1 if isinstance(values, (int, float)) else values.size
        original_count = self.count
        new_count = original_count + count
        delta = (values if isinstance(values, (int, float)) else values.mean()) - self.mean
        new_mean = self.mean + delta * count / new_count
        m2 = 0.0 if isinstance(values, (int, float)) else values.var() * count
        new_m2 = self.m2 + m2 + delta**2 * count * original_count / new_count
        return self.replace(count=new_count, mean=new_mean, m2=new_m2)
    
    def compute(self) -> Any:
        variance = self.m2 / self.count
        stddev = variance**0.5
        std_err_mean = stddev / (self.count**0.5)
        return Statistics(
            mean=self.mean,
            standard_error_of_mean=std_err_mean,
            standard_deviation=stddev,
        )
        

@dataclass
class Accuracy(Average):
    def update(self, *, logits: jax.Array, labels: jax.Array, **_):
        if logits.ndim != labels.ndim + 1 or labels.dtype != jp.int32:
            raise ValueError(
                f"Expected labels.dtype==jnp.int32 and logits.ndim={logits.ndim}==" 
                f"labels.ndim+1={labels.ndim + 1}"
            )
        return super().update(values=(logits.argmax(axis=-1) == labels))
    

class MultiMetric(PyTreeNode):
    _metric_names: Tuple[str, ...] = nonpytree_node(default_factory=tuple)
    metrics: FrozenDict[str, Metric] = field(default_factory=FrozenDict)
    
    @classmethod
    def create(cls, **metrics: Metric) -> "MultiMetric":
        metric_names = tuple(metrics.keys())
        metrics = freeze({name: metric.reset() for name, metric in metrics.items()})
        return cls(_metric_names=metric_names, metrics=metrics)
    
    def reset(self):
        metrics = {}
        for metric_name in self._metric_names:
            metrics[metric_name] = self.metrics[metric_name].reset()
        return self.replace(metrics=freeze(metrics))
    
    def update(self, **kwargs):
        metrics = {}
        for metric_name in self._metric_names:
            metrics[metric_name] = self.metrics[metric_name].update(**kwargs)
        return self.replace(metrics=freeze(metrics))
    
    def compute(self, keys: Sequence[str] = None) -> Dict[str, Any]:
        keys = self._metric_names if keys is None else keys
        return {
            metric_name: self.metrics[metric_name].compute()
            for metric_name in self._metric_names
            if metric_name in keys
        }
        
    def merge(self, other: "MultiMetric"):
        metrics = {**self.metrics, **other.metrics}
        return self.replace(metrics=freeze(metrics))


if __name__ == "__main__":
    logits = jax.random.normal(jax.random.key(0), (5, 2))
    logits2 = jax.random.normal(jax.random.key(1), (5, 2))
    labels = jp.array([1, 1, 0, 1, 0])
    labels2 = jp.array([0, 1, 1, 1, 1])

    batch_loss = jp.array([1, 2, 3, 4])
    batch_loss2 = jp.array([3, 2, 1, 0])

    metrics = MultiMetric.create(acc=Accuracy(), loss=Welford("loss"))
    print(metrics.compute())
    # {'accuracy': Array(nan, dtype=float32), 'loss': Array(nan, dtype=float32)}
    metrics = metrics.update(logits=logits, labels=labels, loss=batch_loss)
    print(metrics.compute())
    # {'accuracy': Array(0.6, dtype=float32), 'loss': Array(2.5, dtype=float32)}
    metrics = metrics.update(logits=logits2, labels=labels2, loss=batch_loss2)
    print(metrics.compute())
    # {'accuracy': Array(0.7, dtype=float32), 'loss': Array(2., dtype=float32)}
    metrics = metrics.reset()
    print(metrics.compute())
    # {'accuracy': Array(nan, dtype=float32), 'loss': Array(nan, dtype=float32)}

    from flax.nnx import metrics

    metrics = metrics.MultiMetric(
        acc=metrics.Accuracy(),
        loss=metrics.Welford("loss"),
    )
    print("========")
    metrics.update(logits=logits, labels=labels, loss=batch_loss)
    print(metrics.compute())

    metrics.update(logits=logits2, labels=labels2, loss=batch_loss2)
    print(metrics.compute())
