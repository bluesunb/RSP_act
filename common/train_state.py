from functools import partial
from typing import Any, Callable, Optional, Tuple, Union

import flax
import flax.linen as nn
import jax
import optax
from flax import struct

nonpytree_field = partial(struct.field, pytree_node=False)  # which is not vectorizable
Params = flax.core.FrozenDict[str, Any]


class TrainState(struct.PyTreeNode):
    apply_fn: Callable[..., Any] = nonpytree_field()
    model_def: Any = nonpytree_field()
    params: Params
    extra_variables: Optional[Params]
    tx: Optional[optax.GradientTransformation] = nonpytree_field()
    opt_state: Optional[optax.OptState]
    step: int
    rng: Any = None

    @classmethod
    def create(
        cls,
        *,
        model_def: nn.Module,
        params: Params,
        tx: Optional[optax.GradientTransformation] = None,
        extra_variables: Optional[Params] = None,
        rng: Any = None,
        **kwargs,
    ) -> "TrainState":
        opt_state = None
        if tx is not None:
            opt_state = tx.init(params)

        if extra_variables is None:
            extra_variables = flax.core.FrozenDict()

        return cls(
            step=0,
            apply_fn=model_def.apply,
            model_def=model_def,
            params=params,
            extra_variables=extra_variables,
            tx=tx,
            opt_state=opt_state,
            rng=rng,
            **kwargs,
        )

    def __call__(
        self,
        *args,
        params: Optional[Params] = None,
        extra_variables: Optional[Params] = None,
        method: Union[str, Callable, None] = None,
        **kwargs,
    ):
        if params is None:
            params = self.params

        if extra_variables is None:
            extra_variables = self.extra_variables
        variables = {"params": params, **extra_variables}

        if isinstance(method, str):
            method = getattr(self.model_def, method)

        return self.apply_fn(variables, *args, method=method, **kwargs)

    def apply_gradients(self, grads, **kwargs) -> "TrainState":
        updates, new_opt_state = self.tx.update(grads, self.opt_state, self.params)
        new_params = optax.apply_updates(self.params, updates)
        return self.replace(step=self.step + 1, params=new_params, opt_state=new_opt_state, **kwargs)

    def apply_loss_fn(self, loss_fn, has_aux: bool = False, pmap_axis=None) -> Tuple["TrainState", Any]:
        if has_aux:
            grads, aux = jax.grad(loss_fn, has_aux=True)(self.params)
            if pmap_axis is not None:
                grads = jax.lax.pmean(grads, axis_name=pmap_axis)
                aux = jax.lax.pmean(aux, axis_name=pmap_axis)
            return self.apply_gradients(grads=grads), aux
        else:
            grads = jax.grad(loss_fn)(self.params)
            if pmap_axis is not None:
                grads = jax.lax.pmean(grads, axis_name=pmap_axis)
            return self.apply_gradients(grads=grads), None
