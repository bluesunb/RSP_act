ALIASE = {
    "adam": "optax.adam",
    "adamw": "optax.adamw",
    "sgd": "optax.sgd",

    "constant": "optax.constant_schedule",
    "cosine_decay": "optax.cosine_decay_schedule",
    "warmup_cosine_decay": "optax.warmup_cosine_decay_schedule",
    "exponential_decay": "optax.exponential_decay",
    "warmup_exponential_decay": "optax.warmup_exponential_decay_schedule",

    "grad_clip_norm": "optax.clip_by_global_norm",
    "grad_clip_value": "optax.clip",
    "grad_noise": "optax.add_noise",
    "weight_decay": "optax.add_decayed_weights",
    "zero_nans": "optax.zero_nans",
}