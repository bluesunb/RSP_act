import flax.linen as nn
import jax
import jax.numpy as jp
from einops import rearrange

DEFAULT_CROP_PCT = 0.875
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
IMAGENET_INCEPTION_MEAN = (0.5, 0.5, 0.5)
IMAGENET_INCEPTION_STD = (0.5, 0.5, 0.5)
IMAGENET_DPN_MEAN = (124 / 255, 117 / 255, 104 / 255)
IMAGENET_DPN_STD = tuple([1 / (0.0167 * 255)] * 3)


def positional_embedding(max_len: int, emb_dim: int):
    def init():
        pe = jp.zeros((max_len, emb_dim), dtype=jp.float32)
        position = jp.arange(max_len)[..., None]
        div_term = jp.exp(jp.arange(0, emb_dim, 2) * (-jp.log(10000.0) / emb_dim))
        sin_pos = jp.sin(position * div_term)
        cos_pos = jp.cos(position * div_term)
        pe = jp.stack([sin_pos, cos_pos], axis=-2).transpose(0, 2, 1).reshape(max_len, emb_dim)
        return pe[None, ...]

    return init


def get_2d_sincos_pos_embed(emb_dim: int, grid_size: int, cls_token: bool = False, n_data: int = 1):
    grid_h = jp.arange(grid_size, dtype=jp.float32)
    grid_w = jp.arange(grid_size, dtype=jp.float32)
    grid = jp.meshgrid(grid_w, grid_h)
    grid = jp.stack(grid, axis=0)
    grid = grid.reshape((2, 1, grid_size, grid_size * n_data))

    pos_emb = get_2d_sincos_pos_embed_from_grid(emb_dim, grid)
    if cls_token:
        pos_emb = jp.concatenate([jp.zeros((1, emb_dim)), pos_emb], axis=0)
    return pos_emb


def get_2d_sincos_pos_embed_from_grid(emb_dim: int, grid: jax.Array):
    assert emb_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(emb_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(emb_dim // 2, grid[1])
    emb = jp.concatenate([emb_h, emb_w], axis=-1)
    return emb


def get_1d_sincos_pos_embed_from_grid(emb_dim: int, pos: jax.Array):
    assert emb_dim % 2 == 0
    omega = jp.arange(emb_dim // 2, dtype=jp.float32)
    omega = omega / (emb_dim / 2.0)
    omega = 1.0 / 10000**omega

    pos = pos.reshape(-1)
    out = jp.einsum("i,j->ij", pos, omega)

    emb_sin = jp.sin(out)
    emb_cos = jp.cos(out)
    emb = jp.concatenate([emb_sin, emb_cos], axis=-1)
    return emb


class DropPath(nn.Module):
    rng: jax.random.PRNGKey
    drop_prob: float = 0.0

    @nn.compact
    def __call__(self, x: jax.Array, train: bool = True):
        if not train or self.drop_prob == 0.0:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + jax.random.uniform(self.rng, shape, dtype=x.dtype)
        random_tensor = jp.floor(random_tensor)
        output = x / keep_prob * random_tensor
        return output


class MLP(nn.Module):
    features: int
    hidden_features: int
    act: nn.activation = nn.gelu
    pdrop: float = 0.0

    @nn.compact
    def __call__(self, x: jax.Array, train: bool = True):
        x = nn.Dense(self.hidden_features)(x)
        x = self.act(x)
        x = nn.Dense(self.features)(x)
        x = nn.Dropout(self.pdrop)(x, deterministic=not train)
        return x


class Attention(nn.Module):
    num_heads: int = 8
    qkv_bias: bool = False
    qk_scale: float = None
    attn_pdrop: float = 0.0
    proj_pdrop: float = 0.0
    causal: bool = False

    @nn.compact
    def __call__(self, x: jax.Array, mask: jax.Array = None, train: bool = True):
        dim = x.shape[-1]
        head_dim = dim // self.num_heads
        scale = self.qk_scale or 1.0 / jp.sqrt(head_dim)

        qkv = nn.Dense(3 * dim, use_bias=self.qkv_bias)(x)
        q, k, v = jp.split(qkv, 3, axis=-1)
        q, k, v = map(self.split_heads, (q, k, v))

        attn = jp.einsum("b h i d, b h j d -> b h i j", q, k) * scale

        seq_len = q.shape[2]
        if mask is None:
            mask = jp.ones((1, 1, seq_len, seq_len), dtype=jp.float32)
        
        if self.causal:
            causal_mask = nn.make_causal_mask(jp.ones((seq_len, )), extra_batch_dims=1)
            mask = nn.combine_masks(mask, causal_mask)

        attn = mask * attn + (1 - mask) * -1e9
        attn = nn.softmax(attn, axis=-1)
        attn = nn.Dropout(self.attn_pdrop)(attn, deterministic=not train)

        x = jp.einsum("b h i j, b h j d -> b h i d", attn, v)
        x = self.merge_heads(x)
        x = nn.Dense(dim)(x)
        x = nn.Dropout(self.proj_pdrop)(x, deterministic=not train)
        return x

    def split_heads(self, x: jax.Array):
        return rearrange(x, "b n (h d) -> b h n d", h=self.num_heads)

    def merge_heads(self, x: jax.Array):
        return rearrange(x, "b h n d -> b n (h d)")


class CrossAttention(nn.Module):
    num_heads: int = 8
    qkv_bias: bool = False
    qk_scale: float = None
    attn_pdrop: float = 0.0
    proj_pdrop: float = 0.0
    causal: bool = False

    @nn.compact
    def __call__(self, x: jax.Array, key_val: jax.Array, mask: jax.Array = None, train: bool = False):
        dim = x.shape[-1]
        head_dim = dim // self.num_heads
        scale = self.qk_scale or 1.0 / jp.sqrt(head_dim)

        kv = nn.Dense(2 * dim, use_bias=self.qkv_bias)(key_val)
        k, v = jp.split(kv, 2, axis=-1)
        k, v = map(self.split_heads, (k, v))

        q = nn.Dense(dim, use_bias=self.qkv_bias)(x)
        q = self.split_heads(q)

        attn = jp.einsum("b h i d, b h j d -> b h i j", q, k) * scale

        q_len, k_len = q.shape[2], k.shape[2]
        if mask is None:
            mask = jp.ones((1, 1, q_len, k_len), dtype=jp.float32)
        
        # if self.causal:
        #     causal_mask = nn.make_causal_mask(jp.ones((seq_len, )), extra_batch_dims=1)
        #     mask = nn.combine_masks(mask, causal_mask)

        attn = mask * attn + (1 - mask) * -1e9
        attn = nn.softmax(attn, axis=-1)
        attn = nn.Dropout(self.attn_pdrop)(attn, deterministic=not train)

        x = jp.einsum("b h i j, b h j d -> b h i d", attn, v)
        x = self.merge_heads(x)
        x = nn.Dense(dim)(x)
        x = nn.Dropout(self.proj_pdrop)(x, deterministic=not train)
        return x

    def split_heads(self, x: jax.Array):
        return rearrange(x, "b n (h d) -> b h n d", h=self.num_heads)

    def merge_heads(self, x: jax.Array):
        return rearrange(x, "b h n d -> b n (h d)")


class Block(nn.Module):
    num_heads: int
    mlp_ratio: float = 4.0
    qkv_bias: bool = False
    qk_scale: float = None
    attn_pdrop: float = 0.0
    proj_pdrop: float = 0.0
    drop_prob: float = 0.0
    act: nn.activation = nn.gelu
    norm_layer: nn.Module = nn.LayerNorm
    causal: bool = False

    @nn.compact
    def __call__(self, x: jax.Array, mask: jax.Array = None, train: bool = True):
        droppath = jax.random.split(self.make_rng("droppath"))
        dim = x.shape[-1]
        y = self.norm_layer()(x)
        y = Attention(
            num_heads=self.num_heads,
            qkv_bias=self.qkv_bias,
            qk_scale=self.qk_scale,
            attn_pdrop=self.attn_pdrop,
            proj_pdrop=self.proj_pdrop,
            causal=self.causal
        )(y, mask=mask, train=train)
        x = x + DropPath(droppath[0], self.drop_prob)(y, train=train)

        y = self.norm_layer()(x)
        y = MLP(
            features=dim,
            hidden_features=int(dim * self.mlp_ratio),
            act=self.act,
            pdrop=self.proj_pdrop
        )(y, train=train)
        x = x + DropPath(droppath[1], self.drop_prob)(y, train=train)
        return x


class CrossAttnBlock(nn.Module):
    num_heads: int
    mlp_ratio: float = 4.0
    qkv_bias: bool = False
    qk_scale: float = None
    attn_pdrop: float = 0.0
    proj_pdrop: float = 0.0
    drop_prob: float = 0.0
    act: nn.activation = nn.gelu
    norm_layer: nn.Module = nn.LayerNorm
    causal: bool = False
    
    @nn.compact
    def __call__(self, x: jax.Array, key_val: jax.Array, mask: jax.Array = None, train: bool = True):
        droppath = jax.random.split(self.make_rng("droppath"))
        q = self.norm_layer()(x)
        kv = self.norm_layer()(key_val)
        dim = x.shape[-1]

        y = CrossAttention(
            num_heads=self.num_heads,
            qkv_bias=self.qkv_bias,
            qk_scale=self.qk_scale,
            attn_pdrop=self.attn_pdrop,
            proj_pdrop=self.proj_pdrop,
            causal=self.causal
        )(q, kv, mask, train=train)
        x = x + DropPath(droppath[0], self.drop_prob)(y, train=train)

        y = self.norm_layer()(x)
        y = Attention(
            num_heads=self.num_heads,
            qkv_bias=self.qkv_bias,
            qk_scale=self.qk_scale,
            attn_pdrop=self.attn_pdrop,
            proj_pdrop=self.proj_pdrop,
            causal=self.causal
        )(y, train=train)
        x = x + DropPath(droppath[1], self.drop_prob)(y, train=train)

        y = self.norm_layer()(x)
        x = MLP(
            features=dim,
            hidden_features=int(dim * self.mlp_ratio),
            act=self.act,
            pdrop=self.proj_pdrop
        )(y, train=train)
        return x


class PatchEmbed(nn.Module):
    patch_size: int = 16
    emb_dim: int = 768
    kernel_init: nn.initializers = nn.initializers.xavier_uniform()

    @nn.compact
    def __call__(self, x: jax.Array):
        x = nn.Conv(
            self.emb_dim,
            kernel_size=(self.patch_size, self.patch_size),
            strides=(self.patch_size, self.patch_size),
            kernel_init=self.kernel_init,
        )(x)
        return x.reshape(x.shape[0], -1, x.shape[-1])
