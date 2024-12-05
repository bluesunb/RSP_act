import distrax
import flax.linen as nn
import jax
import jax.numpy as jp
from einops import rearrange

from .distribution import make_dist
from .vision_transformer import Block, CrossAttnBlock, PatchEmbed, get_2d_sincos_pos_embed

RNG_KEYS = ["dropout", "droppath", "noise", "mask", "sample"]


def patchify(img: jax.Array, patch_size: int):
    return rearrange(img, "b (h p1) (w p2) c -> b (h w) (p1 p2 c)", p1=patch_size, p2=patch_size)


def unpatchify(patches: jax.Array, img_size: int, patch_size: int):
    patched_img_size = img_size // patch_size
    return rearrange(
        patches,
        "b (h w) (p1 p2 c) -> b (h p1) (w p2) c",
        h=patched_img_size,
        w=patched_img_size,
        p1=patch_size,
        p2=patch_size,
    )


def random_masking(rng: jax.random.PRNGKey, x: jax.Array, mask_rate: float = 0.0):
    bs, num_patches, emb_dim = x.shape
    keep_len = int(num_patches * mask_rate)

    if mask_rate == 0:
        mask = None
        ids_restore = jp.tile(jp.arange(num_patches), (bs, 1))
        return x, mask, ids_restore

    noise = jax.random.uniform(rng, (bs, num_patches))
    ids_shuffle = jp.argsort(noise, axis=-1)
    ids_restore = jp.argsort(ids_shuffle, axis=-1)

    ids_keep = ids_shuffle[:, :keep_len]
    ids_keep = jp.expand_dims(ids_keep, axis=-1)
    ids_keep = jp.tile(ids_keep, (1, 1, emb_dim))

    x_masked = jp.take_along_axis(x, ids_keep, axis=1)

    mask = jp.concatenate([jp.ones((bs, keep_len)), jp.zeros((bs, num_patches - keep_len))], axis=-1)
    mask = jp.take_along_axis(mask, ids_restore, axis=-1)
    return x_masked, mask, ids_restore


class Encoder(nn.Module):
    img_size: int = 224
    patch_size: int = 16
    emb_dim: int = 1024
    depth: int = 24
    num_heads: int = 16
    mlp_ratio: float = 4.0
    pdrop: float = 0.0

    @nn.compact
    def __call__(self, img: jax.Array, mask_rate: float = 0.0, train: bool = True):
        num_patches = (self.img_size // self.patch_size) ** 2
        pos_emb = self.variable(
            "pos_emb", "enc_pos_emb", get_2d_sincos_pos_embed, self.emb_dim, int(num_patches**0.5), True
        )
        pos_emb = jp.expand_dims(pos_emb.value, axis=0).astype(jp.float32)

        x = PatchEmbed(self.patch_size, self.emb_dim)(img)  # (bs, num_patches, emb_dim)
        x = x + pos_emb[:, 1:]
        x, mask, ids_restore = random_masking(self.make_rng("mask"), x, mask_rate)

        cls_token = self.param("cls_token", nn.initializers.normal(stddev=0.02), (1, 1, self.emb_dim))
        cls_token = jp.tile(cls_token, (x.shape[0], 1, 1))
        x = jp.concatenate([cls_token, x], axis=1)  # (bs, num_patches + 1, emb_dim)

        for _ in range(self.depth):
            x = Block(
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=True,
                qk_scale=None,
                attn_pdrop=self.pdrop,
                proj_pdrop=self.pdrop,
                drop_prob=self.pdrop,
            )(x, train=train)

        x = nn.LayerNorm()(x)
        return x, mask, ids_restore


class Decoder(nn.Module):
    # image reconstruction
    img_recon: bool = False
    img_size: int = 224
    patch_size: int = 16
    
    # action reconstruction
    action_recon: bool = False
    act_size: int = 0
    seq_len: int = 0
    
    dec_emb_dim: int = 1024
    dec_depth: int = 24
    dec_num_heads: int = 16
    stoch: int = 32
    discrete: int = 32
    mlp_ratio: float = 4.0
    

    def setup(self):
        num_patches = (self.img_size // self.patch_size) ** 2
        self.pos_emb = self.variable(
            "pos_emb", "dec_pos_emb", get_2d_sincos_pos_embed, self.dec_emb_dim, int(num_patches**0.5), True
        )
        self.dec_embed_mae = nn.Dense(self.dec_emb_dim, kernel_init=nn.initializers.xavier_uniform())
        self.dec_embed_deter = nn.Dense(self.dec_emb_dim, kernel_init=nn.initializers.xavier_uniform())
        self.dec_embed_stoch = nn.Dense(self.dec_emb_dim, kernel_init=nn.initializers.xavier_uniform())
        self.dec_blocks = [
            CrossAttnBlock(
                num_heads=self.dec_num_heads,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=True,
                qk_scale=None,
                attn_pdrop=0.0,
                proj_pdrop=0.0,
                drop_prob=0.0,
            )
            for _ in range(self.dec_depth)
        ]
        self.mask_token = self.param("mask_token", nn.initializers.normal(stddev=0.02), (1, 1, self.dec_emb_dim))
        
        # action reconstruction
        self.dec_norm = nn.LayerNorm()
        self.act_pred = nn.Dense(self.act_size, kernel_init=nn.initializers.xavier_uniform())
        self.term_pred = nn.Dense(1, kernel_init=nn.initializers.xavier_uniform())
        
        self.dec_pred = nn.Dense(self.patch_size**2 * 3, kernel_init=nn.initializers.xavier_uniform())

    def predict(self, hidden_state: jax.Array, latent: jax.Array, train: bool = True):
        hidden_state = self.dec_embed_deter(hidden_state) + self.pos_emb.value
        latent = latent.reshape(*latent.shape[:-2], 1, -1)
        latent = self.dec_embed_stoch(latent)
        features = jp.concatenate([hidden_state, latent], axis=1)

        mask_token = jp.tile(self.mask_token, (*hidden_state.shape[:2], 1))
        x = mask_token + self.pos_emb.value

        for block in self.dec_blocks:
            x = block(x, key_val=features, train=train)
        x = self.dec_norm(x)
        x = self.dec_pred(x)
        return x[:, 1:]

    def predict_masked(self, hidden_state: jax.Array, ids_restore: jax.Array, train: bool = True):
        pad_size = ids_restore.shape[1] - hidden_state.shape[1] + 1
        hidden_state = self.dec_embed_mae(hidden_state)
        mask_token = jp.tile(self.mask_token, (hidden_state.shape[0], pad_size, 1))
        cls_token, hidden_state = jp.split(hidden_state, (1,), axis=1)

        ids_restore = jp.expand_dims(ids_restore, axis=-1)
        ids_restore = jp.tile(ids_restore, (1, 1, hidden_state.shape[-1]))

        hidden_state = jp.concatenate([hidden_state, mask_token], axis=1)
        hidden_state = jp.take_along_axis(hidden_state, ids_restore, axis=1)
        hidden_state = jp.concatenate([cls_token, hidden_state], axis=1)
        hidden_state = hidden_state + self.pos_emb.value

        x = jp.tile(self.mask_token, (*hidden_state.shape[:2], 1)) + self.pos_emb.value
        for block in self.dec_blocks:
            x = block(x, key_val=hidden_state, train=train)
        x = self.dec_norm(x)
        x = self.dec_pred(x)
        return x[:, 1:]


class RSP(nn.Module):
    img_size: int = 224
    patch_size: int = 16
    emb_dim: int = 1024
    depth: int = 24
    num_heads: int = 16
    stoch: int = 32
    discrete: int = 32
    dec_emb_dim: int = 512
    dec_depth: int = 8
    dec_num_heads: int = 8
    mlp_ratio: float = 4.0
    mask_rate: float = 0.75
    noise_scale: float = 0.5

    def setup(self):
        self.encoder = Encoder(
            img_size=self.img_size,
            patch_size=self.patch_size,
            emb_dim=self.emb_dim,
            depth=self.depth,
            num_heads=self.num_heads,
            mlp_ratio=self.mlp_ratio,
        )

        self.decoder = Decoder(
            img_size=self.img_size,
            patch_size=self.patch_size,
            dec_emb_dim=self.dec_emb_dim,
            dec_depth=self.dec_depth,
            dec_num_heads=self.dec_num_heads,
            mlp_ratio=self.mlp_ratio,
        )

        stoch_size = self.stoch * self.discrete if self.discrete > 0 else self.stoch * 2
        self.prior_layer = nn.Sequential([nn.Dense(self.emb_dim * 2), nn.relu, nn.Dense(stoch_size)])
        self.posterior_layer = nn.Sequential([nn.Dense(self.emb_dim * 2), nn.relu, nn.Dense(stoch_size)])

    def __call__(self, src_img: jax.Array, tgt_img: jax.Array, train: bool = True):
        noise_rng = self.make_rng("noise")
        noise = jax.random.normal(noise_rng, tgt_img.shape) * self.noise_scale

        src_emb, _, _ = self.encoder(src_img, mask_rate=0.0, train=train)
        tgt_emb, _, _ = self.encoder(tgt_img + noise, mask_rate=0.0, train=train)

        sample_rng = self.make_rng("sample")
        post_rng, prior_rng = jax.random.split(sample_rng)

        post_hidden = jp.concatenate([src_emb[:, 0], tgt_emb[:, 0]], axis=-1)
        post_logits = self.posterior_layer(post_hidden)
        post_dist = make_dist(post_logits, self.stoch, self.discrete)
        post_latent = post_dist.sample(seed=post_rng)

        prior_hidden = src_emb[:, 0]
        prior_logits = self.prior_layer(prior_hidden)
        prior_dist = make_dist(prior_logits, self.stoch, self.discrete)
        prior_latent = prior_dist.sample(seed=prior_rng)

        tgt_pred_post = self.decoder.predict(src_emb, post_latent, train=train)
        tgt_pred_prior = self.decoder.predict(src_emb, prior_latent, train=train)
        tgt_hidden, mask, ids_restore = self.encoder(tgt_img, mask_rate=self.mask_rate, train=train)
        tgt_masked_pred_post = self.decoder.predict_masked(tgt_hidden, ids_restore, train=train)

        return {
            "tgt_pred_prior": tgt_pred_prior,
            "tgt_pred_post": tgt_pred_post,
            "tgt_masked_pred_post": tgt_masked_pred_post,
            "mask": mask,
            "post_dist": post_dist,
            "prior_dist": prior_dist,
        }

    def encode(self, img: jax.Array):
        src_emb, _, _ = self.encoder(img, mask_rate=0.0, train=False)
        prior_hidden = src_emb[:, 0]
        prior_logits = self.prior_layer(prior_hidden)
        prior_dist = make_dist(prior_logits, self.stoch, self.discrete)
        prior_latent = prior_dist.mode()
        return prior_latent


def img_recon_loss(img: jax.Array, pred: jax.Array, patch_size: int, normalize: bool = False, mask: jax.Array = None):
    tgt_img = patchify(img, patch_size)
    if normalize:
        mean = tgt_img.mean(-1, keepdims=True)
        var = tgt_img.var(-1, keepdims=True)
        tgt_img = (tgt_img - mean) / jp.maximum(jp.sqrt(var), 1e-6)

    recon_loss = jp.square(tgt_img - pred)
    if mask is not None:
        recon_loss = recon_loss.mean(-1)
        recon_loss = (recon_loss * mask).sum() / mask.sum()
    else:
        recon_loss = recon_loss.mean()

    return recon_loss


def kl_dist_loss(post_dist: distrax.Distribution, prior_dist: distrax.Distribution, freebit: float, balance: float):
    post_to_prior = post_dist.kl_divergence(prior_dist)
    prior_to_post = prior_dist.kl_divergence(post_dist)
    kl_value = (balance * post_to_prior + (1 - balance) * prior_to_post).mean()
    kl_loss = jp.maximum(kl_value, jp.full_like(kl_value, freebit))
    return kl_loss, kl_value


def rsp_tmp_debug(img_size, **kwargs):
    model = RSP(
        patch_size=16,
        emb_dim=128,
        depth=2,
        num_heads=8,
        dec_emb_dim=256,
        dec_depth=2,
        dec_num_heads=8,
        mlp_ratio=4,
        img_size=img_size,
        **kwargs,
    )
    return model


def rsp_vit_small_patch8_dec512d8b(img_size, **kwargs):
    model = RSP(
        patch_size=8,
        emb_dim=384,
        depth=12,
        num_heads=6,
        dec_emb_dim=512,
        dec_depth=8,
        dec_num_heads=16,
        mlp_ratio=4,
        img_size=img_size,
        **kwargs,
    )
    return model


def rsp_vit_small_patch16_dec512d8b(img_size, **kwargs):
    model = RSP(
        patch_size=16,
        emb_dim=384,
        depth=12,
        num_heads=6,
        dec_emb_dim=512,
        dec_depth=8,
        dec_num_heads=16,
        mlp_ratio=4,
        img_size=img_size,
        **kwargs,
    )
    return model


def rsp_vit_base_patch16_dec512d8b(img_size, **kwargs):
    model = RSP(
        patch_size=16,
        emb_dim=768,
        depth=12,
        num_heads=12,
        dec_emb_dim=512,
        dec_depth=8,
        dec_num_heads=16,
        mlp_ratio=4,
        img_size=img_size,
        **kwargs,
    )
    return model


def rsp_vit_large_patch16_dec512d8b(img_size, **kwargs):
    model = RSP(
        patch_size=16,
        emb_dim=1024,
        depth=24,
        num_heads=16,
        dec_emb_dim=512,
        dec_depth=8,
        dec_num_heads=16,
        mlp_ratio=4,
        img_size=img_size,
        **kwargs,
    )
    return model


rsp_vit_small_patch8 = rsp_vit_small_patch8_dec512d8b  # decoder: 512 dim, 8 blocks
rsp_vit_small_patch16 = rsp_vit_small_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
rsp_vit_base_patch16 = rsp_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
rsp_vit_large_patch16 = rsp_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
