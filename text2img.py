# %%
import base64
import random
import sys
from io import BytesIO
import random
from functools import partial

import jax
import numpy as np
import jax.numpy as jnp
from PIL import Image

from dalle_mini import DalleBart, DalleBartProcessor
from vqgan_jax.modeling_flax_vqgan import VQModel

# from flask import Flask, request, jsonify
# from flask_cors import CORS, cross_origin

from flax.jax_utils import replicate
from flax.training.common_utils import shard_prng_key

import wandb

# dalle-mini
DALLE_MODEL = "dalle-mini/dalle-mini/wzoooa1c:latest"  # can be wandb artifact or 🤗 Hub or local folder or google bucket
DALLE_COMMIT_ID = None

# VQGAN model
VQGAN_REPO = "dalle-mini/vqgan_imagenet_f16_16384"
VQGAN_COMMIT_ID = "e93a26e7707683d349bf5d5c41c5b0ef69b677a9"


# We can customize top_k/top_p used for generating samples
gen_top_k = None
gen_top_p = 0.9
temperature = None
cond_scale = 3.0

wandb.init(anonymous="must")


# Load models & tokenizer
model = DalleBart.from_pretrained(DALLE_MODEL, revision=DALLE_COMMIT_ID)
vqgan = VQModel.from_pretrained(VQGAN_REPO, revision=VQGAN_COMMIT_ID)


model._params = replicate(model.params)
vqgan._params = replicate(vqgan.params)


processor = DalleBartProcessor.from_pretrained(DALLE_MODEL, revision=DALLE_COMMIT_ID)


# model inference
@partial(jax.pmap, axis_name="batch", static_broadcasted_argnums=(3, 4, 5, 6))
def p_generate(tokenized_prompt, key, params, top_k, top_p, temperature, condition_scale):
    return model.generate(
        **tokenized_prompt,
        prng_key=key,
        params=params,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
        condition_scale=condition_scale,
    )


# decode images
@partial(jax.pmap, axis_name="batch")
def p_decode(indices, params):
    return vqgan.decode_code(indices, params=params)

  
def tokenize_prompt(prompt: str):
  tokenized_prompt = processor([prompt])
  return replicate(tokenized_prompt)

def generate_images(prompt:str, num_predictions: int):
  tokenized_prompt = tokenize_prompt(prompt)
  
  # create a random key
  seed = random.randint(0, 2**32 - 1)
  key = jax.random.PRNGKey(seed)

  # generate images
  images = []
  for i in range(num_predictions // jax.device_count()):
      # get a new key
      key, subkey = jax.random.split(key)
      
      # generate images
      encoded_images = p_generate(tokenized_prompt, shard_prng_key(subkey),
          model.params,gen_top_k, gen_top_p, temperature, cond_scale,
      )
      
      # remove BOS
      encoded_images = encoded_images.sequences[..., 1:]

      # decode images
      decoded_images = p_decode(encoded_images, vqgan.params)
      decoded_images = decoded_images.clip(0.0, 1.0).reshape((-1, 256, 256, 3))
      for img in decoded_images:
          images.append(Image.fromarray(np.asarray(img * 255, dtype=np.uint8)))
        
  return images

imgs = generate_images('jan playing around', 8)

for i, img in enumerate(imgs):
    img.save(f'{i}.png')

# %%
