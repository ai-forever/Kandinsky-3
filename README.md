# Kandinsky-3

+ [Habr post](https://github.com/ai-forever/Kandinsky-3)
+ [Demo fusionbrain.ai](https://fusionbrain.ai/diffusion)
+ [Telegram-bot](https://t.me/kandinsky21_bot)

**Description:**

Kandinsky 2.2 brings substantial improvements upon its predecessor, Kandinsky 2.1, by introducing a new, more powerful image encoder - CLIP-ViT-G and the ControlNet support.

The switch to CLIP-ViT-G as the image encoder significantly increases the model's capability to generate more aesthetic pictures and better understand text, thus enhancing the model's overall performance.

The addition of the ControlNet mechanism allows the model to effectively control the process of generating images. This leads to more accurate and visually appealing outputs and opens new possibilities for text-guided image manipulation.

**Architecture details:**

+ Text encoder (XLM-Roberta-Large-Vit-L-14) - 560M
+ Diffusion Image Prior — 1B
+ CLIP image encoder (ViT-bigG-14-laion2B-39B-b160k) - 1.8B
+ Latent Diffusion U-Net - 1.22B
+ MoVQ encoder/decoder - 67M


**Сheckpoints:**

+ [Prior](https://huggingface.co/kandinsky-community/kandinsky-2-2-prior): A prior diffusion model mapping text embeddings to image embeddings
+ [Text-to-Image / Image-to-Image](https://huggingface.co/kandinsky-community/kandinsky-2-2-decoder): A decoding diffusion model mapping image embeddings to images
+ [Inpainting](https://huggingface.co/kandinsky-community/kandinsky-2-2-decoder-inpaint): A decoding diffusion model mapping image embeddings and masked images to images
+ [ControlNet-depth](https://huggingface.co/kandinsky-community/kandinsky-2-2-controlnet-depth): A decoding diffusion model mapping image embedding and additional depth condition to images

## How to use:

Check our jupyter notebooks with examples in `./notebooks` folder
### 1. text2image

```python
from kandinsky2 import get_kandinsky2
model = get_kandinsky2('cuda', task_type='text2img', model_version='2.2')
images = model.generate_text2img(
    "red cat, 4k photo", 
    decoder_steps=50,
    batch_size=1, 
    h=1024,
    w=768,
)
```

```python
from kandinsky2 import get_kandinsky2
from PIL import Image
import numpy as np

model = get_kandinsky2('cuda', task_type='inpainting', model_version='2.1', use_flash_attention=False)
init_image = Image.open('img.jpg')
mask = np.ones((768, 768), dtype=np.float32)
mask[:,:550] =  0
images = model.generate_inpainting(
    'man 4k photo', 
    init_image, 
    mask, 
    num_steps=150,
    batch_size=1, 
    guidance_scale=5,
    h=768, w=768,
    sampler='p_sampler', 
    prior_cf_scale=4,
    prior_steps="5"
)
```

# Authors

+ Arseniy Shakhmatov: [Github](https://github.com/cene555), [Blog](https://t.me/gradientdip)
+ Vladimir Arkhipkin: [Github](https://github.com/oriBetelgeuse)
+ Igor Pavlov: [Github](https://github.com/boomb0om)
+ Andrey Kuznetsov: [Github](https://github.com/kuznetsoffandrey)
+ Denis Dimitrov: [Github](https://github.com/denndimitrov)
