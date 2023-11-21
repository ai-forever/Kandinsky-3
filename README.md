# Kandinsky-3: Text-to-image diffusion model

![](assets/title.jpg)

[Post] | [Fusion Brain](https://fusionbrain.ai) | [Telegram-bot](https://t.me/kandinsky21_bot) |

## Description:

Kandinsky 3.0 is an open source text-to-image diffusion model

## Architecture details:

![](assets/kandinsky.jpg)

+ Text encoder Flan-UL2 - 8.6B
+ Latent Diffusion U-Net - 3B
+ MoVQ encoder/decoder - 267M


## Сheckpoints:

+ [Base](): Base text-to-image diffusion model 
+ [Inpainting](): Inpainting version of the model

## Installing

```
pip install -r requirements.txt
```

## How to use:

Check our jupyter notebooks with examples in `./examples` folder

## Examples of generations

<hr>

<table class="center">
<tr>
  <td><img src="assets/photo_8.jpg" raw=true></td>
  <td><img src="assets/photo_15.jpg"></td>
  <td><img src="assets/photo_16.jpg"></td>
  <td><img src="assets/photo_17.jpg"></td>
</tr>
<tr>
  <td width=25% align="center">"A beautiful landscape outdoors scene in the crochet knitting art style, drawing in style by Alfons Mucha"</td>
  <td width=25% align="center">"gorgeous phoenix, cosmic, darkness, epic, cinematic, moonlight, stars, high - definition, texture,Oscar-Claude Monet"</td>
  <td width=25% align="center">"a yellow house at the edge of the danish fjord, in the style of eiko ojala, ingrid baars, ad posters, mountainous vistas, george ault, realistic details, dark white and dark gray, 4k"</td>
  <td width=25% align="center">"dragon fruit head, upper body, realistic, illustration by Joshua Hoffine Norman Rockwell, scary, creepy, biohacking, futurism, Zaha Hadid style"</td>
</tr>
<tr>
  <td><img src="assets/photo_2.jpg" raw=true></td>
  <td><img src="assets/photo_19.jpg"></td>
  <td><img src="assets/photo_13.jpg"></td>
  <td><img src="assets/photo_14.jpg"></td>
</tr>
<tr>
  <td width=25% align="center">"Amazing playful nice cute strawberry character, dynamic poze, surreal fantazy garden background, gorgeous masterpice, award winning photo, soft natural lighting, 3d, Blender, Octane render, tilt - shift, deep field, colorful, I can't believe how beautiful this is, colorful, cute and sweet baby - loved photo"</td>
  <td width=25% align="center">"beautiful fairy-tale desert, in the sky a wave of sand merges with the milky way, stars, cosmism, digital art, 8k"</td>
  <td width=25% align="center">"Car, mustang, movie, person, poster, car cover, person, in the style of alessandro gottardo, gold and cyan, gerald harvey jones, reflections, highly detailed illustrations, industrial urban scenes""</td>
  <td width=25% align="center">"cloud in blue sky, a red lip, collage art, shuji terayama, dreamy objects, surreal, criterion collection, showa era, intricate details, mirror"</td>
</tr>

</table>

<hr>

## Authors

+ Vladimir Arkhipkin: [Github](https://github.com/oriBetelgeuse)
+ Anastasia Maltseva [Github](https://github.com/NastyaMittseva)
+ Andrei Filatov [Github](https://github.com/anvilarth)
+ Igor Pavlov: [Github](https://github.com/boomb0om)
+ Julia Agafonova 
+ Arseniy Shakhmatov: [Github](https://github.com/cene555), [Blog](https://t.me/gradientdip)
+ Andrey Kuznetsov: [Github](https://github.com/kuznetsoffandrey), [Blog](https://t.me/complete_ai)
+ Denis Dimitrov: [Github](https://github.com/denndimitrov), [Blog](https://t.me/dendi_math_ai)
