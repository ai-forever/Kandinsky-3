from setuptools import setup

setup(
    name="kandinsky3",
    packages=[
        "kandinsky3",
        "kandinsky3/model"
    ],
    install_requires=[
                    "timm",
                    "torch==1.10.1+cu111",
                    "torchvision==0.11.2+cu111",
                    "torchaudio==0.10.1",
                    "pytorch_lightning==1.7.5",
                    "transformers",
                    "accelerate",
                    "diffusers",
                    "setuptools==59.5.0",
                    "omegaconf",
                    "datasets",
                    "einops",
                    "webdataset",
                    "fsspec",
                    "s3fs",
                    "hydra-core",
                    "scikit-image",
                    "matplotlib",
                    "wandb",
                    "albumentations",
                    "bezier",
                    "scipy",
                    "Pillow",
                    "tqdm",
                    "huggingface_hub"
  
    ],
    author="",
)