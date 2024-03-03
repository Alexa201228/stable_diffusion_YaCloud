import argparse
import torch
import shutil
from diffusers import DiffusionPipeline
from diffusers import LMSDiscreteScheduler
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='test args')
    parser.add_argument('-i', '--input', required=True, type=argparse.FileType('r'))

    args = parser.parse_args()

    queries = args.input.readlines()

    base = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
    )
    base.to("cuda")

    base.scheduler = LMSDiscreteScheduler.from_config(base.scheduler.config)

    refiner = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-refiner-1.0",
        text_encoder_2=base.text_encoder_2,
        vae=base.vae,
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
    )
    refiner.to("cuda")

    n_steps = 100
    high_noise_frac = 0.8

    if not os.path.exists('images'):
        os.makedirs('images')

    i = 0
    for prompt in queries:
        print(f'готовимся генерировать картинку {i}')
        image = base(
            prompt=prompt,
            num_inference_steps=n_steps,
            denoising_end=high_noise_frac,
            output_type="latent",
        ).images

        print(f'Запускаем refine model')

        image = refiner(
            prompt=prompt,
            num_inference_steps=n_steps,
            denoising_start=high_noise_frac,
            image=image,
        ).images[0]

        print(f'картинка {i} сгенерирована')
        image.save(f"images/{i}.png")
        i += 1

    print("Создаем архив")
    shutil.make_archive('output_images', 'zip', 'images')
    print("Архив создан")