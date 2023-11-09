import torch
from resampler import Resampler
from transformers import CLIPVisionModel

BATCH_SIZE = 2
OUTPUT_DIM = 1280
NUM_QUERIES = 8
NUM_LATENTS_MEAN_POOLED = 4  # 0 for no mean pooling (previous behavior)
APPLY_POS_EMB = True  # False for no positional embeddings (previous behavior)
IMAGE_ENCODER_NAME_OR_PATH = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"


def main():
    image_encoder = CLIPVisionModel.from_pretrained(IMAGE_ENCODER_NAME_OR_PATH)
    embedding_dim = image_encoder.config.hidden_size
    print(f"image_encoder hidden size: ", embedding_dim)

    image_proj_model = Resampler(
        dim=1024,
        depth=2,
        dim_head=64,
        heads=16,
        num_queries=NUM_QUERIES,
        embedding_dim=embedding_dim,
        output_dim=OUTPUT_DIM,
        ff_mult=2,
        max_seq_len=257,
        apply_pos_emb=APPLY_POS_EMB,
        num_latents_mean_pooled=NUM_LATENTS_MEAN_POOLED,
    )

    dummy_images = torch.randn(BATCH_SIZE, 3, 224, 224)
    with torch.no_grad():
        image_embeds = image_encoder(dummy_images, output_hidden_states=True).hidden_states[-2]
    print("image_embds shape: ", image_embeds.shape)

    with torch.no_grad():
        ip_tokens = image_proj_model(image_embeds)
    print("ip_tokens shape:", ip_tokens.shape)
    assert ip_tokens.shape == (BATCH_SIZE, NUM_QUERIES + NUM_LATENTS_MEAN_POOLED, OUTPUT_DIM)


if __name__ == "__main__":
    main()
