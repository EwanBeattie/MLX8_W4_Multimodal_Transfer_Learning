import transformers

def get_vit_processor():
    return transformers.ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')

def get_clip_processor():
    return transformers.CLIPProcessor.from_pretrained('openai/clip-vit-base-patch16', use_fast=False)

def get_vit_model():
    return transformers.ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')

def get_clip_model():
    return transformers.CLIPModel.from_pretrained('openai/clip-vit-base-patch16', use_safetensors=True)