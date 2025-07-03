import pickle
from datasets import load_dataset
import torch
import os
from torchvision import transforms
import math
import transformers
import time
import externals

vit_processor = externals.get_vit_processor()
clip_processor = externals.get_clip_processor()

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Set the indices
        item_index = math.floor(idx / 5)
        caption_index = idx % 5
        item = self.dataset[item_index]
        if len(item['caption']) != 5:
            raise ValueError(f"This code only works if each image has 5 captions, there are {len(item['caption'])} captions for index {item_index}")
        
        # Get the image
        image = item['image']
        processed_image = vit_processor(images=image, return_tensors="pt")

        # Get the caption
        caption = item['caption'][caption_index]
        pixel_values = processed_image['pixel_values'].squeeze(0)

        return pixel_values, caption
    

def get_data_loaders(batch_size):
    if os.path.exists("train_dataset.pkl"):
        with open("train_dataset.pkl", "rb") as f:
            train_dataset = pickle.load(f)
        with open("test_dataset.pkl", "rb") as f:
            test_dataset = pickle.load(f)
    else:
        dataset = load_dataset("nlphuji/flickr30k", split="test[0:100]")
        dataset = dataset.remove_columns(['sentids', 'img_id', 'filename'])

        train_dataset = dataset.filter(lambda x: x['split'] == 'train')
        test_dataset = dataset.filter(lambda x: x['split'] == 'test')

        with open("train_dataset.pkl", "wb") as f:
            pickle.dump(train_dataset, f) 
        with open("test_dataset.pkl", "wb") as f:
            pickle.dump(test_dataset, f) 

    # start = time.time()
    train_dataset = ImageDataset(train_dataset)
    test_dataset = ImageDataset(test_dataset)
    # print(f"Time to process image: {time.time() - start:.2f} seconds")

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, pin_memory=torch.cuda.is_available(), collate_fn=collate_batch
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, pin_memory=torch.cuda.is_available(), collate_fn=collate_batch
    )

    return train_loader, test_loader


def collate_batch(batch):
    images, captions = zip(*batch)
    # images: tuple of tensors
    images = torch.stack(images)
    # captions: tuple of strings
    caption_texts = [c for c in captions]
    processed_captions = clip_processor(list(caption_texts), return_tensors='pt', padding=True)
    return images, processed_captions