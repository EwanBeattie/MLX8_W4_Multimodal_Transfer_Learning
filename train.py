import torch
from model import Transformer
from configs import hyperparameters, run_config, sweep_config
import torch.nn as nn
from types import SimpleNamespace
from data import get_data_loaders
import externals
from tqdm import tqdm
import wandb

def main():

    torch.manual_seed(42)
    
    if run_config['run_type'] == 'sweep':
        pass
        # sweep_id = wandb.sweep(sweep_config, entity=run_config['entity'], project=run_config['project'])
        # wandb.agent(
        #     sweep_id=sweep_id,
        #     function=train,
        #     project=run_config['project'],
        #     count=40,
        # )
    elif run_config['run_type'] == 'train':
        trained_weights = train(hyperparameters)
        torch.save(trained_weights, "model_weights.pth")


def train(config=None):
    # Obtain correct cofig dict, init wandb then create wandb.config object
    if config is None:
        config = sweep_config
    wandb.init(entity=run_config['entity'], project=run_config['project'], config=config)
    config = wandb.config

    # config = SimpleNamespace(**hyperparameters)

    device = get_device()
    
    train_loader, test_loader = get_data_loaders(batch_size=config.batch_size)
    
    # Initialise the model
    model = Transformer(
        embedding_dim=768,
        num_layers=config.num_layers
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    clip_processor = externals.get_clip_processor()
    loss_function = nn.CrossEntropyLoss(ignore_index=clip_processor.tokenizer.pad_token_id)

    model.train()
    for epoch in range(config.epochs):
        loop = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{config.epochs}")
        for batch_index, (image, caption) in loop:
            image = image.to(device)
            input_ids = caption['input_ids'].to(device)  # [batch, seq_len]
            # attention_mask = caption['attention_mask'].to(device)  # [batch, seq_len]

            output = model(image, caption)  # [batch, total_seq_len, vocab_size]

            # Only use the output positions that correspond to the text tokens
            image_tokens = 197  # For ViT-base: 196 patches + 1 CLS
            output_text = output[:, image_tokens:-1, :] 
            target_text = input_ids[:, 1:]             

            # Flatten for CrossEntropyLoss
            output_text = output_text.reshape(-1, output_text.size(-1))
            target_text = target_text.reshape(-1)

            loss = loss_function(output_text, target_text)
            loss.backward()
            optimizer.step()

            if batch_index % 100 == 0:
                print(f'Epoch: {epoch + 1}, Batch: {batch_index}, Loss: {loss.item():.2f}')
                wandb.log({'loss': loss.item()})
            loop.set_postfix(loss=loss.item())

    test(model, test_loader, device)

    wandb.finish()

    return model.state_dict()

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def test(model, test_loader, device):
    model.eval()
    clip_processor = externals.get_clip_processor()
    pad_token_id = clip_processor.tokenizer.pad_token_id
    total_loss = 0
    total_tokens = 0
    loss_function = nn.CrossEntropyLoss(ignore_index=pad_token_id)
    with torch.no_grad():
        for image, caption in tqdm(test_loader, desc="Testing", total=len(test_loader)):
            image = image.to(device)
            input_ids = caption['input_ids'].to(device)

            output = model(image, caption)  # [batch, total_seq_len, vocab_size]
            image_tokens = 197
            output_text = output[:, image_tokens:-1, :]  # [batch, text_seq_len-1, vocab_size]
            target_text = input_ids[:, 1:]               # [batch, text_seq_len-1]

            output_text = output_text.reshape(-1, output_text.size(-1))
            target_text = target_text.reshape(-1)

            loss = loss_function(output_text, target_text)
            total_loss += loss.item() * target_text.ne(pad_token_id).sum().item()
            total_tokens += target_text.ne(pad_token_id).sum().item()

    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
    print(f"Test Loss: {avg_loss:.4f}")
    wandb.log({'test_loss': avg_loss})

if __name__ == "__main__":
    main()
