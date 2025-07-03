import torch
import torch.nn as nn
import externals

class Transformer(nn.Module):
    def __init__(self, embedding_dim, num_layers=6):
        super().__init__()
        self.decoding_layers = nn.ModuleList([
            Decoder(embedding_dim) for _ in range(num_layers)
        ])
        vit_model = externals.get_vit_model()
        clip_model = externals.get_clip_model()

        self.vit_embeddings = vit_model.embeddings
        self.clip_embeddings = clip_model.text_model.embeddings

        self.clip_to_vit = nn.Linear(512, 768)  

        vocab_size = clip_model.text_model.embeddings.token_embedding.weight.shape[0]
        self.scores = nn.Linear(embedding_dim, vocab_size)

    def forward(self, image, caption):
        image_embeddings = self.vit_embeddings(image)
        caption_embeddings = self.clip_embeddings(caption['input_ids'])
        caption_embeddings = self.clip_to_vit(caption_embeddings)
        caption_mask = caption['attention_mask']

        x = input_embeddings = torch.cat((image_embeddings, caption_embeddings), dim=1)

        for layer in self.decoding_layers:
            x = layer(x)
        x = self.scores(x)
        return x

class Decoder(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.attention = SelfAttention(embedding_dim)
        self.MLP = MLP(embedding_dim)

    def forward(self, x):
        x = self.attention(x)
        x = self.MLP(x)
        return x
    
class SelfAttention(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()

        self.query = nn.Linear(embedding_dim, embedding_dim)
        self.key = nn.Linear(embedding_dim, embedding_dim)
        self.value = nn.Linear(embedding_dim, embedding_dim)
        self.softmax = nn.Softmax(dim=-1)
        self.output = nn.Linear(embedding_dim, embedding_dim)
        self.dropout = nn.Dropout(0.1)
        self.layer_norm = nn.LayerNorm(embedding_dim)

        self.register_buffer('_cached_mask', torch.empty(1), persistent=False)

    def forward(self, x):
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / (key.size(-1) ** 0.5)

        seq_len = x.shape[-2]
        if self._cached_mask.shape[-1] < seq_len:
            mask = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.bool, device=x.device), diagonal=1)
            self._cached_mask = mask
        
        # Since self._cached_mask is on the same device as the model, we can use it directly
        attention_scores = attention_scores.masked_fill(self._cached_mask[:seq_len, :seq_len], float("-inf"))

        attention_weights = self.softmax(attention_scores)
        hidden = torch.matmul(attention_weights, value)
        output = self.output(hidden)
        x = output + x
        x = self.layer_norm(x)
        return x
    
class MLP(nn.Module):
    def __init__(self, embedding_dim, hidden_dim=2048):
        super().__init__()
        self.linear1 = nn.Linear(embedding_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim, embedding_dim)
        self.dropout = nn.Dropout(0.1)
        self.layer_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        x_input = x
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.dropout(x)
        x = x + x_input
        x = self.layer_norm(x)
        return x