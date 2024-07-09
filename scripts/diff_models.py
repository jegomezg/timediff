import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from linear_attention_transformer import LinearAttentionTransformer


def get_torch_trans(heads=8, layers=1, channels=64):
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=channels, nhead=heads, dim_feedforward=64, activation="gelu"
    )
    return nn.TransformerEncoder(encoder_layer, num_layers=layers)

def get_linear_trans(heads=8,layers=1,channels=64,localheads=0,localwindow=0):

  return LinearAttentionTransformer(
        dim = channels,
        depth = layers,
        heads = heads,
        max_seq_len = 256,
        n_local_attn_heads = 0, 
        local_attn_window_size = 0,
    )

def Conv1d_with_init(in_channels, out_channels, kernel_size):
    layer = nn.Conv1d(in_channels, out_channels, kernel_size)
    nn.init.kaiming_normal_(layer.weight)
    return layer


class DiffusionEmbedding(nn.Module):
    def __init__(self, num_steps, embedding_dim=128, projection_dim=None):
        super().__init__()
        if projection_dim is None:
            projection_dim = embedding_dim
        self.register_buffer(
            "embedding",
            self._build_embedding(num_steps, embedding_dim / 2),
            persistent=False,
        )
        self.projection1 = nn.Linear(embedding_dim, projection_dim)
        self.projection2 = nn.Linear(projection_dim, projection_dim)

    def forward(self, diffusion_step):
        x = self.embedding[diffusion_step]
        x = self.projection1(x)
        x = F.silu(x)
        x = self.projection2(x)
        x = F.silu(x)
        return x

    def _build_embedding(self, num_steps, dim=64):
        steps = torch.arange(num_steps).unsqueeze(1)  # (T,1)
        frequencies = 10.0 ** (torch.arange(dim) / (dim - 1) * 4.0).unsqueeze(0)  # (1,dim)
        table = steps * frequencies  # (T,dim)
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)  # (T,dim*2)
        return table


class MetadataEmbedding(nn.Module):
    def __init__(self, num_categories, num_continuous, d_cat, d_cont, d_meta):
        super(MetadataEmbedding, self).__init__()
        self.num_categories = num_categories
        self.num_continuous = num_continuous
        self.d_cat = d_cat
        self.d_cont = d_cont
        self.d_meta = d_meta
        
        if num_categories > 0:
            self.cat_embedding = nn.Embedding(num_categories, num_categories)
            self.cat_tokenizer = nn.Sequential(
                nn.Linear(num_categories, d_cat),
                nn.ReLU(),
                nn.Linear(d_cat, d_cat)
            )
        
        if num_continuous > 0:
            self.cont_tokenizer = nn.Sequential(
                nn.Linear(num_continuous, d_cont),
                nn.ReLU(),
                nn.Linear(d_cont, d_cont)
            )
        
        self.self_attention = nn.MultiheadAttention(d_cat + d_cont, num_heads=8)
        self.fc = nn.Linear(d_cat + d_cont, d_meta)

    def forward(self, c_cat=None, c_cont=None):
        # c_cat: (L, N) or None
        # c_cont: (L, N, num_continuous) or None
        
        embeddings = []
        device = None
        if c_cat is not None and self.num_categories > 0:
            device = c_cat.device
            # Categorical embedding
            c_cat_one_hot = self.cat_embedding(c_cat)  # (N, L, num_categories)
            z_cat = self.cat_tokenizer(c_cat_one_hot)  # (N, L, d_cat)
            embeddings.append(z_cat)
        
        if c_cont is not None and self.num_continuous > 0:
            device = c_cont.device if device is None else device
            # Continuous embedding
            z_cont = self.cont_tokenizer(c_cont)  # (N, L, d_cont)
            embeddings.append(z_cont)
        
        if len(embeddings) == 0:
            raise ValueError("At least one of categorical or continuous metadata must be provided.")
        
        # Concatenate available embeddings
        z_concat = torch.cat(embeddings, dim=-1)  # (N, L, d_cat + d_cont) or (N, L, d_cat) or (N, L, d_cont)
        
        # Adjust input size for self-attention layer and fully connected layer
        embedding_size = z_concat.shape[-1]
        self.self_attention = nn.MultiheadAttention(embedding_size, num_heads=8).to(device)
        self.fc = nn.Linear(embedding_size, self.d_meta).to(device)
        
        # Self-attention layer
        z, _ = self.self_attention(z_concat, z_concat, z_concat)
        
        
        # Fully connected layer
        z = self.fc(z)  # (N, L, d_meta)
        
        return z


class diff_CSDI(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.channels = config["d_meta"] #C

        self.diffusion_embedding = DiffusionEmbedding(
            num_steps=config["num_steps"],
            embedding_dim=config["diffusion_embedding_dim"],
        )
        
        self.condition_embedding = MetadataEmbedding(
            num_categories=config["num_categories_meta"],
            num_continuous=config["num_continious_meta"],
            d_cat=config["d_cat"],
            d_cont=config["d_cont"],
            d_meta=config["d_meta"] #C 
        )

        self.input_projection = Conv1d_with_init(1, self.channels, 1) 
        self.output_projection1 = Conv1d_with_init(self.channels, self.channels, 1)
        self.output_projection2 = Conv1d_with_init(self.channels, 1, 1)
        nn.init.zeros_(self.output_projection2.weight)

        self.residual_layers = nn.ModuleList(
            [
                ResidualBlock(
                    cond_dim=config["d_cat"] + config["d_cont"],
                    channels=self.channels,
                    diffusion_embedding_dim=config["diffusion_embedding_dim"],
                    nheads=config["nheads"],
                    is_linear=config["is_linear"],
                )
                for _ in range(config["layers"])
            ]
        )

    def forward(self, x, cond_info, diffusion_step):
        B, L, K = x.shape #(N,1,F,L)
        x = x.reshape(B, 1, K * L) #(N,1,F*L)
        x = self.input_projection(x) 
        x = F.relu(x)
        x = x.reshape(B, self.channels, K, L) #(N,C,F,L)

        diffusion_emb = self.diffusion_embedding(diffusion_step)  
        
        condition_embeding = self.condition_embedding(cond_info['cat'],cond_info['cont'])
        condition_embeding = condition_embeding.permute(1, 2, 0).unsqueeze(2).repeat(1, 1, K, 1) # (N, d_meta, F, L)

        skip = []
        for layer in self.residual_layers:
            x, skip_connection = layer(x, condition_embeding, diffusion_emb)
            skip.append(skip_connection)

        x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers))
        x = x.reshape(B, self.channels, K * L)
        x = self.output_projection1(x)  # (B,channel,K*L)
        x = F.relu(x)
        x = self.output_projection2(x)  # (B,1,K*L)
        x = x.reshape(B, K, L)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, cond_dim, channels, diffusion_embedding_dim, nheads, is_linear=False):
        super().__init__()
        self.diffusion_projection = nn.Linear(diffusion_embedding_dim, channels)
        self.cond_projection = Conv1d_with_init(cond_dim, 2 * channels, 1)
        self.mid_projection = Conv1d_with_init(channels, 2 * channels, 1)
        self.output_projection = Conv1d_with_init(channels, 2 * channels, 1)

        self.is_linear = is_linear
        if is_linear:
            self.time_layer = get_linear_trans(heads=nheads,layers=1,channels=channels)
            self.feature_layer = get_linear_trans(heads=nheads,layers=1,channels=channels)
        else:
            self.time_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)
            self.feature_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)


    def forward_time(self, y, base_shape):
        B, channel, K, L = base_shape
        if L == 1:
            return y
        y = y.reshape(B, channel, K, L).permute(0, 2, 1, 3).reshape(B * K, channel, L)

        if self.is_linear:
            y = self.time_layer(y.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            y = self.time_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
        y = y.reshape(B, K, channel, L).permute(0, 2, 1, 3).reshape(B, channel, K * L)
        return y


    def forward_feature(self, y, base_shape):
        B, channel, K, L = base_shape
        if K == 1:
            return y
        y = y.reshape(B, channel, K, L).permute(0, 3, 1, 2).reshape(B * L, channel, K)
        if self.is_linear:
            y = self.feature_layer(y.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            y = self.feature_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
        y = y.reshape(B, L, channel, K).permute(0, 2, 3, 1).reshape(B, channel, K * L)
        return y

    def forward(self, x, cond_info, diffusion_emb):
        B, channel, K, L = x.shape
        base_shape = x.shape
        x = x.reshape(B, channel, K * L)

        diffusion_emb = self.diffusion_projection(diffusion_emb).unsqueeze(-1)  # (B,channel,1)
        y = x + diffusion_emb

        y = self.forward_time(y, base_shape)
        y = self.forward_feature(y, base_shape)  # (B,channel,K*L)
        y = self.mid_projection(y)  # (B,2*channel,K*L)

        _, cond_dim, _, _ = cond_info.shape
        cond_info = cond_info.reshape(B, cond_dim, K * L)
        cond_info = self.cond_projection(cond_info)  # (B,2*channel,K*L)
        y = y + cond_info

        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)  # (B,channel,K*L)
        y = self.output_projection(y)

        residual, skip = torch.chunk(y, 2, dim=1)
        x = x.reshape(base_shape)
        residual = residual.reshape(base_shape)
        skip = skip.reshape(base_shape)
        return (x + residual) / math.sqrt(2.0), skip