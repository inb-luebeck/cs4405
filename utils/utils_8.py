import os
os.environ["HUGGINGFACE_HUB_CACHE"] = "/data/huggingface"
os.environ["HF_DATASETS_CACHE"] = "/data/huggingface_datasets"
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from transformers.activations import ACT2FN
import torch
from torch import nn
import math
from collections import OrderedDict

silu = ACT2FN["silu"]

def get_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(
        "microsoft/Phi-3-mini-128k-instruct",
        revision="bb5bf1e4001277a606e11debca0ef80323e5f824"
    )
    return tokenizer

def get_model():
    model = AutoModelForCausalLM.from_pretrained(
            "microsoft/Phi-3-mini-128k-instruct",
            revision="bb5bf1e4001277a606e11debca0ef80323e5f824",
            device_map="cuda",
            torch_dtype="auto",
            trust_remote_code=True,
        )
    return model

def get_model_weights():
    def get_model_state_dict():
        model = get_model()
        return model.state_dict()
    
    def move_state_dict(orig_state_dict):
        # move weight tensors to cpu
        new_state_dict = [(name, weight.cpu()) for name, weight in orig_state_dict.items()]
        return OrderedDict(new_state_dict)

    def transform_state_dict(orig_state_dict):
        new_state_dict = []
        for name, weight in orig_state_dict.items():
            if name.startswith('model.'):
                name = name[len('model.'):]

            if name.endswith('qkv_proj.weight'):
                name_prefix = name[:-len('qkv_proj.weight')]
                q_projs = weight[:3072]
                k_projs = weight[3072:3072*2]
                v_projs = weight[3072*2:]

                num_heads = 32
                head_dim = 3072 // num_heads
                for i in range(num_heads):
                    q_proj = q_projs[head_dim*i:head_dim*(i+1)]
                    k_proj = k_projs[head_dim*i:head_dim*(i+1)]
                    v_proj = v_projs[head_dim*i:head_dim*(i+1)]
                    q_name = name_prefix + f'attention_heads.{i}.q_proj.weight'
                    k_name = name_prefix + f'attention_heads.{i}.k_proj.weight'
                    v_name = name_prefix + f'attention_heads.{i}.v_proj.weight'
                    new_state_dict.append((q_name, q_proj))
                    new_state_dict.append((k_name, k_proj))
                    new_state_dict.append((v_name, v_proj))

                # layers.0.self_attn.qkv_proj.weight torch.Size([9216, 3072])
                # layers.0.self_attn.attention_heads.0.q_proj.weight
                # layers.0.self_attn.attention_heads.1.q_proj.weight
                # ...
                # layers.0.self_attn.attention_heads.0.k_proj.weight
                # layers.0.self_attn.attention_heads.1.k_proj.weight
                # ...
                # layers.0.self_attn.attention_heads.0.v_proj.weight
                # ...
            else:
                new_state_dict.append((name, weight))

        return OrderedDict(new_state_dict)
    
    state_dict = get_model_state_dict()
    state_dict = move_state_dict(state_dict)
    state_dict = transform_state_dict(state_dict)
    
    return state_dict

# Copied from transformers.models.llama.modeling_llama.LlamaRMSNorm with Llama->Phi3
class Phi3RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Phi3RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)
    
class Phi3RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.register_buffer("inv_freq", None, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if self.inv_freq is None:
            self.inv_freq = 1.0 / (
                self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64, device=x.device).float() / self.dim)
            )
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 since bfloat16 loses precision on long contexts
        # See https://github.com/huggingface/transformers/pull/29285
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)
    
class Phi3SuScaledRotaryEmbedding(Phi3RotaryEmbedding):
    def __init__(self, dim, config, device=None):
        super().__init__(dim, config.max_position_embeddings, config.rope_theta, device)

        self.short_factor = config.rope_scaling["short_factor"]
        self.long_factor = config.rope_scaling["long_factor"]
        self.original_max_position_embeddings = config.original_max_position_embeddings

    @torch.no_grad()
    def forward(self, x, position_ids, seq_len=None):
        seq_len = torch.max(position_ids) + 1
        if seq_len > self.original_max_position_embeddings:
            ext_factors = torch.tensor(self.long_factor, dtype=torch.float32, device=x.device)
        else:
            ext_factors = torch.tensor(self.short_factor, dtype=torch.float32, device=x.device)

        inv_freq_shape = torch.arange(0, self.dim, 2, dtype=torch.int64, device=x.device).float() / self.dim
        self.inv_freq = 1.0 / (ext_factors * self.base**inv_freq_shape)

        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()

        # Force float32 since bfloat16 loses precision on long contexts
        # See https://github.com/huggingface/transformers/pull/29285
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)

            scale = self.max_position_embeddings / self.original_max_position_embeddings
            if scale <= 1.0:
                scaling_factor = 1.0
            else:
                scaling_factor = math.sqrt(1 + math.log(scale) / math.log(self.original_max_position_embeddings))

            cos = emb.cos() * scaling_factor
            sin = emb.sin() * scaling_factor
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)
    
# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)
    
# Copied from transformers.models.llama.modeling_llama.apply_rotary_pos_emb
def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.
    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def get_attention_mask(config, hidden_states):
    return _prepare_4d_causal_attention_mask(
            None,
            (hidden_states.size()[0], hidden_states.size()[1]),
            hidden_states,
            0,
            sliding_window=config.sliding_window,
        ).squeeze(1)

def get_config():
    config = AutoConfig.from_pretrained("microsoft/Phi-3-mini-128k-instruct", revision="bb5bf1e4001277a606e11debca0ef80323e5f824")
    return config

# prepare position embedding
head_dim = 96
rotary_emb = Phi3SuScaledRotaryEmbedding(head_dim, get_config())

def apply_pos_emb(hidden_states, queries, keys, values):
    position_ids = torch.arange(0, hidden_states.size()[1], dtype=torch.long, device=hidden_states.device)
    position_ids = position_ids.unsqueeze(0).view(-1, hidden_states.size()[1])
    cos, sin = rotary_emb(values, position_ids, seq_len=hidden_states.size()[1])
    queries, keys = apply_rotary_pos_emb(queries, keys, cos, sin, position_ids)
    queries = queries.squeeze_(0)
    keys = keys.squeeze_(0)
    
    return queries, keys

def format_model_output(logits, return_dict):
    if not return_dict:
        output = (logits,)
        return output

    return CausalLMOutputWithPast(
        loss=None,
        logits=logits,
        past_key_values=None,
        hidden_states=None,
        attentions=None,
    )
