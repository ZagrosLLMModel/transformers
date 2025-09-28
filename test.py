import sys
import torch

transformers_path = r"D:\Backup\Zagros-moe\Zagros-max\transformers"
sys.path.insert(0, transformers_path)

from src.transformers.models.zagros.configuration_zagros import ZagrosConfig
from src.transformers.models.zagros.modeling_zagros import ZagrosForCausalLM

# config_dict کامل با تمام params جدید (سازگار با تغییرات)
config_dict = {
    "architectures": ["ZagrosForCausalLM"],
    "attention_bias": False,
    "attention_dropout": 0.0,
    "decoder_sparse_step": 1,
    "eos_token_id": 151645,
    "head_dim": 128,
    "hidden_act": "silu",
    "hidden_size": 2048,
    "initializer_range": 0.02,
    "intermediate_size": 6144,
    "max_position_embeddings": 262144,
    "max_window_layers": 48,
    "mlp_only_layers": [i for i in range(0, 48, 4)],  # Hybrid
    "model_type": "zagros",
    "moe_intermediate_size": 384,
    "norm_topk_prob": True,
    "num_attention_heads": 32,
    "num_experts": 512,
    "num_experts_per_tok": 8,
    "num_hidden_layers": 4,  # کوچک برای تست روی RTX 4050
    "num_key_value_heads": 4,
    "output_router_logits": False,
    "pad_token_id": 151654,
    "rms_norm_eps": 1e-06,
    "rope_scaling": None,
    "rope_theta": 10000000,
    "router_aux_loss_coef": 0.0,
    "sliding_window": None,
    "tie_word_embeddings": False,
    "transformers_version": "4.57.0.dev0",
    "unsloth_fixed": True,
    "unsloth_version": "2025.9.7",
    "use_cache": True,
    "use_sliding_window": False,
    "vocab_size": 151936,
    "use_dual_routing": True,  # NEW
    "diversity_factor": 0.5,  # NEW
    "super_expert_threshold": 0.005  # NEW
}

config = ZagrosConfig(**config_dict)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

model = ZagrosForCausalLM(config).to(device)
model.eval()

batch_size, seq_len = 1, 10
input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len)).to(device)
attention_mask = torch.ones_like(input_ids).to(device)

with torch.no_grad():
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_router_logits=True)

print("Logits shape:", outputs.logits.shape)
print("Router logits len:", len(outputs.router_logits) if outputs.router_logits else 0)
print("Diversity loss:", outputs.diversity_loss.item() if hasattr(outputs, 'diversity_loss') else "N/A")
print("Test passed! GPU memory:", torch.cuda.memory_allocated() / 1024**3 if device == 'cuda' else "CPU")