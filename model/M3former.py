"""Models for M3former.
    

The code is built upon:
    https://github.com/CIA-Oceanix/TrAISformer
"""

import math
import logging
logger = logging.getLogger(__name__)
import torch
import torch.nn as nn
from torch.nn import functional as F
from types import SimpleNamespace
from transformers import AutoConfig, AutoModel, AutoTokenizer

class LLMStatsEncoder(nn.Module):
    def __init__(self, configs):
        super().__init__()
        if configs.llm == 'llama':

            llama_config = AutoConfig.from_pretrained('/ai/share/workspace/wwtan/wzjin/models/models--meta-llama--Meta-Llama-3-8B/snapshots/8cde5ca8380496c9a6cc7ef3a8b46a0372a1d920')
            llama_config.num_hidden_layers = configs.llm_layers # 一共32层
            llama_config.output_attentions = True
            llama_config.output_hidden_states = True


            base_path = "/ai/share/workspace/wwtan/wzjin/models/models--meta-llama--Meta-Llama-3-8B/snapshots/8cde5ca8380496c9a6cc7ef3a8b46a0372a1d920"

            self.tokenizer = AutoTokenizer.from_pretrained(
                base_path,
                trust_remote_code=True,
                local_files_only=True
            )

            self.llm_model = AutoModel.from_pretrained(
                base_path,
                trust_remote_code=True,
                local_files_only=True,
                config=llama_config,
            )
        elif configs.llm == 'llama3.2-1b':
            base_path = "/ai/share/workspace/wwtan/wzjin/models/models--meta-llama--Llama-3.2-1B/snapshots/4e20de362430cd3b72f300e6b0f18e50e7166e08"
            self.tokenizer = AutoTokenizer.from_pretrained(
                base_path,
                trust_remote_code=True,
                local_files_only=True
            )
            llama_config = AutoConfig.from_pretrained(base_path)
            llama_config.num_hidden_layers = configs.llm_layers # 一共16层
            llama_config.output_attentions = True
            llama_config.output_hidden_states = True
            self.llm_model = AutoModel.from_pretrained(
                base_path,
                trust_remote_code=True,
                local_files_only=True,
                config=llama_config,
            )
        elif configs.llm == 'qwen3':
            qwen_config = AutoConfig.from_pretrained('/ai/share/workspace/wwtan/wzjin/models/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218')
            qwen_config.num_hidden_layers = configs.llm_layers # 一共36层
            qwen_config.output_attentions = True
            qwen_config.output_hidden_states = True
            base_path = "/ai/share/workspace/wwtan/wzjin/models/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"
            self.tokenizer = AutoTokenizer.from_pretrained(
                base_path,
                trust_remote_code=True,
                local_files_only=True
            )

            self.llm_model = AutoModel.from_pretrained(
                base_path, 
                trust_remote_code=True, 
                local_files_only=True,
                config=qwen_config,
            )
        elif configs.llm == 'Qwen3.5-9B':
            base_path = "/ai/share/workspace/wwtan/wzjin/models/models--Qwen--Qwen3.5-9B-Base/snapshots/2d021f1887f1fe402bf2c53ed69d7f0fc4709ec9"
            qwen_config = AutoConfig.from_pretrained(base_path)
            qwen_config.num_hidden_layers = configs.llm_layers # 一共32层
            qwen_config.output_attentions = True
            qwen_config.output_hidden_states = True
            self.tokenizer = AutoTokenizer.from_pretrained(
                base_path,
                trust_remote_code=True,
                local_files_only=True
            )
            self.llm_model = AutoModel.from_pretrained(
                base_path, 
                trust_remote_code=True, 
                local_files_only=True,
                config=qwen_config,
            )
        else:
            raise ValueError(f"Unsupported LLM type: {configs.llm}")
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        for param in self.llm_model.parameters():
            param.requires_grad = False
            
        # 适配层，将预训练模型维度映射到目标维度
        # self.projection = nn.Linear(4096, configs.full_size if configs.add_logits else configs.n_embd)  
        h_f = self.llm_model.config.hidden_size
        self.projection = nn.Sequential(
            nn.Linear(h_f, 4*h_f),  # 扩展层
            nn.GELU(),                  # 激活函数
            nn.Dropout( 0.1),  # dropout
            nn.Linear(4 * h_f, configs.full_size if configs.text_mode==5 else configs.n_embd),  # 投影层
            nn.Dropout(0.1),  # 输出dropout
        )
        # if configs.llm == 'qwen3':
        self.projection = self.projection.to(self.llm_model.dtype)
        self.get_last = configs.stats_last
        self.description = configs.content
        self.text_mode = configs.text_mode
        
    def forward(self, stats):
        prompt = self.generate_text_prompt(stats)  # 生成文本提示（一个batch的列表）
        prompt_ids = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).input_ids.to(self.llm_model.device)
        # prompt_ids shape: (B, max_prompt_length)  # padding后的token ids
        with torch.no_grad():
            # 使用 LLaMA 模型提取文本特征
            if self.get_last or self.text_mode == 6:
                text_embeddings = self.llm_model(input_ids=prompt_ids).last_hidden_state[:, -1, :]
                # text_embeddings shape: (B, llm_dim)  # 取每个样本的最后一个token的embedding
            else:
                text_embeddings = self.llm_model(input_ids=prompt_ids).last_hidden_state
                # text_embeddings shape: (B, L，llm_dim)  # 取每个样本的所有token的embedding
        return self.projection(text_embeddings)
       
    def generate_text_prompt(self,stats):
        return [(f"{s['vessel desc']}. "
                f"traj from {s['start_time']} to {s['end_time']};interval {s['interval']}.") 
                for s in stats]

class ParallelSequenceMoEHead(nn.Module):
    def __init__(self, n_embd, text_dim, full_size, num_experts=8, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.full_size = full_size

        # 1. 路由网络 (Router)
        self.router = nn.Linear(text_dim, num_experts)
        nn.init.zeros_(self.router.weight)
        nn.init.zeros_(self.router.bias)

        # 2. 并行专家网络 (Parallel Experts)
        # 使用 Parameter 存储所有专家的权重，避免循环
        # 假设每个专家是 2 层 MLP: (n_embd -> hidden -> full_size)
        hidden_dim = n_embd 
        self.w1 = nn.Parameter(torch.randn(num_experts, n_embd, hidden_dim))
        self.w2 = nn.Parameter(torch.randn(num_experts, hidden_dim, full_size))
        self.b1 = nn.Parameter(torch.zeros(num_experts, 1, hidden_dim))
        self.b2 = nn.Parameter(torch.zeros(num_experts, 1, full_size))


    def forward(self, x, dec_embeddings):
        """
        x: (bs, seqlen, n_embd)
        dec_embeddings: (bs, text_dim)
        """
        bs, seqlen, n_embd = x.shape
        
        # --- Step 1: 路由计算 ---
        logits = self.router(dec_embeddings) # (bs, num_experts)
        gate_probs = F.softmax(logits, dim=-1)
        
        # 取 Top-K
        topk_weights, topk_indices = torch.topk(gate_probs, self.top_k, dim=-1)
        # 归一化 Top-K 权重
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True) # (bs, top_k)

        # --- Step 2: 并行计算 (The Mask/Parallel Trick) ---
        # 我们构建一个全量输出，然后根据 Top-K 索引进行选择加权
        
        # 1. 先计算所有专家对所有输入的结果 (Parallel MLP)
        # x: (bs, seqlen, n_embd) -> 扩展为 (num_experts, bs, seqlen, n_embd)
        x_expanded = x.unsqueeze(0).expand(self.num_experts, -1, -1, -1)
        
        # 第一层并行：(num_experts, bs*seqlen, n_embd) @ (num_experts, n_embd, hidden)
        # 使用 torch.matmul 的广播特性处理第一维(Expert维)
        hidden = torch.matmul(x_expanded.reshape(self.num_experts, -1, n_embd), self.w1) 
        hidden = F.relu(hidden + self.b1.reshape(self.num_experts, 1, -1))
        
        # 第二层并行：
        out_all_experts = torch.matmul(hidden, self.w2) 
        out_all_experts = out_all_experts + self.b2.reshape(self.num_experts, 1, -1)
        # 还原维度: (num_experts, bs, seqlen, full_size)
        out_all_experts = out_all_experts.view(self.num_experts, bs, seqlen, -1)

        # --- Step 3: 权重整合 (Gather & Sum) ---
        # 这一步将 Top-K 的权重分配到对应的专家输出上
        final_output = torch.zeros(bs, seqlen, self.full_size, device=x.device)
        
        # 记录用于可视化的 mask
        combined_mask = torch.zeros(bs, self.num_experts, device=x.device)

        for k in range(self.top_k):
            idx = topk_indices[:, k] # (bs,)
            w = topk_weights[:, k].view(bs, 1, 1) # (bs, 1, 1)
            
            # 这里的巧妙之处：从全量专家输出中，提取每个 Batch 指定的专家
            # out_all_experts: (num_experts, bs, seqlen, full_size)
            # 使用 gather 提取指定的专家输出
            # 先调整维度为 (bs, num_experts, seqlen * full_size)
            flat_experts = out_all_experts.permute(1, 0, 2, 3).reshape(bs, self.num_experts, -1)
            
            # 提取第 k 个 top 专家索引对应的输出
            current_expert_out = torch.gather(
                flat_experts, 1, idx.view(bs, 1, 1).expand(-1, -1, flat_experts.size(-1))
            ).squeeze(1) # (bs, seqlen * full_size)
            
            # 累加：权重 * 输出
            final_output += w * current_expert_out.view(bs, seqlen, -1)
            
            # 记录 mask
            combined_mask.scatter_(1, idx.unsqueeze(1), 1.0)

        # --- Step 4: Aux Loss ---
        # 使用同样的负载均衡逻辑
        avg_prob = gate_probs.mean(0)
        exp_freq = combined_mask.mean(0)
        aux_loss = self.num_experts * torch.sum(avg_prob * exp_freq)

        return final_output, aux_loss, {"mask": combined_mask, "logits": logits}


class TextPrior(nn.Module):
    def __init__(self, config):
        # D, lat_size, lon_size, sog_size, cog_size, hidden=256
        super().__init__()
        D = config.n_embd
        hidden = config.text_prior_hidden
        self.lat_head = nn.Sequential(nn.Linear(D, hidden), nn.ReLU(), nn.Linear(hidden, config.lat_size))
        self.lon_head = nn.Sequential(nn.Linear(D, hidden), nn.ReLU(), nn.Linear(hidden, config.lon_size))
        self.sog_head = nn.Sequential(nn.Linear(D, hidden), nn.ReLU(), nn.Linear(hidden, config.sog_size))
        self.cog_head = nn.Sequential(nn.Linear(D, hidden), nn.ReLU(), nn.Linear(hidden, config.cog_size))

    def forward(self, dec_emb, temp=1.0):
        # dec_emb: (B, D)
        p_lat = F.softmax(self.lat_head(dec_emb) / temp, dim=-1)  # (B, lat_size)
        p_lon = F.softmax(self.lon_head(dec_emb) / temp, dim=-1)  # (B, lon_size)
        p_sog = F.softmax(self.sog_head(dec_emb) / temp, dim=-1)  # (B, sog_size)
        p_cog = F.softmax(self.cog_head(dec_emb) / temp, dim=-1)  # (B, cog_size)
        return p_lat, p_lon, p_sog, p_cog

def build_traisformer_config():
    config = SimpleNamespace()

    config.useStatsEncode = True
    # =========================================================
    # 1. 物理空间 & 离散化（必须与你 Dataset 完全一致）
    # =========================================================
    config.lat_min = 53.5
    config.lat_max = 59.0
    config.lon_min = 7.5
    config.lon_max = 14.0
    config.sog_range = 30.0   # knot

    # 离散分辨率（⚠️ 核心超参数）
    config.lat_size = 1100     
    config.lon_size = 1300
    config.sog_size = 300      # 0–30 knot
    config.cog_size = 360      # 360 / 5 deg

    config.full_size = (
        config.lat_size
        + config.lon_size
        + config.sog_size
        + config.cog_size
    )

    # =========================================================
    # 2. Embedding 维度（建模容量）
    # =========================================================
    config.n_lat_embd = 256
    config.n_lon_embd = 256
    config.n_sog_embd = 128
    config.n_cog_embd = 128



    config.n_embd = (
        config.n_lat_embd
        + config.n_lon_embd
        + config.n_sog_embd
        + config.n_cog_embd
    )
    # config.stats_embd = 768
    config.text_mode = 1
    # text_mode 1: self-att
    # text_mode 2: cross-att
    # text_mode 3: contact
    # text_mode 4: add
    # text_mode 5: add logits
    # text_mode 6: text att
    config.text_att_alpha = 0.5
    config.text_prior_hidden = 2048
    
    config.stats_last = False
    config.add_logits = False
    config.turn_loss = True
    config.turn_weight = 0.5
    config.turn_thresh = 8
    config.text_att = False

    config.moe_strategy = "mixst"      # ["none", "token", "seq", "mix"]
    config.num_experts = 4
    config.top_k = 1

    config.moe_head = False
    config.head_num_experts = 8
    config.head_ex_top_k = 2
    config.moe_balance_w = 0.0001


    # config.llm = "qwen3"  # ["llama", "qwen3"]
    config.llm = "llama"  # ["llama", "qwen3"]
    config.llm_layers = 6
    # =========================================================
    # 3. Transformer 结构
    # =========================================================
    config.n_layer = 8
    config.n_head = 8
    config.max_seqlen = 200

    config.embd_pdrop = 0.1
    config.resid_pdrop = 0.1
    config.attn_pdrop = 0.1

    # =========================================================
    # 4. 离散方式 & 预测模式
    # =========================================================
    config.partition_mode = "uniform"  # ⚠️ 强烈建议
    config.mode = "pos"                # 分类预测

    # =========================================================
    # 5. Blur 正则（推荐开启）
    # =========================================================
    config.blur = True
    config.blur_learnable = False
    config.blur_loss_w = 0.1
    config.blur_n = 1

    config.blur_kernel_size = 13
    config.blur_sigma = 1.5
    config.gussion_blur = True

    # =========================================================
    # 6. 初始化安全项（防坑）
    # =========================================================
    assert config.n_embd % config.n_head == 0

    return config


class SampleTopKRouter(nn.Module):
    def __init__(self, input_dim, num_experts, top_k=2, mode="mean"):
        super().__init__()
        self.gate = nn.Linear(input_dim, num_experts)
        self.top_k = top_k
        assert mode in ["mean", "cls"], "only support mean or cls pooling"
        self.mode = mode

    def forward(self, x):
        # x: [B, S, D]
        if self.mode == "mean":
            pooled = x.mean(dim=1)           # [B, D]
        else:
            pooled = x[:, 0]                 # [B, D] use first token as CLS

        logits = self.gate(pooled)           # [B, E]
        top_k_logits, indices = logits.topk(self.top_k, dim=-1)  # [B,K], [B,K]
        weights = F.softmax(top_k_logits, dim=-1)                 # [B,K]

        return weights, indices

class SeqMoE(nn.Module):
    def __init__(self, input_dim, num_experts=4, top_k=2, router_mode="mean"):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        
        self.router = SampleTopKRouter(input_dim, num_experts, top_k, mode=router_mode)
        self.experts = nn.ModuleList([
            Expert(input_dim, input_dim*4, input_dim)
            for _ in range(num_experts)
        ])

    def forward(self, x):
        # x: [B, S, D]
        B, S, D = x.shape
        
        # 1. routing: 每个 sample 选 top-k expert
        routing_weights, selected_experts = self.router(x)
        # routing_weights:   [B, K]
        # selected_experts:  [B, K]

        # 2. 初始化输出
        out = torch.zeros_like(x)

        # 3. 对每个 expert 聚合 sample（并行思想核心）
        #   idea：expert e ← 所有选中它的样本（但按 token 展开并行）
        for e in range(self.num_experts):

            # expert e 的 mask: [B, K]
            mask_e = (selected_experts == e)     # bool
            
            if not mask_e.any():
                continue  # 没有 sample 选这个 expert，跳过
            
            # 哪些样本选了 expert e
            selected_samples = mask_e.any(dim=-1)   # [B]
            
            # 提取这些样本:  [num_samples, S, D]
            x_e = x[selected_samples]  
            
            # expert_forward: [num_samples, S, D]
            y_e = self.experts[e](x_e)
            
            # 获取这些样本对应的权重: [num_samples, K]
            w_e_full = routing_weights[selected_samples]
            
            # 获取对应 mask: [num_samples, K]
            mask_e_full = mask_e[selected_samples]

            # 权重合并 multiple-k 的情况（rare but correct）
            # 最终得到 w_e: [num_samples, 1, 1]
            w_e = (w_e_full * mask_e_full).sum(dim=-1).view(-1,1,1)
            
            # 加回输出
            out[selected_samples] += y_e * w_e

        return out

class Expert(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        return self.net(x)

class TopKRouter(nn.Module):
    def __init__(self, input_dim, num_experts, top_k=2):
        super().__init__()
        self.gate = nn.Linear(input_dim, num_experts)
        self.top_k = top_k

    def forward(self, x):
        # x shape: [batch_size, seq_len, input_dim]
        # 计算路由 logits: [batch_size, seq_len, num_experts]
        logits = self.gate(x)
        
        # 获取 Top-K 的分数和索引
        # values: 每个 token 在选定专家上的原始分数
        # indices: 被选中的专家 ID
        top_k_logits, indices = logits.topk(self.top_k, dim=-1)
        
        # 对分数进行 Softmax 归一化，作为权重
        routing_weights = F.softmax(top_k_logits, dim=-1)
        
        return routing_weights, indices

class TokenMoE(nn.Module):
    def __init__(self, input_dim, num_experts=4, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        
        # 初始化路由
        self.router = TopKRouter(input_dim, num_experts, top_k)
        
        # 初始化专家列表 (使用 ModuleList)
        self.experts = nn.ModuleList([
            Expert(input_dim, input_dim*4, input_dim) for _ in range(num_experts)
        ])

    def forward(self, x):
        # x shape: [batch_size, seq_len, input_dim]
        batch_size, seq_len, _ = x.shape
        
        # 1. 路由决策：获取权重和专家索引
        routing_weights, selected_experts_indices = self.router(x)
        # routing_weights shape: [batch_size, seq_len, top_k]
        # selected_experts_indices shape: [batch_size, seq_len, top_k]
        
        # 准备输出张量
        final_output = torch.zeros_like(x)
        
        # 展平数据以便处理：将 batch 和 seq 维度合并
        # flat_x: [total_tokens, input_dim]
        flat_x = x.view(-1, x.shape[-1])
        
        # 展平索引和权重
        flat_indices = selected_experts_indices.view(-1, self.top_k)
        flat_weights = routing_weights.view(-1, self.top_k)
        
        # 遍历每一个专家
        for i in range(self.num_experts):
            # 创建掩码：找出所有选择了专家 i 的 token
            # expert_mask shape: [total_tokens, top_k]
            expert_mask = (flat_indices == i)
            
            # 只要这个专家在 top_k 中出现过任意一次，我们就认为它被选中了
            # any_matches shape: [total_tokens] -> True/False
            any_matches = expert_mask.any(dim=-1)
            
            if any_matches.sum() == 0:
                continue # 如果没有 token 选这个专家，跳过
            
            # 选出在这个专家上需要计算的 token
            # selected_input shape: [num_selected_tokens, input_dim]
            selected_input = flat_x[any_matches]
            
            # --- 专家 i 进行前向传播 ---
            # expert_output shape: [num_selected_tokens, output_dim]
            expert_output = self.experts[i](selected_input)
            
            # --- 结果加权并聚合 ---
            weight_for_expert = flat_weights[any_matches]
            # mask_for_selected: [num_selected_tokens, top_k]
            mask_for_selected = expert_mask[any_matches]
            weight_val = (weight_for_expert * mask_for_selected).sum(dim=-1).unsqueeze(-1)
            update = torch.zeros_like(final_output.view(-1, final_output.shape[-1]))
            update[any_matches] = expert_output * weight_val
            
            final_output = final_output + update.view(batch_size, seq_len, -1)
            
        return final_output



class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("mask", torch.tril(torch.ones(config.max_seqlen, config.max_seqlen))
                                     .view(1, 1, config.max_seqlen, config.max_seqlen))
        self.n_head = config.n_head

    def forward(self, x, layer_past=None):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y

class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config,layer_id=None):
        super().__init__()
        self.layer_id = layer_id
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        strategy = config.moe_strategy
        if strategy == "none":
            self.mlp = nn.Sequential(
                nn.Linear(config.n_embd, 4 * config.n_embd),
                nn.GELU(),
                nn.Linear(4 * config.n_embd, config.n_embd),
                nn.Dropout(config.resid_pdrop),
            )

        elif strategy == "token":
            self.mlp = TokenMoE(config.n_embd, num_experts=config.num_experts, top_k=config.top_k)

        elif strategy == "seq":
            self.mlp = SeqMoE(config.n_embd, num_experts=config.num_experts, top_k=config.top_k)

        elif strategy == "interleave":
            # 偶数层 token，奇数层 seq
            if layer_id % 2 == 0:
                self.mlp = TokenMoE(config.n_embd, num_experts=config.num_experts, top_k=config.top_k)
            else:
                self.mlp = SeqMoE(config.n_embd, num_experts=config.num_experts, top_k=config.top_k)
            
        # 前一半层用tokenmoe，后一半层用seqmoe
        elif strategy == "mixts":
            if layer_id < config.n_layer // 2:
                self.mlp = TokenMoE(config.n_embd, num_experts=config.num_experts, top_k=config.top_k)
            else:
                self.mlp = SeqMoE(config.n_embd, num_experts=config.num_experts, top_k=config.top_k)
        # 前一半层用seqmoe，后一半层用tokenmoe
        elif strategy == "mixst":
            if layer_id < config.n_layer // 2:
                self.mlp = SeqMoE(config.n_embd, num_experts=config.num_experts, top_k=config.top_k)
            else:
                self.mlp = TokenMoE(config.n_embd, num_experts=config.num_experts, top_k=config.top_k)

        else:
            raise ValueError(f"Unknown moe_strategy: {strategy}")

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class Model(nn.Module):
    """Transformer for AIS trajectories."""

    def __init__(self, config, partition_model = None):
        super().__init__()
        self.name = 'traisformerQ'
        self.turn_weight = config.turn_weight
        self.turn_thresh = config.turn_thresh
        self.add_logits = config.add_logits
        self.lat_size = config.lat_size
        self.lon_size = config.lon_size
        self.sog_size = config.sog_size
        self.cog_size = config.cog_size
        self.full_size = config.full_size
        self.n_lat_embd = config.n_lat_embd
        self.n_lon_embd = config.n_lon_embd
        self.n_sog_embd = config.n_sog_embd
        self.n_cog_embd = config.n_cog_embd
        self.moe_head = config.moe_head
        self.moe_balance_w = config.moe_balance_w
        self.register_buffer(
            "att_sizes", 
            torch.tensor([config.lat_size, config.lon_size, config.sog_size, config.cog_size]))
        self.register_buffer(
            "emb_sizes", 
            torch.tensor([config.n_lat_embd, config.n_lon_embd, config.n_sog_embd, config.n_cog_embd]))
        
        self.stats_en = LLMStatsEncoder(config)
        self.attn = nn.MultiheadAttention(config.n_embd, 6,batch_first=True)

        self.text_mode = config.text_mode
        self.turn_loss = config.turn_loss
        self.text_att_alpha = config.text_att_alpha
        if config.text_mode == 6:
            self.text_prior = TextPrior(config)

        if hasattr(config,"partition_mode"):
            self.partition_mode = config.partition_mode
        else:
            self.partition_mode = "uniform"
        self.partition_model = partition_model
        
        if hasattr(config,"blur"):
            self.blur = config.blur
            self.blur_learnable = config.blur_learnable
            self.blur_loss_w = config.blur_loss_w
            self.blur_n = config.blur_n
            if self.blur:
                if config.gussion_blur:
                    kernel_size = config.blur_kernel_size
                    sigma = config.blur_sigma      
                    g_kernel = self.create_gaussian_kernel(kernel_size, sigma)

                    self.blur_module = nn.Conv1d(1, 1, kernel_size, padding=kernel_size//2, bias=False)
                    self.blur_module.weight.data = g_kernel
                    self.blur_module.weight.requires_grad = False # 强烈建议设为False
                else:
                    self.blur_module = nn.Conv1d(1, 1, 3, padding = 1, padding_mode = 'replicate', groups=1, bias=False)
                    if not self.blur_learnable:
                        for params in self.blur_module.parameters():
                            params.requires_grad = False
                            params.fill_(1/3)
            else:
                self.blur_module = None
                
        
        if hasattr(config,"lat_min"): # the ROI is provided.
            self.lat_min = config.lat_min
            self.lat_max = config.lat_max
            self.lon_min = config.lon_min
            self.lon_max = config.lon_max
            self.lat_range = config.lat_max-config.lat_min
            self.lon_range = config.lon_max-config.lon_min
            self.sog_range = 30.
            
        if hasattr(config,"mode"): # mode: "pos" or "velo".
            # "pos": predict directly the next positions.
            # "velo": predict the velocities, use them to 
            # calculate the next positions.
            self.mode = config.mode
        else:
            self.mode = "pos"
    

        # Passing from the 4-D space to a high-dimentional space
        self.lat_emb = nn.Embedding(self.lat_size, config.n_lat_embd)
        self.lon_emb = nn.Embedding(self.lon_size, config.n_lon_embd)
        self.sog_emb = nn.Embedding(self.sog_size, config.n_sog_embd)
        self.cog_emb = nn.Embedding(self.cog_size, config.n_cog_embd)
            
            
        self.pos_emb = nn.Parameter(torch.zeros(1, config.max_seqlen, config.n_embd))
        self.drop = nn.Dropout(config.embd_pdrop)
        
        # transformer
        self.blocks = nn.ModuleList([Block(config, layer_id=i) for i in range(config.n_layer)])

        
        # decoder head
        self.ln_f = nn.LayerNorm(config.n_embd)
        if self.mode in ("mlp_pos","mlp"):
            self.head = nn.Linear(config.n_embd, config.n_embd, bias=False)
        elif self.moe_head:
            self.head = ParallelSequenceMoEHead(config.n_embd, 
                                                config.n_embd, 
                                                self.full_size, 
                                                num_experts=config.head_num_experts, 
                                                top_k=config.head_ex_top_k)
        else:
            self.head = nn.Linear(config.n_embd, self.full_size, bias=False) # Classification head
            
        self.max_seqlen = config.max_seqlen
        self.apply(self._init_weights)

        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

    def create_gaussian_kernel(self,kernel_size=13, sigma=1.5):
        """生成一个1D高斯卷积核"""
        x = torch.arange(-(kernel_size // 2), kernel_size // 2 + 1).float()
        kernel = torch.exp(-0.5 * (x / sigma)**2)
        kernel = kernel / kernel.sum() # 归一化，保证概率总和为1
        return kernel.view(1, 1, -1)


    def get_max_seqlen(self):
        return self.max_seqlen

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv1d)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add('pos_emb')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer
   
    
    def to_indexes(self, x, mode="uniform"):
        """Convert tokens to indexes.
        
        Args:
            x: a Tensor of size (batchsize, seqlen, 4). x has been truncated 
                to [0,1).
            model: currenly only supports "uniform".
        
        Returns:
            idxs: a Tensor (dtype: Long) of indexes.
        """
        bs, seqlen, data_dim = x.shape
        if mode == "uniform":
            # idxs = (x*self.att_sizes).long()
            idxs = torch.floor(x * self.att_sizes).long()
            idxs = torch.minimum(idxs, self.att_sizes - 1)
            return idxs, idxs
        elif mode in ("freq", "freq_uniform"):
            
            idxs = torch.floor(x * self.att_sizes).long()
            idxs = torch.minimum(idxs, self.att_sizes - 1)
            idxs_uniform = idxs.clone()
            discrete_lats, discrete_lons, lat_ids, lon_ids = self.partition_model(x[:,:,:2])
            idxs[:,:,0] = torch.round(lat_ids.reshape((bs,seqlen))).long()
            idxs[:,:,1] = torch.round(lon_ids.reshape((bs,seqlen))).long()                               
            return idxs, idxs_uniform
    def forward(self, x, masks=None,
                x_dec=None, x_mark_dec=None,
                stats=None, with_targets=False, return_loss_tuple=False):
    # def forward(self, x, masks = None, with_targets=False, return_loss_tuple=False):
        """
        Args:
            x: a Tensor of size (batchsize, seqlen, 4). x has been truncated 
                to [0,1).
            masks: a Tensor of the same size of x. masks[idx] = 0. if 
                x[idx] is a padding.
            with_targets: if True, inputs = x[:,:-1,:], targets = x[:,1:,:], 
                otherwise inputs = x.
        Returns: 
            logits, loss
        """
        x = x.contiguous()
        x = x[:, :, :4]
        if self.mode in ("mlp_pos","mlp",):
            idxs, idxs_uniform = x, x # use the real-values of x.
        else:            
            # Convert to indexes
            idxs, idxs_uniform = self.to_indexes(x, mode=self.partition_mode)
        
        if with_targets:
            inputs = idxs[:,:-1,:].contiguous()
            targets = idxs[:,1:,:].contiguous()
            targets_uniform = idxs_uniform[:,1:,:].contiguous()
            inputs_real = x[:,:-1,:].contiguous()
            targets_real = x[:,1:,:].contiguous()
        else:
            inputs_real = x
            inputs = idxs
            targets = None
            
        batchsize, seqlen, _ = inputs.size()
        assert seqlen <= self.max_seqlen, "Cannot forward, model block size is exhausted."

        # forward the GPT model
        lat_embeddings = self.lat_emb(inputs[:,:,0]) # (bs, seqlen, lat_size)
        lon_embeddings = self.lon_emb(inputs[:,:,1]) 
        sog_embeddings = self.sog_emb(inputs[:,:,2]) 
        cog_embeddings = self.cog_emb(inputs[:,:,3])      
        
        
        if self.text_mode > 0:
            # --------------- cross att ----------------------------------
            dec_embeddings = self.stats_en(stats) # b,L,768 or b,768
            if self.stats_en.get_last:
                dec_embeddings = dec_embeddings.unsqueeze(1).repeat(1, lat_embeddings.size(1), 1) # b,t，768
            if self.text_mode == 6:
                p_lat, p_lon, p_sog, p_cog = self.text_prior(dec_embeddings, temp=1.0)
            token_embeddings = torch.cat((lat_embeddings, lon_embeddings, sog_embeddings, cog_embeddings),dim=-1)

            # token_embeddings, _ = self.attn(token_embeddings, dec_embeddings, dec_embeddings)
            # --------------- cross att ----------------------------------
            if self.text_mode == 1:
                position_embeddings = self.pos_emb[:,:seqlen+dec_embeddings.shape[1],:]
            else:
                position_embeddings = self.pos_emb[:, :seqlen, :] # each position maps to a (learnable) vector (1, seqlen, n_embd)
            
            if self.text_mode == 1:
                token_embeddings = torch.cat((dec_embeddings,token_embeddings),dim=1)

            if self.text_mode == 4:
                fea = self.drop(token_embeddings + position_embeddings + dec_embeddings) # (bs, seqlen, n_embd)
            else:
                fea = self.drop(token_embeddings + position_embeddings) # (bs, seqlen, n_embd)
            if self.text_mode == 1:
                # fea = self.blocks(fea)[:,-seqlen:,:] # (bs, seqlen, n_embd)
                for block in self.blocks:
                    fea = block(fea)
                fea = fea[:, -seqlen:, :]
            else:
                # fea = self.blocks(fea) # (bs, seqlen, n_embd)
                for block in self.blocks:
                    fea = block(fea)

            fea = self.ln_f(fea) # (bs, seqlen, n_embd)
            if self.moe_head:
                last_token = dec_embeddings[:, -1, :]  # (bs, n_embd)
                # Mean Pooling: 捕捉全局背景语义 (排除 Padding 部分)
                mean_pool = torch.mean(dec_embeddings, dim=1) 

                # Max Pooling: 捕捉强指令特征
                max_pool, _ = torch.max(dec_embeddings, dim=1)

                gate_input = max_pool
                logits, aux_loss, moe_stats = self.head(fea, gate_input) # (bs, seqlen, full_size) 
            else:
                logits = self.head(fea) # (bs, seqlen, full_size) or (bs, seqlen, n_embd)
            if self.text_mode == 5 and self.stats_en.get_last:
                logits = logits + dec_embeddings
            lat_logits, lon_logits, sog_logits, cog_logits =\
                torch.split(logits, (self.lat_size, self.lon_size, self.sog_size, self.cog_size), dim=-1)
            if self.text_mode == 6:
                B, S, _ = lat_logits.shape
                p_lat = p_lat.unsqueeze(1).expand(B, S, -1)  # (B, S, C_lat)
                p_lon = p_lon.unsqueeze(1).expand(B, S, -1)
                p_sog = p_sog.unsqueeze(1).expand(B, S, -1)
                p_cog = p_cog.unsqueeze(1).expand(B, S, -1)
                alpha = self.text_att_alpha  # 超参，0.1 ~ 1.0 可调
                eps = 1e-9

                lat_logits = lat_logits + alpha * torch.log(p_lat + eps)
                lon_logits = lon_logits + alpha * torch.log(p_lon + eps)
                sog_logits = sog_logits + alpha * torch.log(p_sog + eps)
                cog_logits = cog_logits + alpha * torch.log(p_cog + eps)
        else:
            token_embeddings = torch.cat((lat_embeddings, lon_embeddings, sog_embeddings, cog_embeddings),dim=-1)
            
            position_embeddings = self.pos_emb[:, :seqlen, :] # each position maps to a (learnable) vector (1, seqlen, n_embd)
            fea = self.drop(token_embeddings + position_embeddings)
            # fea = self.blocks(fea)
            for block in self.blocks:
                fea = block(fea)

            fea = self.ln_f(fea) # (bs, seqlen, n_embd)
            logits = self.head(fea) # (bs, seqlen, full_size) or (bs, seqlen, n_embd)
            
            lat_logits, lon_logits, sog_logits, cog_logits =\
                torch.split(logits, (self.lat_size, self.lon_size, self.sog_size, self.cog_size), dim=-1)
            
        # Calculate the loss
        loss = None
        loss_tuple = None
        if targets is not None:

            sog_loss = F.cross_entropy(sog_logits.view(-1, self.sog_size), 
                                       targets[:,:,2].view(-1), 
                                       reduction="none").view(batchsize,seqlen)
            cog_loss = F.cross_entropy(cog_logits.view(-1, self.cog_size), 
                                       targets[:,:,3].view(-1), 
                                       reduction="none").view(batchsize,seqlen)
            lat_loss = F.cross_entropy(lat_logits.view(-1, self.lat_size), 
                                       targets[:,:,0].view(-1), 
                                       reduction="none").view(batchsize,seqlen)
            lon_loss = F.cross_entropy(lon_logits.view(-1, self.lon_size), 
                                       targets[:,:,1].view(-1), 
                                       reduction="none").view(batchsize,seqlen)                     

            if self.blur:
                lat_probs = F.softmax(lat_logits, dim=-1) 
                lon_probs = F.softmax(lon_logits, dim=-1)
                sog_probs = F.softmax(sog_logits, dim=-1)
                cog_probs = F.softmax(cog_logits, dim=-1)

                for _ in range(self.blur_n):
                    blurred_lat_probs = self.blur_module(lat_probs.reshape(-1,1,self.lat_size)).reshape(lat_probs.shape)
                    blurred_lon_probs = self.blur_module(lon_probs.reshape(-1,1,self.lon_size)).reshape(lon_probs.shape)
                    blurred_sog_probs = self.blur_module(sog_probs.reshape(-1,1,self.sog_size)).reshape(sog_probs.shape)
                    blurred_cog_probs = self.blur_module(cog_probs.reshape(-1,1,self.cog_size)).reshape(cog_probs.shape)

                    blurred_lat_loss = F.nll_loss(blurred_lat_probs.view(-1, self.lat_size),
                                                  targets[:,:,0].view(-1),
                                                  reduction="none").view(batchsize,seqlen)
                    blurred_lon_loss = F.nll_loss(blurred_lon_probs.view(-1, self.lon_size),
                                                  targets[:,:,1].view(-1),
                                                  reduction="none").view(batchsize,seqlen)
                    blurred_sog_loss = F.nll_loss(blurred_sog_probs.view(-1, self.sog_size),
                                                  targets[:,:,2].view(-1),
                                                  reduction="none").view(batchsize,seqlen)
                    blurred_cog_loss = F.nll_loss(blurred_cog_probs.view(-1, self.cog_size),
                                                  targets[:,:,3].view(-1),
                                                  reduction="none").view(batchsize,seqlen)

                    lat_loss += self.blur_loss_w*blurred_lat_loss
                    lon_loss += self.blur_loss_w*blurred_lon_loss
                    sog_loss += self.blur_loss_w*blurred_sog_loss
                    cog_loss += self.blur_loss_w*blurred_cog_loss

                    lat_probs = blurred_lat_probs
                    lon_probs = blurred_lon_probs
                    sog_probs = blurred_sog_probs
                    cog_probs = blurred_cog_probs
                    

            loss_tuple = (lat_loss, lon_loss, sog_loss, cog_loss)
            loss = sum(loss_tuple)
            # loss += self.moe_balance_w*aux_loss if self.moe_head else 0.0
        

            # =========[ 关键改动：转向加权 Loss ]==========
            if self.turn_loss:
                # 获取 COG 连续差分（基于分类ID, 也可用 regression COG）
                cog_vals = targets[:,:,3].float()
                dcog = torch.abs(cog_vals[:,1:] - cog_vals[:,:-1])
                turn_mask = (dcog > self.turn_thresh)    # e.g. turn_thresh = 2~5 bins
                turn_mask = torch.cat([turn_mask, turn_mask[:,-1:]], dim=1) # pad

                # 应用加权
                loss = loss * (1 + self.turn_weight * turn_mask.float())

            loss = loss.mean()
        
        if return_loss_tuple:
            return logits, loss, loss_tuple
        else:
            return logits, loss
        





def top_k_logits(logits, k):
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:, [-1]]] = -float('Inf')
    return out

def top_k_nearest_idx(att_logits, att_idxs, r_vicinity):
    """Keep only k values nearest the current idx.
    
    Args:
        att_logits: a Tensor of shape (bachsize, data_size). 
        att_idxs: a Tensor of shape (bachsize, 1), indicates 
            the current idxs.
        r_vicinity: number of values to be kept.
    """
    device = att_logits.device
    idx_range = torch.arange(att_logits.shape[-1]).to(device).repeat(att_logits.shape[0],1)
    idx_dists = torch.abs(idx_range - att_idxs)
    out = att_logits.clone()
    out[idx_dists >= r_vicinity/2] = -float('Inf')
    return out

@torch.no_grad()
def sample(model,
           seqs,
           stats,
           steps,
           temperature=1.0,
           sample=False,
           sample_mode="pos_vicinity",
           r_vicinity=20,
           top_k=None):
    """
    Take a conditoning sequence of AIS observations seq and predict the next observation,
    feed the predictions back into the model each time. 
    """
    max_seqlen = model.get_max_seqlen()
    model.eval()
    for k in range(steps):
        seqs_cond = seqs if seqs.size(1) <= max_seqlen else seqs[:, -max_seqlen:]  # crop context if needed

        # logits.shape: (batch_size, seq_len, data_size)
        logits, _ = model(seqs_cond,None,None,None,stats)
        d2inf_pred = torch.zeros((logits.shape[0], 4)).to(seqs.device) + 0.5

        # pluck the logits at the final step and scale by temperature
        logits = logits[:, -1, :] / temperature  # (batch_size, data_size)

        lat_logits, lon_logits, sog_logits, cog_logits = \
            torch.split(logits, (model.lat_size, model.lon_size, model.sog_size, model.cog_size), dim=-1)

        # optionally crop probabilities to only the top k options
        if sample_mode in ("pos_vicinity",):
            idxs, idxs_uniform = model.to_indexes(seqs_cond[:, -1:, :])
            lat_idxs, lon_idxs = idxs_uniform[:, 0, 0:1], idxs_uniform[:, 0, 1:2]
            lat_logits = top_k_nearest_idx(lat_logits, lat_idxs, r_vicinity)
            lon_logits = top_k_nearest_idx(lon_logits, lon_idxs, r_vicinity)

        if top_k is not None:
            lat_logits = top_k_logits(lat_logits, top_k)
            lon_logits = top_k_logits(lon_logits, top_k)
            sog_logits = top_k_logits(sog_logits, top_k)
            cog_logits = top_k_logits(cog_logits, top_k)

        # apply softmax to convert to probabilities
        lat_probs = F.softmax(lat_logits, dim=-1)
        lon_probs = F.softmax(lon_logits, dim=-1)
        sog_probs = F.softmax(sog_logits, dim=-1)
        cog_probs = F.softmax(cog_logits, dim=-1)

        # sample from the distribution or take the most likely
        if sample:
            lat_ix = torch.multinomial(lat_probs, num_samples=1)  # (batch_size, 1)
            lon_ix = torch.multinomial(lon_probs, num_samples=1)
            sog_ix = torch.multinomial(sog_probs, num_samples=1)
            cog_ix = torch.multinomial(cog_probs, num_samples=1)
        else:
            _, lat_ix = torch.topk(lat_probs, k=1, dim=-1)
            _, lon_ix = torch.topk(lon_probs, k=1, dim=-1)
            _, sog_ix = torch.topk(sog_probs, k=1, dim=-1)
            _, cog_ix = torch.topk(cog_probs, k=1, dim=-1)

        ix = torch.cat((lat_ix, lon_ix, sog_ix, cog_ix), dim=-1)
        # convert to x (range: [0,1))
        x_sample = (ix.float() + d2inf_pred) / model.att_sizes

        # append to the sequence and continue
        seqs = torch.cat((seqs, x_sample.unsqueeze(1)), dim=1)

    return seqs