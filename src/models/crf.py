"""纯 PyTorch 实现的 CRF (Conditional Random Field) 层。

用于 BERT 之上, 提升实体边界的合法性与准确率。

参考: pytorch-crf / "Bidirectional LSTM-CRF Models for Sequence Tagging"
特性:
  - 支持 batch, 自定义 START/END tag
  - forward() 返回负对数似然损失 (-log likelihood)
  - decode() 用 Viterbi 算法返回最优标签序列

用法:
    crf = CRF(num_tags=9, batch_first=True)
    emissions = bert(...)            # [B, T, num_tags]
    loss = -crf(emissions, tags, mask=mask)  # 标量
    best = crf.decode(emissions, mask=mask)  # list[list[int]]
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn


class CRF(nn.Module):
    def __init__(self, num_tags: int, batch_first: bool = True):
        super().__init__()
        self.num_tags = num_tags
        self.batch_first = batch_first
        # 转移矩阵 transitions[i, j]: 从 tag i 转移到 tag j 的分数
        self.start_transitions = nn.Parameter(torch.empty(num_tags))
        self.end_transitions = nn.Parameter(torch.empty(num_tags))
        self.transitions = nn.Parameter(torch.empty(num_tags, num_tags))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)
        nn.init.uniform_(self.transitions, -0.1, 0.1)

    # ---------------- 前向算法 (配分函数) ----------------
    def _forward_alg(self, emissions: torch.Tensor,
                     mask: torch.Tensor) -> torch.Tensor:
        # emissions: [B, T, C], mask: [B, T]
        B, T, C = emissions.shape
        # alpha: 当前时刻各 tag 的累积对数概率
        alpha = self.start_transitions.unsqueeze(0) + emissions[:, 0]  # [B, C]
        for t in range(1, T):
            # 广播: alpha[B,1,C] + transitions[1,C,C] + emit[B,1,C]
            emit = emissions[:, t].unsqueeze(1)         # [B, 1, C]
            trans = self.transitions.unsqueeze(0)        # [1, C, C]
            scores = alpha.unsqueeze(2) + trans + emit   # [B, C(from), C(to)]
            new_alpha = torch.logsumexp(scores, dim=1)   # [B, C]
            m = mask[:, t].unsqueeze(1)                  # [B, 1]
            alpha = m * new_alpha + (1 - m) * alpha
        alpha = alpha + self.end_transitions.unsqueeze(0)
        return torch.logsumexp(alpha, dim=1)             # [B]

    # ---------------- 分子 (gold 序列分数) ----------------
    def _score(self, emissions: torch.Tensor, tags: torch.LongTensor,
               mask: torch.Tensor) -> torch.Tensor:
        B, T, C = emissions.shape
        # 起点
        score = self.start_transitions[tags[:, 0]]       # [B]
        score = score + emissions[torch.arange(B), 0, tags[:, 0]]
        for t in range(1, T):
            emit = emissions[torch.arange(B), t, tags[:, t]]
            trans = self.transitions[tags[:, t - 1], tags[:, t]]
            m = mask[:, t]
            score = score + m * (trans + emit)
        # 末尾转移
        # 找到每个样本最后一个有效 tag
        seq_lens = mask.sum(dim=1).long() - 1            # [B]
        last_tags = tags[torch.arange(B), seq_lens]
        score = score + self.end_transitions[last_tags]
        return score

    def forward(self, emissions: torch.Tensor, tags: torch.LongTensor,
                mask: torch.Tensor | None = None) -> torch.Tensor:
        """返回对数似然 logZ 分子-分母 (越大越好)。loss = -forward()."""
        if not self.batch_first:
            emissions = emissions.transpose(0, 1)
            tags = tags.transpose(0, 1)
            if mask is not None:
                mask = mask.transpose(0, 1)
        if mask is None:
            mask = torch.ones_like(tags, dtype=torch.float)
        mask = mask.float()
        # 忽略 -100 padding: 把 mask 中对应位置视为 0
        # (注意调用方应保证 mask 已正确, -100 仅在标签张量中)
        Z = self._forward_alg(emissions, mask)          # [B]
        num = self._score(emissions, tags, mask)         # [B]
        return num - Z                                   # [B]

    # ---------------- Viterbi 解码 ----------------
    @torch.no_grad()
    def decode(self, emissions: torch.Tensor,
               mask: torch.Tensor | None = None) -> list[list[int]]:
        if not self.batch_first:
            emissions = emissions.transpose(0, 1).contiguous()
            if mask is not None:
                mask = mask.transpose(0, 1).contiguous()
        B, T, C = emissions.shape
        if mask is None:
            mask = torch.ones(B, T, dtype=torch.float, device=emissions.device)
        mask = mask.float()

        results = []
        for b in range(B):
            seq_len = int(mask[b].sum().item())
            emit = emissions[b, :seq_len].unsqueeze(0)  # [1, L, C]
            history = []
            alpha = self.start_transitions.unsqueeze(0) + emit[:, 0]  # [1, C]
            for t in range(1, seq_len):
                scores = alpha.unsqueeze(2) + self.transitions.unsqueeze(0) \
                         + emit[:, t].unsqueeze(1)         # [1, C(from), C(to)]
                best, idx = scores.max(dim=1)               # [1, C]
                alpha = best
                history.append(idx.squeeze(0))              # [C]
            alpha = alpha + self.end_transitions.unsqueeze(0)
            best_last = int(alpha.argmax(dim=1).item())
            # 回溯
            best_path = [best_last]
            for hist in reversed(history):
                best_last = int(hist[best_last].item())
                best_path.append(best_last)
            best_path.reverse()
            results.append(best_path)
        return results
