"""纯 PyTorch 实现的 CRF（Conditional Random Field）层。

用于序列标注任务，通常接在 BERT 等编码器之上，
通过建模标签间的转移关系来提升实体边界的合法性与准确率。

参考:
    - pytorch-crf
    - "Bidirectional LSTM-CRF Models for Sequence Tagging"

特性:
    - 支持 batch 运算
    - 支持自定义 START / END tag 转移分数
    - forward() 返回对数似然（loss = -forward()）
    - decode() 使用 Viterbi 算法解码最优标签序列

用法::

    crf = CRF(num_tags=9, batch_first=True)
    emissions = bert(...)                        # [B, T, num_tags]
    loss = -crf(emissions, tags, mask=mask)       # 标量损失
    best = crf.decode(emissions, mask=mask)       # list[list[int]]
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CRF(nn.Module):
    """条件随机场层。

    Args:
        num_tags: 标签类别数。
        batch_first: 若为 True，输入张量的第一维为 batch；否则为序列长度。
    """

    def __init__(self, num_tags: int, batch_first: bool = True) -> None:
        super().__init__()
        self.num_tags = num_tags
        self.batch_first = batch_first

        # 转移矩阵：transitions[i, j] 表示从 tag i 转移到 tag j 的分数
        self.transitions = nn.Parameter(torch.empty(num_tags, num_tags))
        self.start_transitions = nn.Parameter(torch.empty(num_tags))
        self.end_transitions = nn.Parameter(torch.empty(num_tags))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """使用均匀分布初始化所有转移参数。"""
        nn.init.uniform_(self.transitions, -0.1, 0.1)
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)

    # ------------------------------------------------------------------
    # 公共接口
    # ------------------------------------------------------------------

    def forward(
        self,
        emissions: torch.Tensor,
        tags: torch.LongTensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """计算对数似然（gold 序列分数 − log Z）。

        返回值越大越好；训练时取负号作为损失：``loss = -crf(...)``。

        Args:
            emissions: 发射分数，形状 ``[B, T, num_tags]``。
            tags: 金标签序列，形状 ``[B, T]``。
            mask: 掩码，形状 ``[B, T]``，1 表示有效位置，0 表示填充。
                  若为 None，则默认全 1。

        Returns:
            每个样本的对数似然，形状 ``[B]``。
        """
        if not self.batch_first:
            emissions = emissions.transpose(0, 1)
            tags = tags.transpose(0, 1)
            if mask is not None:
                mask = mask.transpose(0, 1)

        if mask is None:
            mask = torch.ones_like(tags, dtype=torch.float)
        mask = mask.float()

        log_z = self._compute_log_partition(emissions, mask)   # [B]
        gold_score = self._compute_gold_score(emissions, tags, mask)  # [B]
        return gold_score - log_z                               # [B]

    @torch.no_grad()
    def decode(
        self,
        emissions: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> list[list[int]]:
        """Viterbi 解码，返回每个样本的最优标签序列。

        Args:
            emissions: 发射分数，形状 ``[B, T, num_tags]``。
            mask: 掩码，形状 ``[B, T]``。若为 None，则默认全 1。

        Returns:
            ``list[list[int]]``，每个子列表为对应样本的最优标签路径。
        """
        if not self.batch_first:
            emissions = emissions.transpose(0, 1).contiguous()
            if mask is not None:
                mask = mask.transpose(0, 1).contiguous()

        B, T, C = emissions.shape
        if mask is None:
            mask = torch.ones(B, T, dtype=torch.float, device=emissions.device)
        mask = mask.float()

        results: list[list[int]] = []
        for b in range(B):
            seq_len = int(mask[b].sum().item())
            emit = emissions[b, :seq_len].unsqueeze(0)          # [1, L, C]
            best_path = self._viterbi(emit)
            results.append(best_path)
        return results

    # ------------------------------------------------------------------
    # 内部实现
    # ------------------------------------------------------------------

    def _compute_log_partition(
        self,
        emissions: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """前向算法：计算配分函数的对数（log Z）。

        Args:
            emissions: ``[B, T, C]``
            mask: ``[B, T]``

        Returns:
            ``[B]``
        """
        B, T, C = emissions.shape

        # alpha[b, c] = 截至当前时刻、以 tag c 结尾的所有路径的对数分数之和
        alpha = self.start_transitions.unsqueeze(0) + emissions[:, 0]  # [B, C]

        for t in range(1, T):
            emit = emissions[:, t].unsqueeze(1)          # [B, 1, C]
            trans = self.transitions.unsqueeze(0)        # [1, C, C]
            scores = alpha.unsqueeze(2) + trans + emit   # [B, C_from, C_to]
            new_alpha = torch.logsumexp(scores, dim=1)   # [B, C]

            # 对填充位置保持 alpha 不变
            m = mask[:, t].unsqueeze(1)                  # [B, 1]
            alpha = m * new_alpha + (1 - m) * alpha

        alpha = alpha + self.end_transitions.unsqueeze(0)
        return torch.logsumexp(alpha, dim=1)             # [B]

    def _compute_gold_score(
        self,
        emissions: torch.Tensor,
        tags: torch.LongTensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """计算金标签序列的未归一化分数（分子）。

        Args:
            emissions: ``[B, T, C]``
            tags: ``[B, T]``
            mask: ``[B, T]``

        Returns:
            ``[B]``
        """
        B, T, C = emissions.shape

        # 起点分数：START → tag_0 + emission_0
        score = self.start_transitions[tags[:, 0]]                  # [B]
        score = score + emissions[torch.arange(B), 0, tags[:, 0]]

        # 逐步累加转移分数与发射分数
        for t in range(1, T):
            trans = self.transitions[tags[:, t - 1], tags[:, t]]
            emit = emissions[torch.arange(B), t, tags[:, t]]
            m = mask[:, t]
            score = score + m * (trans + emit)

        # 末尾分数：last_tag → END
        seq_lens = mask.sum(dim=1).long() - 1                       # [B]
        last_tags = tags[torch.arange(B), seq_lens]
        score = score + self.end_transitions[last_tags]
        return score                                                # [B]

    def _viterbi(self, emit: torch.Tensor) -> list[int]:
        """对单条序列执行 Viterbi 解码。

        Args:
            emit: ``[1, L, C]``，单条序列的发射分数。

        Returns:
            最优标签路径。
        """
        seq_len = emit.size(1)
        history: list[torch.Tensor] = []

        alpha = self.start_transitions.unsqueeze(0) + emit[:, 0]    # [1, C]

        for t in range(1, seq_len):
            scores = (
                alpha.unsqueeze(2)
                + self.transitions.unsqueeze(0)
                + emit[:, t].unsqueeze(1)
            )                                                       # [1, C_from, C_to]
            best_scores, best_indices = scores.max(dim=1)           # [1, C]
            alpha = best_scores
            history.append(best_indices.squeeze(0))                 # [C]

        # 加上 END 转移，找到最优末尾标签
        alpha = alpha + self.end_transitions.unsqueeze(0)
        best_last = int(alpha.argmax(dim=1).item())

        # 回溯构建最优路径
        best_path = [best_last]
        for hist in reversed(history):
            best_last = int(hist[best_last].item())
            best_path.append(best_last)
        best_path.reverse()
        return best_path