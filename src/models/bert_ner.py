"""BERT NER 模型：bert-base-chinese + Linear（+ 可选 CRF）。

模型结构::

    输入字符 → BERT (12 层 Transformer) → Dropout → Linear (768 → num_tags) → [CRF] → BIO 标签

两种模式：
    - 不带 CRF：损失使用 CrossEntropyLoss(ignore_index=-100)，推理使用 argmax。
    - 带 CRF：损失使用 CRF 负对数似然，推理使用 Viterbi 解码。
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from src.config import BERT_MODEL_NAME, ID2TAG, NUM_TAGS, USE_CRF
from src.models.crf import CRF


class BertNER(nn.Module):
    """BERT + 线性分类头（+ 可选 CRF）。

    Args:
        num_tags: 标签类别数，默认取自 config.NUM_TAGS。
        model_name: BERT 预训练模型名称或路径。
        use_crf: 是否在分类头后接 CRF 层。
    """

    def __init__(
        self,
        num_tags: int = NUM_TAGS,
        model_name: str = BERT_MODEL_NAME,
        use_crf: bool = USE_CRF,
    ) -> None:
        super().__init__()
        from transformers import AutoModel  # 延迟导入，避免模块级重依赖

        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_tags)

        self.num_tags = num_tags
        self.use_crf = use_crf
        if use_crf:
            self.crf = CRF(num_tags, batch_first=True)

    # ------------------------------------------------------------------
    # 前向计算
    # ------------------------------------------------------------------

    def emit(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """经 BERT + 分类头前向传播，返回每个位置的发射分数。

        Args:
            input_ids: token id，形状 ``[B, T]``。
            attention_mask: 注意力掩码，形状 ``[B, T]``。
            token_type_ids: segment id（可选），形状 ``[B, T]``。

        Returns:
            发射分数，形状 ``[B, T, num_tags]``。
        """
        kwargs: dict = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids

        hidden = self.bert(**kwargs).last_hidden_state       # [B, T, H]
        return self.classifier(self.dropout(hidden))         # [B, T, num_tags]

    def loss(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """计算训练损失。

        Args:
            input_ids: token id，形状 ``[B, T]``。
            attention_mask: 注意力掩码，形状 ``[B, T]``。
            labels: 金标签，形状 ``[B, T]``；-100 表示需忽略的位置
                    （[CLS] / [SEP] / [PAD]）。
            token_type_ids: segment id（可选），形状 ``[B, T]``。

        Returns:
            标量损失（batch 均值）。
        """
        emissions = self.emit(input_ids, attention_mask, token_type_ids)

        if self.use_crf:
            # CRF 不接受 -100，将忽略位置替换为合法标签并用 mask 屏蔽
            crf_labels = labels.clone()
            crf_labels[crf_labels == -100] = 0
            mask = (labels != -100).float()

            log_likelihood = self.crf(emissions, crf_labels, mask=mask)  # [B]
            return -log_likelihood.mean()

        # 无 CRF：标准交叉熵，展平后计算
        loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        B, T, C = emissions.shape
        return loss_fn(emissions.view(-1, C), labels.view(-1))

    # ------------------------------------------------------------------
    # 推理
    # ------------------------------------------------------------------

    @torch.no_grad()
    def decode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
    ) -> list[list[int]]:
        """解码每个样本的最优标签序列。

        返回结果已去除 [CLS]（位置 0）和 [SEP]（最后一个有效位置），
        与金标签中的 -100 位置对齐。

        Args:
            input_ids: token id，形状 ``[B, T]``。
            attention_mask: 注意力掩码，形状 ``[B, T]``。
            token_type_ids: segment id（可选），形状 ``[B, T]``。

        Returns:
            ``list[list[int]]``，每个子列表为对应样本的标签 id 序列。
        """
        emissions = self.emit(input_ids, attention_mask, token_type_ids)
        B = emissions.size(0)

        if self.use_crf:
            raw = self.crf.decode(emissions, attention_mask.float())
            # CRF 返回的序列含 [CLS]/[SEP]，去掉首尾
            return [seq[1:-1] if len(seq) >= 2 else seq for seq in raw]

        # 无 CRF：argmax 后按 attention_mask 截取有效区间
        preds = emissions.argmax(dim=-1)                     # [B, T]
        results: list[list[int]] = []
        for i in range(B):
            seq_len = int(attention_mask[i].sum().item())
            # 跳过位置 0 ([CLS]) 和位置 seq_len - 1 ([SEP])
            results.append(preds[i, 1 : seq_len - 1].tolist())
        return results

    # ------------------------------------------------------------------
    # 持久化
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """保存模型权重到指定目录（tokenizer 由调用方自行保存）。

        Args:
            path: 目标目录路径，不存在时自动创建。
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path / "pytorch_model.bin")

    @classmethod
    def load(
        cls,
        path: str | Path,
        map_location: str = "cpu",
        use_crf: bool | None = None,
    ) -> BertNER:
        """从目录加载模型权重。

        Args:
            path: 模型权重所在目录。
            map_location: torch.load 的设备映射。
            use_crf: 是否启用 CRF；为 None 时取 config.USE_CRF。

        Returns:
            加载权重后处于 eval 模式的 BertNER 实例。
        """
        if use_crf is None:
            use_crf = USE_CRF

        model = cls(use_crf=use_crf)
        state = torch.load(
            Path(path) / "pytorch_model.bin",
            map_location=map_location,
            weights_only=True,
        )
        model.load_state_dict(state)
        model.eval()
        return model


# ======================================================================
# BIO 解码工具
# ======================================================================

def tags_to_entities(
    tag_ids: list[int],
    id2tag: dict[int, str] = ID2TAG,
) -> list[dict]:
    """将标签 id 序列解码为实体列表（BIO 格式）。

    Args:
        tag_ids: 标签 id 序列。
        id2tag: 标签 id 到字符串的映射表。

    Returns:
        实体列表，每个元素形如 ``{"type": "LOC", "start": 4, "end": 9}``，
        其中 ``end`` 为 exclusive（切片风格），调用方以 ``text[start:end]`` 取实体文本。

    Example::

        >>> tags_to_entities([0, 3, 4, 4, 0])  # O B-LOC I-LOC I-LOC O
        [{'type': 'LOC', 'start': 1, 'end': 4}]
    """
    entities: list[dict] = []
    current: dict | None = None  # 正在拼接的实体：{"type": str, "start": int}

    for i, tid in enumerate(tag_ids):
        tag = id2tag.get(int(tid), "O")

        if tag == "O":
            if current is not None:
                entities.append({**current, "end": i})
                current = None
            continue

        prefix, entity_type = tag.split("-", 1)

        if prefix == "B" or current is None or current["type"] != entity_type:
            # B- 开头、类型变化、或前一个标签是 O —— 均视为新实体起点
            if current is not None:
                entities.append({**current, "end": i})
            current = {"type": entity_type, "start": i}
        # I- 且类型一致：继续当前实体，无需操作

    if current is not None:
        entities.append({**current, "end": len(tag_ids)})

    return entities