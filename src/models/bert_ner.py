"""BERT NER 模型: bert-base-chinese + Linear (+ 可选 CRF)。

输出: 每个 token 对应 9 个标签 (O, B/I-LOC, B/I-PER, B/I-DIS, B/I-NEED) 的分布。
训练时:
  - 不带 CRF: 用 CrossEntropyLoss (忽略 -100 padding)
  - 带 CRF: 用 CRF 的负对数似然
推理时: CRF 用 Viterbi 解码; 否则 argmax。

提供两个便捷方法:
  - predict(tokens) : 输入原始文本(字符串或字列表), 返回实体列表
  - extract(text)   : 同上, 返回更友好的结构化结果
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from src.config import (
    BERT_MODEL_NAME, ENTITY_TYPES, ID2TAG, MAX_LEN, NUM_TAGS, TAG2ID, USE_CRF,
)
from src.models.crf import CRF


class BertNER(nn.Module):
    def __init__(self, num_tags: int = NUM_TAGS,
                 model_name: str = BERT_MODEL_NAME, use_crf: bool = USE_CRF):
        super().__init__()
        from transformers import AutoModel
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        hidden = self.bert.config.hidden_size
        self.classifier = nn.Linear(hidden, num_tags)
        self.use_crf = use_crf
        self.num_tags = num_tags
        if use_crf:
            self.crf = CRF(num_tags, batch_first=True)

    # ---------------- 编码 (返回 emissions) ----------------
    def emit(self, input_ids, attention_mask, token_type_ids=None) -> torch.Tensor:
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        out = self.bert(**kwargs).last_hidden_state  # [B, T, H]
        out = self.dropout(out)
        return self.classifier(out)                   # [B, T, num_tags]

    # ---------------- 训练损失 ----------------
    def loss(self, input_ids, attention_mask, labels,
             token_type_ids=None) -> torch.Tensor:
        emissions = self.emit(input_ids, attention_mask, token_type_ids)
        # labels 中 -100 表示忽略; CRF 需要把 -100 改成有效标签且 mask=False
        if self.use_crf:
            # 构造用于 CRF 的标签和 mask
            crf_labels = labels.clone()
            crf_labels[crf_labels == -100] = 0
            mask = (labels != -100).float()
            # CRF.forward 返回对数似然 (per-sample), loss = -mean
            ll = self.crf(emissions, crf_labels, mask=mask)
            return -ll.mean()
        else:
            # 普通交叉熵, ignore_index=-100
            loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
            B, T, C = emissions.shape
            return loss_fn(emissions.view(-1, C), labels.view(-1))

    # ---------------- 推理 ----------------
    @torch.no_grad()
    def decode(self, input_ids, attention_mask, token_type_ids=None) -> list[list[int]]:
        """返回每个样本 (去除 [CLS]/[SEP]/[PAD]) 的标签 id 序列。

        与 labels 的对齐: labels 中 [CLS]/[SEP]/[PAD] 位置均为 -100 (忽略),
        所以 decode 输出也必须跳过这些位置, 才能和 gold 逐位置比较。
        bert-base-chinese 格式: [CLS] x x x ... x [SEP] [PAD]...
        即位置 0 是 [CLS], 第 L-1 个有效位置是 [SEP] (L = attention_mask.sum())。
        """
        emissions = self.emit(input_ids, attention_mask, token_type_ids)
        if self.use_crf:
            mask = attention_mask.float()
            raw = self.crf.decode(emissions, mask=mask)
            # CRF decode 返回包含 [CLS]/[SEP] 的序列, 需去掉首([CLS])尾([SEP])
            results = []
            for seq in raw:
                # 去掉第一个([CLS]) 和最后一个([SEP])
                if len(seq) >= 2:
                    results.append(seq[1:-1])
                else:
                    results.append(seq)
            return results
        else:
            preds = emissions.argmax(dim=-1)            # [B, T]
            results = []
            for i in range(preds.size(0)):
                L = int(attention_mask[i].sum().item())
                # 跳过位置 0 ([CLS]) 和位置 L-1 ([SEP])
                results.append(preds[i, 1:L-1].tolist())
            return results

    # ---------------- 保存 / 加载 ----------------
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path / "pytorch_model.bin")
        # 同步保存 tokenizer (调用方负责), 这里只保存模型

    @classmethod
    def load(cls, path: str | Path, map_location="cpu",
             use_crf: bool | None = None) -> "BertNER":
        path = Path(path)
        # use_crf 默认按 config
        if use_crf is None:
            use_crf = USE_CRF
        model = cls(use_crf=use_crf)
        state = torch.load(path / "pytorch_model.bin", map_location=map_location,
                           weights_only=True)
        model.load_state_dict(state)
        model.eval()
        return model


# ==================================================================
# 标签序列 -> 实体 (BIO 解码)
# ==================================================================
def tags_to_entities(tag_ids: list[int], id2tag: dict = ID2TAG) -> list[dict]:
    """把标签 id 序列解码为实体列表 [{type, start, end, text 占位}]。

    输出 text 为空, 需调用方填入对应 token。
    """
    entities = []
    cur = None  # {"type", "start"}
    for i, tid in enumerate(tag_ids):
        tag = id2tag.get(int(tid), "O")
        if tag == "O":
            if cur is not None:
                entities.append({"type": cur["type"], "start": cur["start"],
                                 "end": i})
                cur = None
        else:
            prefix, etype = tag.split("-", 1)
            if prefix == "B" or cur is None or cur["type"] != etype:
                if cur is not None:
                    entities.append({"type": cur["type"], "start": cur["start"],
                                     "end": i})
                cur = {"type": etype, "start": i}
            else:
                # I- 继续当前实体
                continue
    if cur is not None:
        entities.append({"type": cur["type"], "start": cur["start"],
                         "end": len(tag_ids)})
    return entities
