import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================
# 1. SE Block
# =========================
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


# =========================
# 2. Transformer Encoder Block
# =========================
class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads=8, dim_ff=1024, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + self.dropout1(attn_out))

        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_out))
        return x


# =========================
# 3. 主模型（双分支）
# =========================
class modul(nn.Module):
    """
    输入: (B, 1, F, T)
    F = 8  -> 单分支
    F = 16 -> 原始 + 一阶差分双分支
    """
    def __init__(
        self,
        num_classes=13,
        feature_dim=16,
        time_steps=30,
        transformer_dropout=0.2,
        classifier_dropout=0.3,
        aux_num_classes=0,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.time_steps = time_steps
        self.transformer_dropout = transformer_dropout
        self.aux_num_classes = aux_num_classes

        # 原始分支
        self.conv1_a = nn.Conv2d(1, 16, kernel_size=5, padding=2)
        self.conv2_a = nn.Conv2d(16, 16, kernel_size=5, padding=2)
        self.se_a = SEBlock(16,reduction=4)

        # 动态分支
        self.conv1_b = nn.Conv2d(1, 16, kernel_size=5, padding=2)
        self.conv2_b = nn.Conv2d(16, 16, kernel_size=5, padding=2)
        self.se_b = SEBlock(16,reduction=4)

        if self.feature_dim < 16:
            d_model = self._calc_d_model(self.feature_dim)
            self.trans_a = self._build_transformer(d_model)
            self.trans_b = None
            self.fc = nn.Linear(d_model * self.time_steps, self.num_classes)
            aux_input_dim = d_model
        else:
            raw_feat = self.feature_dim // 2
            # dyn_feat = self.feature_dim - raw_feat
            d_model_raw = self._calc_d_model(raw_feat)
            
            # [修改 1] 只需要实例化一个 Transformer，废弃 trans_b
            self.trans_a = self._build_transformer(d_model_raw)
            self.trans_b = None 
            
            # [修改 2] 因为使用了特征相加，维度没有翻倍，所以全连接层的输入维度变为 d_model_raw
            self.fc = nn.Linear(
                d_model_raw * self.time_steps,
                self.num_classes
            )
            aux_input_dim = d_model_raw
        self.classifier_dropout = nn.Dropout(classifier_dropout)
        self.aux_head = (
            nn.Linear(aux_input_dim, self.aux_num_classes)
            if self.aux_num_classes > 0
            else None
        )
    def _calc_d_model(self, feat_dim):
        return 16 * feat_dim

    def _build_transformer(self, d_model):
        num_heads = 8
        return TransformerEncoderBlock(
            d_model,
            num_heads=num_heads,
            dim_ff=d_model * 2,
            dropout=self.transformer_dropout,
        )

    def _to_sequence(self, x):
        # (B, C, H, W) -> (B, W, C*H)
        b, c, h, w = x.shape
        x = x.permute(0, 3, 1, 2).contiguous()
        return x.view(b, w, c * h), c * h

    def _project_outputs(self, token_features):
        b = token_features.size(0)
        flat_features = self.classifier_dropout(token_features.reshape(b, -1))
        logits = self.fc(flat_features)

        aux_logits = None
        if self.aux_head is not None:
            pooled_features = self.classifier_dropout(token_features.mean(dim=1))
            aux_logits = self.aux_head(pooled_features)

        return logits, aux_logits

    def forward(self, x, return_aux=False):
        # x: (B, 1, F, T)
        b, _, feat_dim, T = x.shape

        if feat_dim != self.feature_dim:
            raise ValueError(
                f"Expected feature_dim={self.feature_dim}, but got {feat_dim}"
            )
        if T != self.time_steps:
            raise ValueError(
                f"Expected time_steps={self.time_steps}, but got {T}"
            )

        # ========= 单分支 =========
        if feat_dim < 16:
            a = F.relu(self.conv1_a(x))
            a = F.relu(self.conv2_a(a))
            a = self.se_a(a)

            a_seq, d_model = self._to_sequence(a)

            a_out = self.trans_a(a_seq)
            logits, aux_logits = self._project_outputs(a_out)
            if return_aux and aux_logits is not None:
                return logits, aux_logits
            return logits

        # ========= 双分支 (早融合版) =========
        half = feat_dim // 2
        x_raw = x[:, :, :half, :]
        x_dyn = x[:, :, half:, :]

        # 原始分支
        a = F.relu(self.conv1_a(x_raw))
        a = F.relu(self.conv2_a(a))
        a = self.se_a(a)

        # 动态分支
        d = F.relu(self.conv1_b(x_dyn))
        d = F.relu(self.conv2_b(d))
        d = self.se_b(d)

        a_seq, d_model_a = self._to_sequence(a)
        d_seq, d_model_d = self._to_sequence(d)

        # [核心修改]：特征相加，让静态和动态信息提前交互
        fused_seq = a_seq + d_seq
        
        # 仅送入单路 Transformer
        fused_out = self.trans_a(fused_seq)

        logits, aux_logits = self._project_outputs(fused_out)
        
        if return_aux and aux_logits is not None:
            return logits, aux_logits
        return logits