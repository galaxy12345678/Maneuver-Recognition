import os
import sys
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F

matplotlib.use("Agg")

# --- 2. 全局 Matplotlib 样式美化 ---
plt.rcParams.update({
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "axes.spines.top": False,      # 移除顶部边框
    "axes.spines.right": False,    # 移除右侧边框
    "axes.grid": True,             # 开启网格
    "grid.alpha": 0.3,             # 网格透明度
    "grid.linestyle": "--",        # 网格虚线
    "axes.facecolor": "#fbfbfb",   # 极浅的灰色背景
    "figure.facecolor": "white",
})

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from Dataset_corrected import F16FlightDatasetCorrected
from cnn import modul

# ═══════════════════════════════════════════════════════════════
# 模型定义 (保持不变)
# ═══════════════════════════════════════════════════════════════
class _SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class _TransBlock(nn.Module):
    def __init__(self, d_model, num_heads=8, dim_ff=128, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.drop1 = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_ff), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x):
        out, _ = self.attn(x, x, x)
        x = self.norm1(x + self.drop1(out))
        x = self.norm2(x + self.drop2(self.ffn(x)))
        return x

class LegacyModul(nn.Module):
    def __init__(self, num_classes=13, feature_dim=16, time_steps=10,
                 transformer_dropout=0.35, classifier_dropout=0.5, aux_num_classes=3):
        super().__init__()
        self.feature_dim = feature_dim
        self.time_steps = time_steps

        self.conv1_a = nn.Conv2d(1, 32, 5, padding=2)
        self.conv2_a = nn.Conv2d(32, 64, 5, padding=2)
        self.se_a = _SEBlock(64, reduction=16)

        self.conv1_b = nn.Conv2d(1, 32, 5, padding=2)
        self.conv2_b = nn.Conv2d(32, 64, 5, padding=2)
        self.se_b = _SEBlock(64, reduction=16)

        d = 64 * (feature_dim // 2)
        self.trans_a = _TransBlock(d, num_heads=8, dim_ff=d // 4, dropout=transformer_dropout)
        self.trans_b = _TransBlock(d, num_heads=8, dim_ff=d // 4, dropout=transformer_dropout)

        self.drop = nn.Dropout(classifier_dropout)
        self.fc = nn.Linear(d * 2 * time_steps, num_classes)
        self.aux_head = nn.Linear(d * 2, aux_num_classes) if aux_num_classes > 0 else None

    @staticmethod
    def _seq(x):
        b, c, h, w = x.shape
        return x.permute(0, 3, 1, 2).contiguous().view(b, w, c * h)

    def forward(self, x, return_aux=False):
        half = self.feature_dim // 2
        a = self.se_a(F.relu(self.conv2_a(F.relu(self.conv1_a(x[:, :, :half, :])))))
        d = self.se_b(F.relu(self.conv2_b(F.relu(self.conv1_b(x[:, :, half:, :])))))
        fused = torch.cat([self.trans_a(self._seq(a)), self.trans_b(self._seq(d))], dim=-1)
        logits = self.fc(self.drop(fused.reshape(fused.size(0), -1)))
        aux = self.aux_head(self.drop(fused.mean(1))) if self.aux_head else None
        if return_aux and aux is not None:
            return logits, aux
        return logits


# ═══════════════════════════════════════════════════════════════
# 常量定义
# ═══════════════════════════════════════════════════════════════
FEATURE_NAMES = ["经度", "纬度", "高度", "滚转角", "俯仰角", "偏航角", "滚转率", "特征8"]
VERTICAL_CN = ["下降", "平飞", "上升"]
VERTICAL_EN = ["Down", "Level", "Up"]

CLASS_CN = {
    "Descent": "俯冲", "Level Flight": "平飞", "Roll Left": "左滚转",
    "Roll Right": "右滚转", "Turn Left": "左转弯", "Turn Left Descent": "左转俯冲",
    "Turn Left Up": "左转爬升", "Turn Right": "右转弯", "Turn Right Descent": "右转俯冲",
    "Turn Right Up": "右转爬升", "Up": "爬升", "Vertical Turn Descent": "垂直转俯冲",
    "Vertical Turn Up": "垂直转爬升",
}

VERTICAL_GROUP = {
    "Descent": "Down", "Level Flight": "Level", "Roll Left": "Level",
    "Roll Right": "Level", "Turn Left": "Level", "Turn Left Descent": "Down",
    "Turn Left Up": "Up", "Turn Right": "Level", "Turn Right Descent": "Down",
    "Turn Right Up": "Up", "Up": "Up", "Vertical Turn Descent": "Down",
    "Vertical Turn Up": "Up",
}
VERTICAL_CN_MAP = {"Down": "下降", "Level": "平飞", "Up": "上升"}
VERTICAL_ICON = {"Up": "↗️", "Level": "➡️", "Down": "↘️"}

MANEUVER_FILES = {
    "Up": "01up.txt", "Level Flight": "02Level Flight.txt", "Descent": "03Descent.txt",
    "Turn Right": "04Turn right.txt", "Turn Left": "05Turn left.txt",
    "Turn Right Up": "06Turn right up.txt", "Turn Right Descent": "07Turn right descent.txt",
    "Turn Left Up": "08Turn left up.txt", "Turn Left Descent": "09Turn left descent.txt",
    "Vertical Turn Up": "10Vertical turn up.txt", "Roll Right": "11Roll right.txt",
    "Roll Left": "12Roll left.txt", "Vertical Turn Descent": "13Vertical turn descetn.txt",
}
CLASS_NAMES = sorted(MANEUVER_FILES.keys())

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "flight_data")
MODEL_PATH = os.path.join(BASE_DIR, "model_weight", "best_model.pth")


# ═══════════════════════════════════════════════════════════════
# 资源加载
# ═══════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="⏳ 正在初始化高维空间与加载模型权重...")
def load_resources():
    ds = F16FlightDatasetCorrected(
        data_folder=DATA_DIR, time_steps=10, features_per_step=8,
        windows=[8, 10], window_strides={8: 2, 10: 1},
        add_delta=True, normalize=True, train_ratio=0.8,
    )
    ds.load_data()
    ds.preprocess_data(split_mode="grouped_random", random_state=42)

    m = modul(num_classes=13, feature_dim=16, time_steps=10,
              transformer_dropout=0.35, classifier_dropout=0.5, aux_num_classes=3)
    m.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    m.eval()
    return m, ds.scaler

@st.cache_data(show_spinner=False)
def load_sequences(name: str):
    path = os.path.join(DATA_DIR, MANEUVER_FILES[name])
    seqs, expected = [], 1 + 10 * 8
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            v = line.strip().split(",")
            if len(v) < expected: continue
            try:
                seqs.append(np.array(list(map(float, v[1:expected])), dtype=np.float32).reshape(10, 8))
            except ValueError: continue
    return seqs


def preprocess(seq, scaler):
    delta = np.vstack([np.zeros((1, 8), np.float32), np.diff(seq, axis=0)])
    norm = scaler.transform(np.concatenate([seq, delta], axis=1))
    return torch.FloatTensor(norm.T).unsqueeze(0).unsqueeze(0)

def infer_single(model, tensor):
    se = {}
    handles = [
        model.se_a.fc.register_forward_hook(lambda m, i, o: se.update({"raw": o.detach().cpu().numpy().flatten()})),
        model.se_b.fc.register_forward_hook(lambda m, i, o: se.update({"delta": o.detach().cpu().numpy().flatten()})),
    ]
    with torch.no_grad():
        logits, aux = model(tensor, return_aux=True)
    for h in handles: h.remove()
    return (torch.softmax(logits, 1).squeeze().numpy(), torch.softmax(aux, 1).squeeze().numpy(), se)

def infer_batch(model, seqs, scaler, batch_size=128, progress_cb=None):
    n, preds = len(seqs), []
    for i in range(0, n, batch_size):
        batch = torch.cat([preprocess(s, scaler) for s in seqs[i:i + batch_size]])
        with torch.no_grad():
            preds.extend(torch.argmax(model(batch), 1).tolist())
        if progress_cb: progress_cb(min(i + batch_size, n) / n)
    return np.array(preds)


# ═══════════════════════════════════════════════════════════════
# 绘图美化函数
# ═══════════════════════════════════════════════════════════════
COLORS8 = ["#4361EE", "#3A0CA3", "#7209B7", "#F72585", "#4CC9F0", "#00B4D8", "#F8961E", "#90BE6D"]

def fig_sensor(seq, title):
    fig, axes = plt.subplots(4, 2, figsize=(9, 7))
    for i, (ax, name, c) in enumerate(zip(axes.flat, FEATURE_NAMES, COLORS8)):
        # 绘制主线
        ax.plot(range(10), seq[:, i], "o-", color=c, markersize=5, linewidth=2)
        # 添加底部颜色填充，增加现代感
        ax.fill_between(range(10), seq[:, i], np.min(seq[:, i]) - (np.max(seq[:,i])-np.min(seq[:,i]))*0.1, color=c, alpha=0.1)
        ax.set_title(name, fontsize=10, fontweight="bold", pad=8)
        ax.set_xticks(range(10))
        ax.tick_params(axis='both', which='major', labelsize=8, colors='#555555')
    
    fig.suptitle(f"真实标签：{CLASS_CN[title]} ({title})", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    return fig

def fig_confidence(probs, true_class=None):
    cn = [CLASS_CN[c] for c in CLASS_NAMES]
    idx = np.argsort(probs)
    
    colors = []
    for i in idx:
        if true_class and CLASS_NAMES[i] == true_class:
            colors.append("#10B981") # 绿色-正确
        elif i == np.argmax(probs):
            colors.append("#EF4444") # 红色-错误预测最大
        else:
            colors.append("#D1D5DB") # 灰色-其他

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.barh([cn[i] for i in idx], probs[idx] * 100, color=colors, height=0.6, edgecolor='none')
    
    ax.set_xlabel("置信度 (%)", fontsize=10, color="#555")
    ax.set_xlim(0, 115)
    ax.tick_params(axis='y', labelsize=10, left=False)
    ax.xaxis.grid(True, linestyle='--', alpha=0.5)
    ax.yaxis.grid(False)

    for bar, val in zip(bars, probs[idx] * 100):
        if val > 0.1:
            ax.text(val + 2, bar.get_y() + bar.get_height() / 2, f"{val:.1f}%", 
                    va="center", fontsize=9, fontweight='bold', color="#333")
            
    plt.tight_layout()
    return fig

def fig_aux(aux_probs):
    colors = ["#3B82F6" if p == aux_probs.max() else "#E5E7EB" for p in aux_probs]
    fig, ax = plt.subplots(figsize=(5, 3))
    bars = ax.bar(VERTICAL_CN, aux_probs * 100, color=colors, width=0.5)
    ax.set_ylabel("置信度 (%)", fontsize=9)
    ax.set_ylim(0, 120)
    ax.xaxis.grid(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['bottom'].set_color('#ddd')
    
    for b, v in zip(bars, aux_probs * 100):
        ax.text(b.get_x() + b.get_width() / 2, v + 3, f"{v:.1f}%", 
                ha="center", fontsize=10, fontweight='bold', color="#444")
    plt.tight_layout()
    return fig

def fig_confusion(cm):
    norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
    labels = [CLASS_CN[c] for c in CLASS_NAMES]
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(norm, cmap="PuBu", vmin=0, vmax=1) # 更柔和的蓝紫色系
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(13))
    ax.set_yticks(range(13))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    for i in range(13):
        for j in range(13):
            v = norm[i, j]
            if v > 0.01: # 只显示>1%的值，保持画面整洁
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=8, color="white" if v > 0.6 else "#333333")
    ax.set_xlabel("预测类别", fontweight='bold')
    ax.set_ylabel("真实类别", fontweight='bold')
    plt.tight_layout()
    return fig

def render_progress_html(label, percentage, color):
    """自定义 HTML 渲染炫酷的横向进度条"""
    return f"""
    <div style="margin-bottom: 8px;">
        <div style="display: flex; justify-content: space-between; font-size: 13px; font-weight: 600; margin-bottom: 4px; color: #333;">
            <span>{label}</span>
            <span>{percentage:.1f}%</span>
        </div>
        <div style="width: 100%; background-color: #e5e7eb; border-radius: 4px; height: 8px; overflow: hidden;">
            <div style="width: {percentage}%; background-color: {color}; height: 100%; border-radius: 4px; transition: width 0.5s ease-in-out;"></div>
        </div>
    </div>
    """


# ═══════════════════════════════════════════════════════════════
# 页面配置与自定义 CSS
# ═══════════════════════════════════════════════════════════════
st.set_page_config(page_title="F-16 飞行动作识别", page_icon="✈️", layout="wide")

st.markdown("""
<style>
    /* 全局背景色微调 */
    .stApp { background-color: #f4f7f6; }
    
    /* 大标题样式 */
    h1 { color: #1e293b; font-weight: 800; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
    
    /* 指标卡片 (Metrics) 样式增强 */
    [data-testid="stMetricValue"] { color: #2563eb; font-weight: 800; font-size: 2rem;}
    [data-testid="stMetricLabel"] { font-size: 1rem; color: #64748b; font-weight: 600;}
    
    /* 容器加阴影与白底 */
    .css-1r6slb0, .css-12oz5g7 {
        background: white;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
        padding: 1.5rem;
    }
    
    /* 侧边栏样式 */
    [data-testid="stSidebar"] { background-color: #ffffff; border-right: 1px solid #e2e8f0; }
    
    /* Tabs 样式美化 */
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; background-color: transparent; border-radius: 4px 4px 0px 0px; gap: 1px; padding-top: 10px; padding-bottom: 10px; font-weight: 600; font-size: 16px;}
    .stTabs [aria-selected="true"] { color: #2563eb !important; border-bottom: 3px solid #2563eb !important; }
</style>
""", unsafe_allow_html=True)

st.title("✈️ F-16 机动动作智能识别系统")
st.markdown("<p style='color: #64748b; font-size: 16px; margin-top: -15px;'>基于双分支 CNN + SE 注意力机制 + Transformer | 13 类高机动动作精确识别</p>", unsafe_allow_html=True)

model, scaler = load_resources()

# ── 侧边栏：模型信息 ───────────────────────────────────────────
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3233/3233869.png", width=80)
    st.markdown("### 🧠 核心模型指标")
    c1, c2 = st.columns(2)
    c1.metric("Acc", "94.40%")
    c2.metric("M-F1", "94.20%")
    st.caption("参数量: **~163K** | 耗时: **< 10ms/条**")
    
    st.divider()
    
    st.markdown("### 🗂️ 动作图谱库 (13类)")
    for cls in CLASS_NAMES:
        grp = VERTICAL_GROUP[cls]
        st.markdown(f"<div style='padding: 4px 8px; background: #f8fafc; border-radius: 6px; margin-bottom: 4px; font-size: 14px;'>{VERTICAL_ICON[grp]} <b>{CLASS_CN[cls]}</b> <span style='color:#94a3b8; font-size: 12px;'>({cls})</span></div>", unsafe_allow_html=True)

# ── 主界面：三个 Tab ───────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🎯 单样本精细推理", "📊 批量数据评测", "🔍 易混淆类对比"])

# ════════════════════════════════════
# Tab 1：单样本推理
# ════════════════════════════════════
with tab1:
    col_ctrl, col_main = st.columns([1, 2.5], gap="large")

    with col_ctrl:
        with st.container(border=True):
            st.markdown("#### ⚙️ 数据抽取台")
            sel_cls = st.selectbox("选择目标真实机动类别", CLASS_NAMES, format_func=lambda x: f"{CLASS_CN[x]} ({x})")
            seqs1 = load_sequences(sel_cls)
            idx1 = st.slider("滑动抽取样本编号", 0, min(len(seqs1) - 1, 999), 0)
            st.caption(f"当前类别数据池：**{len(seqs1)}** 条")
            
            st.markdown("<br>", unsafe_allow_html=True)
            btn1 = st.button("🚀 启动时空推理特征捕获", type="primary", use_container_width=True)

    seq1 = seqs1[idx1]

    with col_main:
        if btn1:
            with st.spinner("🧠 Transformer 全局时序依赖建模中..."):
                probs, aux_probs, se = infer_single(model, preprocess(seq1, scaler))

            pred_cls = CLASS_NAMES[np.argmax(probs)]
            conf = probs.max() * 100
            correct = pred_cls == sel_cls

            # 顶部结果 Summary Card
            with st.container(border=True):
                r1, r2, r3 = st.columns([1, 1, 1.5])
                with r1:
                    st.metric("AI 预测决策", CLASS_CN[pred_cls])
                with r2:
                    st.metric("置信阈值", f"{conf:.2f}%")
                with r3:
                    if correct:
                        st.success("✅ **判定成功**：与地面真值 (Ground Truth) 完全吻合。")
                    else:
                        st.error(f"❌ **判定偏差**：发生了误判。\n\n实际应为：**{CLASS_CN[sel_cls]}**")
            
            # 详情面板
            detail_col1, detail_col2 = st.columns([1.5, 1], gap="medium")
            
            with detail_col1:
                st.markdown("##### 📈 飞行参数时序拓扑")
                st.pyplot(fig_sensor(seq1, sel_cls))
                plt.close("all")
                
            with detail_col2:
                st.markdown("##### 🎯 类别预测概率分布")
                st.pyplot(fig_confidence(probs, sel_cls))
                plt.close("all")
                
                st.markdown("##### ⚖️ 多任务辅助：垂直空间正则化")
                exp_v = VERTICAL_GROUP[sel_cls]
                pred_v = VERTICAL_EN[int(np.argmax(aux_probs))]
                st.caption(f"真值方向：**{VERTICAL_CN_MAP[exp_v]}** | 预测方向：**{VERTICAL_CN_MAP[pred_v]}**")
                st.pyplot(fig_aux(aux_probs))
                plt.close("all")
                
        else:
            # 未运行时的默认占位图
            st.info("👈 请在左侧选择数据并点击「启动推理」。下面是当前选中数据的原始传感器态势。")
            st.pyplot(fig_sensor(seq1, sel_cls))
            plt.close("all")


# ════════════════════════════════════
# Tab 2：批量测试
# ════════════════════════════════════
with tab2:
    with st.container(border=True):
        st.markdown("#### ⚙️ 压力测试场设置")
        b1, b2, b3 = st.columns([1, 1, 1])
        with b1:
            mode = st.radio("测试域选择", ["全局混测 (全部 13 类)", "特定科目 (指定类别)"])
        with b2:
            b_cls = st.selectbox("指定科目", CLASS_NAMES, format_func=lambda x: f"{CLASS_CN[x]}") if mode == "特定科目 (指定类别)" else None
            n_per = st.slider("单类别抽取量 (Sample Size)", 10, 300, 100, step=10)
        with b3:
            st.markdown("<br>", unsafe_allow_html=True)
            b_btn = st.button("▶ 执行高并发批量验证", type="primary", use_container_width=True)

    if b_btn:
        test_cls = CLASS_NAMES if mode == "全局混测 (全部 13 类)" else [b_cls]
        all_seqs, all_labels = [], []
        for cls in test_cls:
            s = load_sequences(cls)
            n = min(n_per, len(s))
            chosen = np.random.choice(len(s), n, replace=False)
            all_seqs += [s[i] for i in chosen]
            all_labels += [CLASS_NAMES.index(cls)] * n

        bar = st.progress(0, text="📡 数据流注入，张量计算中...")
        preds = infer_batch(model, all_seqs, scaler, progress_cb=lambda p: bar.progress(p, f"📡 进度: {p*100:.0f}%"))
        bar.empty()

        labels = np.array(all_labels)
        overall = (preds == labels).mean() * 100

        st.markdown("### 📋 评测报告")
        m1, m2, m3 = st.columns(3)
        m1.metric("综合准确率 (Overall Acc)", f"{overall:.2f}%")
        m2.metric("吞吐量 (Samples)", f"{len(all_seqs)} 帧")
        m3.metric("覆盖科目", f"{len(test_cls)} 类")

        if mode == "全局混测 (全部 13 类)":
            c_graph1, c_graph2 = st.columns(2)
            with c_graph1:
                st.markdown("##### 📍 各科目精准度 (Precision)")
                accs = [(preds[labels == i] == i).mean() * 100 for i in range(13)]
                
                # 画一个干净的柱状图
                fig, ax = plt.subplots(figsize=(6, 5))
                colors = ["#10B981" if a >= 95 else "#F59E0B" if a >= 85 else "#EF4444" for a in accs]
                ax.barh([CLASS_CN[c] for c in CLASS_NAMES], accs, color=colors)
                ax.set_xlim(0, 110)
                for i, v in enumerate(accs):
                    ax.text(v + 1, i, f"{v:.1f}%", va='center', fontsize=9, fontweight='bold')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close("all")

            with c_graph2:
                st.markdown("##### 🧩 决策归一化混淆矩阵")
                cm = np.zeros((13, 13), int)
                for t, p in zip(labels, preds): cm[t, p] += 1
                st.pyplot(fig_confusion(cm))
                plt.close("all")
        else:
            st.success(f"**{CLASS_CN[b_cls]}** 专项测试完成。识别率已稳定在 **{overall:.2f}%**。")


# ════════════════════════════════════
# Tab 3：多类对比
# ════════════════════════════════════
with tab3:
    st.markdown("### 🔍 跨科目边界对比分析")
    st.markdown("部分方向性复合动作（如：左转俯冲 vs 左转爬升）在时空特征上高度重叠，此处可直观对比模型在易混淆类别的差异化决策。")
    
    with st.container(border=True):
        default_cls = ["Turn Left Up", "Turn Left Descent"]
        sel_multi = st.multiselect(
            "加入对比观测池 (推荐提取具有空间相似性的科目)",
            CLASS_NAMES, default=default_cls, format_func=lambda x: f"{CLASS_CN[x]} ({x})"
        )
        c_btn = st.button("▶ 随机平行抽样对齐", type="primary")

    if c_btn:
        if not sel_multi:
            st.warning("⚠️ 请至少选择一个科目进行观测。")
        else:
            for cls in sel_multi:
                with st.container(border=True):
                    seqs_c = load_sequences(cls)
                    seq_c = seqs_c[np.random.randint(len(seqs_c))]
                    probs_c, _, _ = infer_single(model, preprocess(seq_c, scaler))
                    pred_c = CLASS_NAMES[np.argmax(probs_c)]
                    ok = pred_c == cls

                    ca, cb, cc = st.columns([1.5, 1.5, 1])
                    with ca:
                        st.markdown(f"**📡 地面真值：{CLASS_CN[cls]}**")
                        st.pyplot(fig_sensor(seq_c, cls))
                        plt.close("all")
                    with cb:
                        if ok:
                            st.markdown(f"**🤖 识别判定：<span style='color:#10B981;'>{CLASS_CN[pred_c]}</span>** (正确 ✅)", unsafe_allow_html=True)
                        else:
                            st.markdown(f"**🤖 识别判定：<span style='color:#EF4444;'>{CLASS_CN[pred_c]}</span>** (误判 ❌)", unsafe_allow_html=True)
                        st.markdown(f"<span style='font-size:14px; color:#64748b;'>主导置信度：</span> **{probs_c.max()*100:.1f}%**", unsafe_allow_html=True)
                        st.pyplot(fig_confidence(probs_c, cls))
                        plt.close("all")
                    with cc:
                        st.markdown("**🏆 Top-3 决策权重**")
                        st.markdown("<br>", unsafe_allow_html=True)
                        top_indices = np.argsort(probs_c)[::-1][:3]
                        colors_top3 = ["#3B82F6", "#9CA3AF", "#D1D5DB"]
                        for rank, i in enumerate(top_indices):
                            perc = probs_c[i]*100
                            label = f"{rank+1}. {CLASS_CN[CLASS_NAMES[i]]}"
                            st.markdown(render_progress_html(label, perc, colors_top3[rank]), unsafe_allow_html=True)
                            st.markdown(render_progress_html(label, perc, colors_top3[rank]), unsafe_allow_html=True)
                
                # 为每一个对比样例卡片底部添加优雅的分割线
                st.divider()
                