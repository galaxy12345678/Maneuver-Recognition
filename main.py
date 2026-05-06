import math
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim

try:
    from sklearn.metrics import classification_report
except ImportError:
    classification_report = None

from Dataset_corrected import F16FlightDatasetCorrected
from cnn import modul


VERTICAL_CLASS_NAMES = ["Down", "Level", "Up"]


def build_vertical_target_lookup(class_names):
    vertical_group_by_class = {
        "Descent": "Down",
        "Level Flight": "Level",
        "Roll Left": "Level",
        "Roll Right": "Level",
        "Turn Left": "Level",
        "Turn Left Descent": "Down",
        "Turn Left Up": "Up",
        "Turn Right": "Level",
        "Turn Right Descent": "Down",
        "Turn Right Up": "Up",
        "Up": "Up",
        "Vertical Turn Descent": "Down",
        "Vertical Turn Up": "Up",
    }
    vertical_name_to_idx = {
        name: idx for idx, name in enumerate(VERTICAL_CLASS_NAMES)
    }
    lookup = torch.empty(len(class_names), dtype=torch.long)

    for idx, class_name in enumerate(class_names):
        if class_name not in vertical_group_by_class:
            raise KeyError(f"Missing vertical-group mapping for class: {class_name}")
        lookup[idx] = vertical_name_to_idx[vertical_group_by_class[class_name]]

    return lookup


def build_confusion_matrix(labels, preds, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true_label, pred_label in zip(labels, preds):
        cm[int(true_label), int(pred_label)] += 1
    return cm


def fuse_main_and_vertical_logits(
    main_logits,
    vertical_logits,
    class_to_vertical_lookup,
    fusion_weight,
):
    vertical_bias = vertical_logits[:, class_to_vertical_lookup]
    return main_logits + fusion_weight * vertical_bias


def compute_metrics_from_confusion_matrix(cm):
    support = cm.sum(axis=1).astype(np.float64)
    pred_count = cm.sum(axis=0).astype(np.float64)
    true_positive = np.diag(cm).astype(np.float64)

    precision = np.divide(
        true_positive,
        pred_count,
        out=np.zeros_like(true_positive),
        where=pred_count > 0,
    )
    recall = np.divide(
        true_positive,
        support,
        out=np.zeros_like(true_positive),
        where=support > 0,
    )
    precision_plus_recall = precision + recall
    f1 = np.divide(
        2.0 * precision * recall,
        precision_plus_recall,
        out=np.zeros_like(true_positive),
        where=precision_plus_recall > 0,
    )

    total = cm.sum()
    weighted_support = support / total if total > 0 else np.zeros_like(support)

    return {
        "accuracy": 100.0 * true_positive.sum() / total if total > 0 else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support.astype(np.int64),
        "macro_f1": float(f1.mean()) if len(f1) else 0.0,
        "weighted_f1": float(np.sum(f1 * weighted_support)) if total > 0 else 0.0,
    }


def evaluate(model, data_loader, device, num_classes, vertical_target_lookup=None):
    model.eval()
    all_preds, all_labels = [], []
    vertical_correct = 0
    vertical_total = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            if vertical_target_lookup is not None:
                outputs, vertical_outputs = model(images, return_aux=True)
                outputs = fuse_main_and_vertical_logits(
                    outputs,
                    vertical_outputs,
                    vertical_target_lookup,
                    fusion_weight=logit_fusion_weight,
                )
                vertical_labels = vertical_target_lookup[labels]
                vertical_preds = torch.argmax(vertical_outputs, dim=1)
                vertical_total += labels.size(0)
                vertical_correct += (vertical_preds == vertical_labels).sum().item()
            else:
                outputs = model(images)

            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    cm = build_confusion_matrix(all_labels, all_preds, num_classes=num_classes)
    metrics = compute_metrics_from_confusion_matrix(cm)
    metrics["confusion_matrix"] = cm
    metrics["preds"] = all_preds
    metrics["labels"] = all_labels
    metrics["vertical_acc"] = (
        100.0 * vertical_correct / vertical_total if vertical_total > 0 else None
    )
    return metrics


def ema_smooth(values, alpha=0.6):
    if not values:
        return []

    smoothed = [values[0]]
    for value in values[1:]:
        smoothed.append(alpha * value + (1 - alpha) * smoothed[-1])
    return smoothed


def safe_torch_load(weights_path, device):
    try:
        return torch.load(weights_path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(weights_path, map_location=device)


def print_fallback_classification_report(class_names, metrics):
    print(f"{'Class':>24} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    for idx, name in enumerate(class_names):
        print(
            f"{name:>24} "
            f"{metrics['precision'][idx]:>10.4f} "
            f"{metrics['recall'][idx]:>10.4f} "
            f"{metrics['f1'][idx]:>10.4f} "
            f"{metrics['support'][idx]:>10d}"
        )


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

hard_config = {
    "windows": [8, 10],
    "window_strides": {8: 2, 10: 1},
    "split_mode": "contiguous",
    "purge_gap": 0,
    "train_ratio": 0.8,
    "eval_perturbation": {
        "noise_sigma": 0.05,
        "missing_prob": 0.02,
        "time_shift_range": 1,
    },
}
print("Experiment profile: hard")

dataset_manager = F16FlightDatasetCorrected(
    data_folder="flight_data",
    time_steps=10,
    features_per_step=8,
    windows=hard_config["windows"],
    add_delta=True,
    train_ratio=hard_config["train_ratio"],
    window_strides=hard_config["window_strides"],
    eval_perturbation=hard_config["eval_perturbation"],
)

dataset_manager.load_data()
dataset_manager.preprocess_data(
    split_mode=hard_config["split_mode"],
    purge_gap=hard_config["purge_gap"],
)

class_names = dataset_manager.class_names
num_classes = len(class_names)
vertical_target_lookup = build_vertical_target_lookup(class_names).to(device)
focus_class_name = "Turn Left Descent"
confuser_class_name = "Turn Left Up"
focus_class_idx = class_names.index(focus_class_name)
confuser_class_idx = class_names.index(confuser_class_name)

print(
    f"Focus setup | target={focus_class_name} (idx={focus_class_idx}) | "
    f"confuser={confuser_class_name} (idx={confuser_class_idx})"
)
print(
    "Auxiliary task | vertical classes: "
    + ", ".join(VERTICAL_CLASS_NAMES)
)

batch_size = 32
train_loader, test_loader = dataset_manager.get_dataloader(
    batch_size=batch_size,
    augment=True,
    noise_sigma=0.02,
    missing_prob=0.01,
    time_shift_range=1,
    scale_sigma=0.03,
    feature_dropout_prob=0.02,
    use_weighted_sampler=True,
)

model = modul(
    num_classes=num_classes,
    feature_dim=dataset_manager.features_per_step,
    time_steps=dataset_manager.time_steps,
    transformer_dropout=0.35,
    classifier_dropout=0.5,
    aux_num_classes=len(VERTICAL_CLASS_NAMES),
).to(device)

# 对 Level Flight 和 Descent 加权，缓解混淆
class_weights = torch.ones(num_classes, device=device)
for name, w in [("Level Flight", 2.0), ("Descent", 1.5),
                ("Turn Left Descent", 2.5), ("Turn Left Up", 2.5)]:
    if name in class_names:
        class_weights[class_names.index(name)] = w
main_criterion = nn.CrossEntropyLoss(label_smoothing=0.05, weight=class_weights)
aux_criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
aux_loss_weight = 0.15
logit_fusion_weight = 0.15
learning_rate = 1e-4
weight_decay = 5e-3
grad_clip_max_norm = 1.0
optimizer = optim.AdamW(
    model.parameters(),
    lr=learning_rate,
    weight_decay=weight_decay,
)

epochs = 50
min_epochs = 20
early_stopping_patience = 12
save_dir = "./model_weight"
best_model_path = os.path.join(save_dir, "best_model.pth")
os.makedirs(save_dir, exist_ok=True)

train_batch_fraction = 1.0
max_train_batches_per_epoch = max(1, math.ceil(len(train_loader) * train_batch_fraction))
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=epochs,
    eta_min=1e-5,
)

print(
    f"Training batches per epoch: {max_train_batches_per_epoch}/{len(train_loader)} "
    f"({train_batch_fraction:.0%})"
)
print(
    f"Loss setup | fused_main_ce + {aux_loss_weight:.2f} * vertical_ce | "
    f"logit_fusion_weight={logit_fusion_weight:.2f}"
)

best_acc = float("-inf")
best_selection_score = float("-inf")
epochs_without_improvement = 0
train_total_loss_list = []
train_main_loss_list = []
train_aux_loss_list = []
test_acc_list = []
macro_f1_list = []
focus_recall_list = []
vertical_acc_list = []
lr_list = []

for epoch in range(epochs):
    model.train()
    running_total_loss = 0.0
    running_main_loss = 0.0
    running_aux_loss = 0.0
    train_batches_used = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        vertical_labels = vertical_target_lookup[labels]

        outputs, vertical_outputs = model(images, return_aux=True)
        fused_outputs = fuse_main_and_vertical_logits(
            outputs,
            vertical_outputs,
            vertical_target_lookup,
            fusion_weight=logit_fusion_weight,
        )
        main_loss = main_criterion(fused_outputs, labels)
        aux_loss = aux_criterion(vertical_outputs, vertical_labels)
        loss = main_loss + aux_loss_weight * aux_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_max_norm)
        optimizer.step()

        running_total_loss += loss.item()
        running_main_loss += main_loss.item()
        running_aux_loss += aux_loss.item()
        train_batches_used += 1

        if train_batches_used >= max_train_batches_per_epoch:
            break

    avg_total_loss = running_total_loss / train_batches_used
    avg_main_loss = running_main_loss / train_batches_used
    avg_aux_loss = running_aux_loss / train_batches_used
    train_total_loss_list.append(avg_total_loss)
    train_main_loss_list.append(avg_main_loss)
    train_aux_loss_list.append(avg_aux_loss)

    metrics = evaluate(
        model,
        test_loader,
        device,
        num_classes,
        vertical_target_lookup=vertical_target_lookup,
    )
    acc = metrics["accuracy"]
    macro_f1 = metrics["macro_f1"]
    focus_recall = metrics["recall"][focus_class_idx]
    focus_f1 = metrics["f1"][focus_class_idx]
    vertical_acc = metrics["vertical_acc"] or 0.0
    selection_score = acc

    test_acc_list.append(acc)
    macro_f1_list.append(macro_f1 * 100.0)
    focus_recall_list.append(focus_recall * 100.0)
    vertical_acc_list.append(vertical_acc)
    lr_list.append(optimizer.param_groups[0]["lr"])

    print(
        f"Epoch [{epoch + 1}/{epochs}] | "
        f"Updates: {train_batches_used} | "
        f"LR: {optimizer.param_groups[0]['lr']:.6f} | "
        f"Loss: {avg_total_loss:.4f} | "
        f"Main CE: {avg_main_loss:.4f} | "
        f"Vertical CE: {avg_aux_loss:.4f} | "
        f"Val Acc: {acc:.2f}% | "
        f"Macro F1: {macro_f1 * 100.0:.2f}% | "
        f"Vert Acc: {vertical_acc:.2f}% | "
        f"{focus_class_name} Recall: {focus_recall * 100.0:.2f}% | "
        f"{focus_class_name} F1: {focus_f1 * 100.0:.2f}% | "
        f"Selection: {selection_score:.4f}"
    )

    improved = selection_score > best_selection_score + 1e-6
    tie_with_better_acc = (
        abs(selection_score - best_selection_score) <= 1e-6 and acc > best_acc
    )
    if improved or tie_with_better_acc:
        best_selection_score = selection_score
        best_acc = acc
        torch.save(model.state_dict(), best_model_path)
        print(
            f"Best model updated: score={best_selection_score:.4f}, "
            f"val_acc={best_acc:.2f}%"
        )
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1

    if (epoch + 1) % 25 == 0:
        torch.save(model.state_dict(), os.path.join(save_dir, f"epoch_{epoch + 1}.pth"))

    scheduler.step()

    if epoch + 1 >= min_epochs and epochs_without_improvement >= early_stopping_patience:
        print(
            f"Early stopping triggered at epoch {epoch + 1}. "
            f"No selection-score improvement for {early_stopping_patience} epochs."
        )
        break

print(
    f"\nTraining complete | best selection score: {best_selection_score:.4f} | "
    f"best validation accuracy: {best_acc:.2f}%"
)

if os.path.exists(best_model_path):
    model.load_state_dict(safe_torch_load(best_model_path, device))
    print(f"Loaded best checkpoint: {best_model_path}")
else:
    print(f"[WARN] Best checkpoint not found, using current model: {best_model_path}")

final_metrics = evaluate(
    model,
    test_loader,
    device,
    num_classes,
    vertical_target_lookup=vertical_target_lookup,
)
final_acc = final_metrics["accuracy"]
all_preds = final_metrics["preds"]
all_labels = final_metrics["labels"]
cm = final_metrics["confusion_matrix"]

print(f"Best-checkpoint validation accuracy: {final_acc:.2f}%")
print(
    f"Vertical-trend auxiliary accuracy: {final_metrics['vertical_acc']:.2f}%"
)
print(
    f"Focused metrics | {focus_class_name} Precision: "
    f"{final_metrics['precision'][focus_class_idx]:.4f} | "
    f"Recall: {final_metrics['recall'][focus_class_idx]:.4f} | "
    f"F1: {final_metrics['f1'][focus_class_idx]:.4f}"
)
print(
    f"Focused confusion | {focus_class_name} -> {confuser_class_name}: "
    f"{int(cm[focus_class_idx, confuser_class_idx])}"
)

# =====================================================================
# 论文可视化 — 所有图保存到 image/ 文件夹
# =====================================================================
from matplotlib.ticker import MaxNLocator
from sklearn.manifold import TSNE

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})

img_dir = "./image"
os.makedirs(img_dir, exist_ok=True)
epoch_range = list(range(1, len(train_total_loss_list) + 1))

# ---- 1. 训练损失曲线 ----
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(epoch_range, train_total_loss_list, "o-", markersize=3, label="Total Loss")
ax.plot(epoch_range, train_main_loss_list, "s--", markersize=3, alpha=0.8, label="Main CE Loss")
ax.plot(epoch_range, train_aux_loss_list, "^--", markersize=3, alpha=0.8, label="Auxiliary CE Loss")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.set_title("Training Loss Curves")
ax.legend()
ax.grid(True, alpha=0.3)
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
fig.savefig(os.path.join(img_dir, "train_loss.png"))
plt.close(fig)

# ---- 2. 验证准确率 & Macro-F1 曲线 ----
fig, ax = plt.subplots(figsize=(8, 5))
val_acc_smoothed = ema_smooth(test_acc_list, alpha=0.6)
ax.plot(epoch_range, test_acc_list, alpha=0.3, color="tab:blue", label="Val Accuracy (raw)")
ax.plot(epoch_range, val_acc_smoothed, linewidth=2, color="tab:blue", label="Val Accuracy (smoothed)")
ax.plot(epoch_range, macro_f1_list, "s-", markersize=3, color="tab:green", label="Macro F1 (%)")
ax.set_xlabel("Epoch")
ax.set_ylabel("Score (%)")
ax.set_title("Validation Accuracy & Macro-F1")
ax.legend()
ax.grid(True, alpha=0.3)
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
fig.savefig(os.path.join(img_dir, "val_acc_f1.png"))
plt.close(fig)

# ---- 3. 焦点类召回率 & 辅助任务准确率曲线 ----
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(epoch_range, focus_recall_list, "D-", markersize=3, color="tab:red", label=f"{focus_class_name} Recall (%)")
ax.plot(epoch_range, vertical_acc_list, "^-", markersize=3, color="tab:purple", label="Vertical Aux Accuracy (%)")
ax.set_xlabel("Epoch")
ax.set_ylabel("Score (%)")
ax.set_title("Focus-Class Recall & Auxiliary Task Accuracy")
ax.legend()
ax.grid(True, alpha=0.3)
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
fig.savefig(os.path.join(img_dir, "focus_recall_vertical.png"))
plt.close(fig)

# ---- 4. 学习率调度曲线 ----
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(epoch_range, lr_list, "o-", markersize=3, color="tab:orange")
ax.set_xlabel("Epoch")
ax.set_ylabel("Learning Rate")
ax.set_title("Cosine Annealing Learning Rate Schedule")
ax.grid(True, alpha=0.3)
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
fig.savefig(os.path.join(img_dir, "lr_schedule.png"))
plt.close(fig)

# ---- 5. 混淆矩阵（原始计数） ----
fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names, ax=ax)
ax.set_xlabel("Predicted Label")
ax.set_ylabel("True Label")
ax.set_title("Confusion Matrix (Counts)")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
fig.savefig(os.path.join(img_dir, "confusion_matrix.png"))
plt.close(fig)

# ---- 6. 混淆矩阵（归一化，按行百分比） ----
cm_norm = cm.astype(np.float64)
row_sums = cm_norm.sum(axis=1, keepdims=True)
row_sums[row_sums == 0] = 1
cm_norm = cm_norm / row_sums * 100

fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(cm_norm, annot=True, fmt=".1f", cmap="YlOrRd",
            xticklabels=class_names, yticklabels=class_names, ax=ax,
            vmin=0, vmax=100)
ax.set_xlabel("Predicted Label")
ax.set_ylabel("True Label")
ax.set_title("Normalized Confusion Matrix (%)")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
fig.savefig(os.path.join(img_dir, "confusion_matrix_normalized.png"))
plt.close(fig)

# ---- 7. 各类别 Precision / Recall / F1 柱状图 ----
x_pos = np.arange(num_classes)
bar_w = 0.25
fig, ax = plt.subplots(figsize=(14, 6))
ax.bar(x_pos - bar_w, final_metrics["precision"], bar_w, label="Precision", color="tab:blue")
ax.bar(x_pos, final_metrics["recall"], bar_w, label="Recall", color="tab:orange")
ax.bar(x_pos + bar_w, final_metrics["f1"], bar_w, label="F1-Score", color="tab:green")
ax.set_xticks(x_pos)
ax.set_xticklabels(class_names, rotation=45, ha="right")
ax.set_ylabel("Score")
ax.set_title("Per-Class Precision, Recall, and F1-Score")
ax.set_ylim(0, 1.08)
ax.legend()
ax.grid(axis="y", alpha=0.3)
for i in range(num_classes):
    ax.text(i + bar_w, final_metrics["f1"][i] + 0.01,
            f"{final_metrics['f1'][i]:.2f}", ha="center", va="bottom", fontsize=7)
fig.savefig(os.path.join(img_dir, "per_class_metrics.png"))
plt.close(fig)

# ---- 8. 训练/测试集各类别样本分布 ----
train_labels_np = dataset_manager.processed_data["y_train"]
test_labels_np = dataset_manager.processed_data["y_test"]
train_counts = np.bincount(train_labels_np, minlength=num_classes)
test_counts = np.bincount(test_labels_np, minlength=num_classes)

fig, ax = plt.subplots(figsize=(14, 6))
ax.bar(x_pos - 0.2, train_counts, 0.4, label="Train", color="tab:blue")
ax.bar(x_pos + 0.2, test_counts, 0.4, label="Test", color="tab:orange")
ax.set_xticks(x_pos)
ax.set_xticklabels(class_names, rotation=45, ha="right")
ax.set_ylabel("Sample Count")
ax.set_title("Class Distribution (Train / Test)")
ax.legend()
ax.grid(axis="y", alpha=0.3)
for i in range(num_classes):
    ax.text(i - 0.2, train_counts[i] + 20, str(train_counts[i]),
            ha="center", va="bottom", fontsize=7)
    ax.text(i + 0.2, test_counts[i] + 20, str(test_counts[i]),
            ha="center", va="bottom", fontsize=7)
fig.savefig(os.path.join(img_dir, "class_distribution.png"))
plt.close(fig)

# ---- 9. 焦点类混淆细节（Turn Left Descent 误分类去向） ----
focus_row = cm[focus_class_idx]
other_indices = [i for i in range(num_classes) if i != focus_class_idx and focus_row[i] > 0]
if other_indices:
    fig, ax = plt.subplots(figsize=(8, 5))
    misclass_names = [class_names[i] for i in other_indices]
    misclass_counts = [focus_row[i] for i in other_indices]
    sorted_pairs = sorted(zip(misclass_counts, misclass_names), reverse=True)
    misclass_counts, misclass_names = zip(*sorted_pairs)
    bars = ax.barh(range(len(misclass_names)), misclass_counts, color="tab:red", alpha=0.8)
    ax.set_yticks(range(len(misclass_names)))
    ax.set_yticklabels(misclass_names)
    ax.set_xlabel("Misclassified Count")
    ax.set_title(f"Misclassification Breakdown: {focus_class_name}")
    ax.invert_yaxis()
    for bar, count in zip(bars, misclass_counts):
        ax.text(bar.get_width() + 2, bar.get_y() + bar.get_height() / 2,
                str(count), va="center", fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    fig.savefig(os.path.join(img_dir, "focus_misclassification.png"))
    plt.close(fig)

# ---- 10. t-SNE 特征可视化 ----
print("Generating t-SNE visualization...")
model.eval()
features_list, labels_list = [], []
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        b, _, feat_dim_in, T_in = images.shape
        if feat_dim_in >= 16:
            half = feat_dim_in // 2
            x_raw = images[:, :, :half, :]
            x_dyn = images[:, :, half:, :]
            a = torch.relu(model.conv2_a(torch.relu(model.conv1_a(x_raw))))
            a = model.se_a(a)
            d = torch.relu(model.conv2_b(torch.relu(model.conv1_b(x_dyn))))
            d = model.se_b(d)
            a_seq, _ = model._to_sequence(a)
            d_seq, _ = model._to_sequence(d)
            fused_seq = a_seq + d_seq
            fused = model.trans_a(fused_seq)
        else:
            a = torch.relu(model.conv2_a(torch.relu(model.conv1_a(images))))
            a = model.se_a(a)
            a_seq, _ = model._to_sequence(a)
            fused = model.trans_a(a_seq)
        pooled = fused.mean(dim=1)
        features_list.append(pooled.cpu().numpy())
        labels_list.append(labels.numpy())

all_features = np.concatenate(features_list)
all_tsne_labels = np.concatenate(labels_list)

tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter_without_progress=1000)
embeddings = tsne.fit_transform(all_features)

fig, ax = plt.subplots(figsize=(12, 10))
cmap = plt.cm.get_cmap("tab20", num_classes)
for cls_idx in range(num_classes):
    mask = all_tsne_labels == cls_idx
    ax.scatter(embeddings[mask, 0], embeddings[mask, 1],
               s=8, alpha=0.6, color=cmap(cls_idx), label=class_names[cls_idx])
ax.set_title("t-SNE Visualization of Learned Features")
ax.set_xlabel("t-SNE Dimension 1")
ax.set_ylabel("t-SNE Dimension 2")
ax.legend(markerscale=3, loc="best", fontsize=8)
ax.grid(True, alpha=0.2)
fig.savefig(os.path.join(img_dir, "tsne_features.png"))
plt.close(fig)

# ---- 11. SE注意力权重热力图 ----
print("Generating SE attention visualization...")
model.eval()
se_weights_raw, se_weights_dyn = [], []
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        b, _, fd, _ = images.shape
        if fd >= 16:
            half = fd // 2
            a = torch.relu(model.conv2_a(torch.relu(model.conv1_a(images[:, :, :half, :]))))
            d = torch.relu(model.conv2_b(torch.relu(model.conv1_b(images[:, :, half:, :]))))
            w_a = model.se_a.pool(a).view(b, -1)
            w_a = model.se_a.fc(w_a)
            w_d = model.se_b.pool(d).view(b, -1)
            w_d = model.se_b.fc(w_d)
            for i in range(b):
                lbl = labels[i].item()
                se_weights_raw.append((lbl, w_a[i].cpu().numpy()))
                se_weights_dyn.append((lbl, w_d[i].cpu().numpy()))
        break  # 一个batch足够

for branch_name, se_data in [("Raw Branch", se_weights_raw), ("Delta Branch", se_weights_dyn)]:
    per_class = {}
    for lbl, w in se_data:
        per_class.setdefault(lbl, []).append(w)
    present_classes = sorted(per_class.keys())
    heat = np.array([np.mean(per_class[c], axis=0) for c in present_classes])
    fig, ax = plt.subplots(figsize=(14, max(4, len(present_classes) * 0.5)))
    sns.heatmap(heat, cmap="viridis", ax=ax,
                yticklabels=[class_names[c] for c in present_classes],
                xticklabels=[f"Ch{i}" for i in range(heat.shape[1])])
    ax.set_title(f"SE Attention Weights — {branch_name}")
    ax.set_xlabel("Channel")
    ax.set_ylabel("Class")
    fname = "se_attention_raw.png" if "Raw" in branch_name else "se_attention_delta.png"
    fig.savefig(os.path.join(img_dir, fname))
    plt.close(fig)

print(f"\nAll figures saved to {img_dir}/")

# ---- 分类报告 ----
print("\nClassification Report:")
if classification_report is not None:
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))
else:
    print_fallback_classification_report(class_names, final_metrics)
