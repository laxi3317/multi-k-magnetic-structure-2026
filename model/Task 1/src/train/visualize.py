import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, roc_auc_score, recall_score, precision_score

def plot_roc_pr_curve(y_test, test_pred_dict, th_dict, cfg, save_path="propagation_classification_curves.png"):
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    color_map = cfg["colors"]
    ax_pr = axes[0]
    for name, (proba, _) in test_pred_dict.items():
        p, r, _ = precision_recall_curve(y_test, proba)
        auc_pr = float(np.trapz(p[::-1], r[::-1]))
        ax_pr.plot(r, p, label=f"{name} (AUC={auc_pr:.3f})", color=color_map[name], lw=2)
        th = th_dict[name]
        pred_op = (proba >= th).astype(int)
        rec_op = recall_score(y_test, pred_op, pos_label=1, zero_division=0)
        pre_op = precision_score(y_test, pred_op, pos_label=1, zero_division=0)
        ax_pr.scatter(rec_op, pre_op, color=color_map[name], s=100, zorder=5, edgecolors="black", linewidths=0.5)
    ax_pr.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
    ax_pr.set_xlabel("Recall (nonzero_k)", fontsize=12)
    ax_pr.set_ylabel("Precision (nonzero_k)", fontsize=12)
    ax_pr.set_title("PR Curve（Test Set — nonzero_k）", fontsize=13)
    ax_pr.legend(fontsize=9)
    ax_pr.grid(alpha=0.3)
    ax_pr.set_xlim([0,1])
    ax_pr.set_ylim([0,1])
    ax_roc = axes[1]
    for name, (proba, _) in test_pred_dict.items():
        auc_roc = roc_auc_score(y_test, proba)
        fpr, tpr, _ = roc_curve(y_test, proba)
        ax_roc.plot(fpr, tpr, label=f"{name} (AUC={auc_roc:.3f})", color=color_map[name], lw=2)
        th = th_dict[name]
        pred_op = (proba >= th).astype(int)
        tpr_op = recall_score(y_test, pred_op, pos_label=1, zero_division=0)
        fpr_op = 1.0 - recall_score(y_test, pred_op, pos_label=0, zero_division=0)
        ax_roc.scatter(fpr_op, tpr_op, color=color_map[name], s=100, zorder=5, edgecolors="black", linewidths=0.5)
    ax_roc.plot([0,1], [0,1], "k--", alpha=0.5, label="Random")
    ax_roc.set_xlabel("False Positive Rate", fontsize=12)
    ax_roc.set_ylabel("True Positive Rate", fontsize=12)
    ax_roc.set_title("ROC Curve（Test Set）", fontsize=13)
    ax_roc.legend(fontsize=9)
    ax_roc.grid(alpha=0.3)
    ax_roc.set_xlim([0,1])
    ax_roc.set_ylim([0,1])
    plt.suptitle("Binary Classifier — zero_k vs nonzero_k（Hold-out Test Set）", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    return save_path