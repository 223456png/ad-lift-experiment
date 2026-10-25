"""
CATE 排序质量独立复核（确定性 fold，样本外）
=============================================
analysis.py 的 CATE 段是「五折轮转交叉拟合」；本脚本独立重算其中一对折
（fold 3 训练 → fold 4 评估），三重目的：

  1. 对账：同一对折上的 Qini 应与 analysis.py / cate_folds.json 逐位一致，
     两条独立代码路径算出同一个数，才叫可复现；
  2. 置换检验：在评估折上打乱 score 的 null 分布，给观测 Qini 一个经验 p；
  3. 校准复核：样本外「实际/预测」分位表，判断预测值能否当增量用。

折划分见 sql/00_samples.sql（md5 确定性分折——不再用 reservoir，
同一 seed 两次运行抽不出同一样本的历史问题就此终结）。

运行：python src/cate_check.py
输出：终端打印 + 与 report/cate_folds.json 的对账结果
"""
import json

import duckdb

from dbutil import connect as duck_connect
import numpy as np
import pandas as pd
from scipy import optimize, stats

FEATS = [f"f{i}" for i in range(12)]
TRAIN_FOLD, EVAL_FOLD = 3, 4


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -30, 30)))


def fit_logit(X, y, l2=1e-6, maxiter=400):
    """与 analysis.py 同款（L-BFGS + 解析梯度，否则 280 万行拟合要几分钟）"""
    Xd = np.column_stack([np.ones(len(X)), X])

    def nll(b):
        z = Xd @ b
        return np.sum(np.logaddexp(0, z) - y * z) + l2 * np.sum(b[1:] ** 2)

    def grad(b):
        z = Xd @ b
        g = Xd.T @ (sigmoid(z) - y)
        g[1:] += 2 * l2 * b[1:]
        return g

    return optimize.minimize(nll, np.zeros(Xd.shape[1]), jac=grad,
                             method="L-BFGS-B", options={"maxiter": maxiter}).x


def decile_stats(y, w, score, k=10):
    """按 score 降序切 k 箱，返回每箱 (预测均值, 实际增量, 样本量)。"""
    order = np.argsort(-score)
    y, w = y[order], w[order].astype(bool)
    s_sorted = score[order]
    preds, incs, sizes = [], [], []
    for b in np.array_split(np.arange(len(y)), k):
        yb, wb = y[b], w[b]
        if wb.sum() == 0 or (~wb).sum() == 0:
            continue
        preds.append(s_sorted[b].mean())
        incs.append(yb[wb].mean() - yb[~wb].mean())
        sizes.append(len(b))
    return np.array(preds), np.array(incs), np.array(sizes)


def qini_norm(y, w, score, k=10):
    """归一化 Qini = 2×曲线面积−1；只可在同一定义下比较。"""
    _, incs, sizes = decile_stats(y, w, score, k)
    cum = np.cumsum(incs * sizes)
    total = cum[-1] if abs(cum[-1]) > 1e-12 else 1e-12
    x = np.concatenate([[0.0], np.cumsum(sizes) / sizes.sum()])
    yv = np.concatenate([[0.0], cum / total])
    return 2 * float(np.trapezoid(yv, x)) - 1


def main():
    con = duck_connect()
    for f in ["sql/01_build.sql", "sql/00_samples.sql"]:
        con.execute(open(f, encoding="utf-8").read())

    def load_fold(i):
        d = con.execute(f"SELECT {','.join(FEATS)}, treatment, conversion "
                        f"FROM det WHERE fold = {i}").df()
        return (d[FEATS].values.astype(float),
                d.conversion.values.astype(float),
                d.treatment.values.astype(int))

    Xtr, ytr, wtr = load_fold(TRAIN_FOLD)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    b1 = fit_logit(((Xtr - mu) / sd)[wtr == 1], ytr[wtr == 1])
    b0 = fit_logit(((Xtr - mu) / sd)[wtr == 0], ytr[wtr == 0])

    Xte, yte, wte = load_fold(EVAL_FOLD)
    Xd = np.column_stack([np.ones(len(Xte)), (Xte - mu) / sd])
    s = sigmoid(Xd @ b1) - sigmoid(Xd @ b0)

    q_obs = qini_norm(yte, wte, s)
    print(f"独立复核：fold {TRAIN_FOLD} 训练 → fold {EVAL_FOLD} 评估（确定性折）")
    print(f"  观测 Qini（归一化） = {q_obs:+.4f}   评估折 ATE = "
          f"{yte[wte == 1].mean() - yte[wte == 0].mean():+.6f}   n = {len(yte):,}")

    # ---------- 对账：与 analysis.py 的同一折结果逐位比对 ----------
    try:
        with open("report/cate_folds.json", encoding="utf-8") as fh:
            ref = json.load(fh)
        ref_q = ref["fold_qini"][TRAIN_FOLD]
        ok = abs(ref_q - q_obs) < 5e-4
        print(f"  对账：cate_folds.json 里 fold {TRAIN_FOLD} 的 Qini = {ref_q:+.4f} "
              f"-> {'一致 ✓' if ok else '不一致 ✗（重跑 analysis.py 后再比对）'}")
    except FileNotFoundError:
        print("  对账：report/cate_folds.json 不存在，先跑 analysis.py 再回来对")

    # ---------- 分位校准（样本外） ----------
    preds, incs, sizes = decile_stats(yte, wte, s)
    cum = np.cumsum(incs * sizes)
    total = cum[-1]
    print(f"\n  {'分位':<8}{'样本':>10}{'预测增量':>13}{'实际增量':>13}{'实际/预测':>12}{'累计占比':>12}")
    cumshare = 0.0
    for i, (p_, inc, sz) in enumerate(zip(preds, incs, sizes)):
        cumshare += inc * sz / total
        ratio = inc / p_ if abs(p_) > 1e-9 else np.nan
        print(f"  前{(i+1)*10:>3}%  {sz:>10,}{p_:>13.6f}{inc:>13.6f}{ratio:>12.2f}{cumshare:>12.1%}")

    # ---------- 置换 null：打乱评估折 score ----------
    # 注意打乱的是 score 与 (y,w) 的对应关系，等价于「随机排序」的原假设
    rng = np.random.default_rng(42)
    nulls = np.array([qini_norm(yte, wte, rng.permutation(s)) for _ in range(100)])
    p = float((nulls >= q_obs).mean())
    print(f"\n  置换 null（B=100，打乱评估折排序）：均值 {nulls.mean():+.4f}  "
          f"95% 分位 {np.percentile(nulls, 95):+.4f}")
    print(f"  观测 {q_obs:+.4f} -> 经验 p = {p:.3f}"
          f"（100 次置换最小只能到 {1/101:.3f}，报告写 p<{max(p, 1/101):.3f}）")

    # ---------- Spearman：分段预测均值 vs 分段实际增量（100 段更细） ----------
    _, incs100, _ = decile_stats(yte, wte, s, k=100)
    preds100, _, _ = decile_stats(yte, wte, s, k=100)
    rho = stats.spearmanr(preds100, incs100)
    print(f"\n  Spearman（100 段预测均值 vs 实际增量）= {rho.statistic:+.3f}（p={rho.pvalue:.2g}）")
    print("  判读：")
    print("    · 显著为正   -> 排序携带信息，可作为定向的辅助证据")
    print("    · 不显著     -> Qini 再高也只是运气，不能据此裁剪人群")
    print("    · 校准比值远离 1 -> 预测值不能直接当增量绝对量用")


if __name__ == "__main__":
    main()
