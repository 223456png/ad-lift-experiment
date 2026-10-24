"""
CATE 排序质量复核
=================
analysis.py 输出的 Qini=样本外 0.0465 与"分位实际增量非单调"矛盾，本脚本回答：
   1. 排序是否真的有效（Spearman 秩相关 + 随机重排 null 分布）
   2. 用更稳的模型（梯度提升，若可用）是否改善
   3. 结论是否足够支持"定向投放"

运行：python src/cate_check.py
"""
import duckdb
import numpy as np
import pandas as pd
from scipy import stats, optimize

FEATS = [f"f{i}" for i in range(12)]


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -30, 30)))


def fit_logit(X, y, l2=1e-6, maxiter=400):
    Xd = np.column_stack([np.ones(len(X)), X])
    def nll(b):
        z = Xd @ b
        return np.sum(np.logaddexp(0, z) - y * z) + l2 * np.sum(b[1:] ** 2)
    return optimize.minimize(nll, np.zeros(Xd.shape[1]), method="L-BFGS-B",
                             options={"maxiter": maxiter}).x


def decile_uplift(y, w, score, k=10):
    """按 score 降序切 k 箱，返回每箱的 (预测均值, 实际增量, 样本量)"""
    order = np.argsort(-score)
    y, w, s = y[order], w[order].astype(bool), score[order]
    preds, incs, sizes = [], [], []
    for b in np.array_split(np.arange(len(y)), k):
        yb, wb, sb = y[b], w[b], s[b]
        if wb.sum() == 0 or (~wb).sum() == 0:
            continue
        preds.append(sb.mean())
        incs.append(yb[wb].mean() - yb[~wb].mean())
        sizes.append(len(b))
    return np.array(preds), np.array(incs), np.array(sizes)


def qini(y, w, score, k=10):
    preds, incs, sizes = decile_uplift(y, w, score, k)
    cum = np.cumsum(incs * sizes)
    total = cum[-1] if abs(cum[-1]) > 1e-12 else 1e-12
    x = np.concatenate([[0.0], np.cumsum(sizes) / sizes.sum()])
    yv = np.concatenate([[0.0], cum / total])
    return 2 * float(np.trapezoid(yv, x)) - 1


def main():
    con = duckdb.connect()
    con.execute("SET memory_limit='6GB'")
    d = con.execute("""
        SELECT f0,f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,treatment,conversion
        FROM read_csv('data/criteo-uplift-v2.1.csv.gz', header=true)
        WHERE hash(f0,f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,treatment,conversion,visit,exposure) % 10000 < 430
    """).df()
    X = d[FEATS].values.astype(float)
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-9)
    y = d.conversion.values.astype(float)
    w = d.treatment.values.astype(int)
    print(f"样本 {len(y):,}  ATE = {y[w==1].mean()-y[w==0].mean():+.6f}")

    # ---------- 模型 1：T-learner（logistic） ----------
    b1 = fit_logit(Xs[w == 1], y[w == 1])
    b0 = fit_logit(Xs[w == 0], y[w == 0])
    Xd = np.column_stack([np.ones(len(Xs)), Xs])
    cate_logit = sigmoid(Xd @ b1) - sigmoid(Xd @ b0)

    # ---------- 模型 2：GBDT（若 sklearn 可用） ----------
    cate_gb = None
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier
        m1 = HistGradientBoostingClassifier(max_iter=120, max_depth=4, learning_rate=0.1,
                                            random_state=0).fit(Xs[w == 1], y[w == 1])
        m0 = HistGradientBoostingClassifier(max_iter=120, max_depth=4, learning_rate=0.1,
                                            random_state=0).fit(Xs[w == 0], y[w == 0])
        cate_gb = m1.predict_proba(Xs)[:, 1] - m0.predict_proba(Xs)[:, 1]
        print("GBDT 可用，已训练")
    except ImportError:
        print("sklearn 不可用，跳过 GBDT")

    # ---------- 评估 ----------
    models = [("T-learner (logistic)", cate_logit)]
    if cate_gb is not None:
        models.append(("T-learner (GBDT)", cate_gb))

    print(f"\n{'模型':<24}{'Qini':>9}{'Spearman':>11}{'p':>10}{'单调性':>10}")
    print("-" * 66)
    for name, sc in models:
        q = qini(y, w, sc)
        preds, incs, _ = decile_uplift(y, w, sc)
        rho, p = stats.spearmanr(preds, incs)
        mono = np.all(np.diff(incs) <= 1e-9)   # 是否单调递减
        print(f"{name:<24}{q:>9.4f}{rho:>11.3f}{p:>10.3g}{str(mono):>10}")

    # ---------- null 分布：随机打乱 score ----------
    rng = np.random.default_rng(0)
    nulls = []
    for _ in range(100):
        sc = rng.permutation(cate_logit)
        nulls.append(qini(y, w, sc))
    nulls = np.array(nulls)
    q_obs = qini(y, w, cate_logit)
    print(f"\n随机重排 null：Qini 均值 {nulls.mean():+.4f}  标准差 {nulls.std():.4f}  "
          f"95% 分位 {np.percentile(nulls,95):+.4f}")
    print(f"观测 Qini {q_obs:+.4f} -> 经验 p = {(nulls >= q_obs).mean():.3f}")

    # ---------- 每个分位的明细（logistic） ----------
    preds, incs, sizes = decile_uplift(y, w, cate_logit)
    print(f"\n{'分位':<8}{'样本':>9}{'预测增量':>13}{'实际增量':>13}{'实际/预测':>12}")
    for i, (p_, inc, sz) in enumerate(zip(preds, incs, sizes)):
        ratio = inc / p_ if abs(p_) > 1e-9 else np.nan
        print(f"前{(i+1)*10:>3}%  {sz:>9,}{p_:>13.6f}{inc:>13.6f}{ratio:>12.2f}")

    # 累计贡献
    cum = np.cumsum(incs * sizes); total = cum[-1]
    print(f"\n累计增量贡献：前 10% = {cum[0]/total:.1%}，前 20% = {cum[1]/total:.1%}，"
          f"前 50% = {cum[4]/total:.1%}")
    print("\n判读要点：")
    print("  · Spearman 显著为正  -> 排序有信息，可做定向投放")
    print("  · Spearman 不显著    -> Qini 虽高但排序不可靠，不能据此裁剪人群")
    print("  · 实际/预测 比值远离 1 -> 模型校准差，预测值不能直接当增量用")


if __name__ == "__main__":
    main()
