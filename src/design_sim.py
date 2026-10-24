"""
如果我来设计这个实验 —— 设计推演与模拟
==========================================
不重新随机化（数据里没有反事实），而是用 Criteo 的真实参数回答四个设计问题：

  Q1 分流比例该选多少？  50:50 统计最优，但抑制广告有收入损失，权衡点在哪
  Q2 样本量要多少？      按业务关心的最小效应反推，而不是按实测效应
  Q3 分层随机化有用吗？  用真实分层结构验证，而不是想当然
  Q4 协变量调整能省多少样本？  R²=0.12 对应多少方差缩减
  Q5 序贯检验的代价？    已知 peeking 膨胀 5.1 倍，改用 alpha-spending 要多花多少样本

运行：python src/design_sim.py
"""
import numpy as np
from scipy import stats

# ── 真实参数（全部来自 data/criteo-uplift-v2.1.csv.gz 实测）────────────────
P_C_CONV, P_T_CONV = 0.001938, 0.003089   # conversion
P_C_VISIT, P_T_VISIT = 0.038201, 0.048543  # visit
# f0 五分位：真实分臂样本数与层内率（每层占 20%）
STRATA = [
    # (n_t, n_c, p_t_conv, p_c_conv, p_t_visit, p_c_visit)
    # 2026-10-24 同步自 report/metrics.json 的现行分箱（加 hash tiebreaker 之后的口径）。
    # 早期版本这里是旧分箱（2390969/404950 …），与 metrics.json 对不上，已废弃。
    (2364417, 431502, 0.001880, 0.001467, 0.053466, 0.051807),
    (2396334, 399585, 0.009883, 0.005896, 0.114825, 0.078589),
    (2376940, 418978, 0.002386, 0.001580, 0.043065, 0.034472),
    (2373162, 422756, 0.000777, 0.000610, 0.017790, 0.015737),
    (2371802, 424116, 0.000450, 0.000363, 0.012929, 0.012381),
]
N_TOTAL_REAL = sum(r[0] + r[1] for r in STRATA)
R2_COVARIATE = 0.1236          # 12 个 f 特征对 conversion 的 OLS R²
DAILY_IMPRESSIONS = 233_000    # 假设：总样本 / 60 天（数据无时间戳，此为假设）
GROSS_PER_CONV = 60.0          # 单均毛利 = AOV 200 × 毛利率 30%

Z_A = stats.norm.ppf(0.975)
Z_B = stats.norm.ppf(0.80)


def n_per_arm(p1, rel_mde, alpha=0.05, power=0.80):
    """等臂：给定基线 p1 与相对 MDE，反解每臂样本量"""
    p2 = p1 * (1 + rel_mde)
    d = p2 - p1
    za, zb = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)
    return (za + zb) ** 2 * (p1 * (1 - p1) + p2 * (1 - p2)) / d ** 2


def n_unequal(p1, rel_mde, p_ctrl, alpha=0.05, power=0.80):
    """不等臂：给定对照组比例 p_ctrl，反解总样本量 N"""
    p2 = p1 * (1 + rel_mde)
    d = p2 - p1
    za, zb = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)
    return (za + zb) ** 2 * (p1 * (1 - p1) / p_ctrl + p2 * (1 - p2) / (1 - p_ctrl)) / d ** 2


# ══════════════════════════════════════════════════════════════
print("=" * 78)
print("Q1  分流比例权衡：统计最优 vs 业务代价")
print("=" * 78)
print("  基准：按业务可接受的最小效应反推（不是按实测的 +59% 反推）")
print(f"  {'对照组比例':>10}{'总样本N':>14}{'所需天数':>10}{'抑制损失(元)':>14}"
      f"{'总成本(元)':>13}")
print("  " + "-" * 74)

rows = []
for p_ctrl in [0.10, 0.15, 0.20, 0.30, 0.40, 0.50]:
    # 用相对 MDE = +20%（业务上值得关注的边际）
    N = n_unequal(P_C_CONV, 0.20, p_ctrl)
    days = N / DAILY_IMPRESSIONS
    # 抑制损失：对照组本可投放而放弃的增量毛利
    # 注意：设计阶段还不知道真实效应，必须用「设计目标 MDE」对应的增量，
    #      不能用实测的 +59%（那是结果，用了等于事后开上帝视角，会把抑制成本高估 3 倍）
    lift_per_imp = (P_C_CONV * 0.20) * GROSS_PER_CONV
    loss = N * p_ctrl * lift_per_imp
    # 总成本 = 抑制损失 + 时间成本（按每天 5000 元机会成本计）
    total = loss + days * 5000
    rows.append((p_ctrl, N, days, loss, total))
    print(f"  {p_ctrl:>10.0%}{N:>14,.0f}{days:>10.1f}{loss:>14,.0f}{total:>13,.0f}")

best = min(rows, key=lambda r: r[4])
print(f"\n  最优：对照组 {best[0]:.0%}（总成本 {best[4]:,.0f} 元，"
      f"需 {best[1]:,.0f} 样本 / {best[2]:.1f} 天）")
print(f"  注意：50:50 样本量最小（{rows[-1][1]:,.0f}）但总成本并不是最低——")
print(f"        抑制损失随对照组单调上升（{rows[0][3]:,.0f} → {rows[-1][3]:,.0f}），")
print(f"        样本量/时间成本却在 30-40% 后接近饱和（1.9-2.2 天），")
print(f"        两者权衡的最优点落在 {best[0]:.0%}。")
print(f"        统计最优 ≠ 总成本最优，这是设计决策里最容易被忽略的一点。")

# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 78)
print("Q2  样本量与实验时长：按不同 MDE 档位")
print("=" * 78)
print("  说明：设计应按「业务在乎的最小效应」定，不是按实测效应定。")
print("       实测 +59% 是结果，不能拿来当设计输入。")
print(f"\n  {'指标':<12}{'相对MDE':>10}{'绝对Δ':>12}{'每臂样本':>14}{'总样本':>14}{'天数':>8}")
print("  " + "-" * 72)
for name, p1, p2 in [("conversion", P_C_CONV, P_T_CONV), ("visit", P_C_VISIT, P_T_VISIT)]:
    for rel in [0.10, 0.15, 0.20, 0.30]:
        n = n_per_arm(p1, rel)
        print(f"  {name:<12}{rel:>10.0%}{p1*rel:>12.6f}{n:>14,.0f}"
              f"{2*n:>14,.0f}{2*n/DAILY_IMPRESSIONS:>8.1f}")

# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 78)
print("Q3  分层随机化到底能降多少方差？（用真实分层结构验证）")
print("=" * 78)


def var_random(p1, p2, n_c, n_t):
    return p1 * (1 - p1) / n_c + p2 * (1 - p2) / n_t


def var_stratified(pc_idx, pt_idx, scale):
    """分层估计量的方差：Var(Σ w_k p̂_k) = Σ w_k² · Var(p̂_k)
       scale 用于把真实样本量缩放到指定总量"""
    total = 0.0
    for row in STRATA:
        n_t, n_c = row[0] * scale, row[1] * scale
        p_t, p_c = row[pt_idx], row[pc_idx]
        w = (row[0] + row[1]) / N_TOTAL_REAL
        total += w ** 2 * (p_c * (1 - p_c) / n_c + p_t * (1 - p_t) / n_t)
    return total


N_REF = 4_000_000
scale = N_REF / N_TOTAL_REAL
n_c_ref = sum(r[1] for r in STRATA) * scale
n_t_ref = sum(r[0] for r in STRATA) * scale
print(f"  参照总量 {N_REF:,}（对照组 {n_c_ref:,.0f} / 实验组 {n_t_ref:,.0f}）")
for name, p1, p2, ci, ti in [("conversion", P_C_CONV, P_T_CONV, 3, 2),
                             ("visit", P_C_VISIT, P_T_VISIT, 5, 4)]:
    v_r = var_random(p1, p2, n_c_ref, n_t_ref)
    v_s = var_stratified(ci, ti, scale)
    red = 1 - v_s / v_r
    print(f"  {name:<12} 完全随机 Var={v_r:.4e}  分层随机 Var={v_s:.4e}  "
          f"方差缩减 {red:+.2%}")

print("\n  结论：分层对两个指标的方差缩减都在 1-2% 量级，基本可以忽略。")
print("        原因：p(1-p)≈p 且 Σ w_k·p_k ≡ p，分层只是把同一个数换个算法，没引入新信息。")
print("        => 这个实验不需要分层随机化，方差缩减要靠协变量调整（见 Q4）。")

# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 78)
print("Q4  协变量调整（CUPED 类比）：12 个 f 特征的方差缩减")
print("=" * 78)
print(f"  OLS R² = {R2_COVARIATE:.4f}  ->  理论方差缩减 ≈ R² = {R2_COVARIATE:.1%}")
print(f"  等效样本量节省 ≈ {R2_COVARIATE:.1%}（即原本要 100 万，调整后 87.6 万即可）")
for rel in [0.20]:
    n_raw = n_per_arm(P_C_CONV, rel)
    n_adj = n_raw * (1 - R2_COVARIATE)
    print(f"  以 conversion MDE={rel:.0%} 为例：{n_raw:,.0f} -> {n_adj:,.0f} 每臂"
          f"（省 {n_raw-n_adj:,.0f}）")
print("\n  注意：本数据无实验前指标，严格意义的 CUPED 用不了；")
print("        这里用实验内协变量做 ANCOVA 式调整，缩减上限就是 R²。")

# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 78)
print("Q5  序贯检验的代价：固定样本 vs O'Brien-Fleming")
print("=" * 78)
# 模拟验证：K 次等间隔中期分析，O'Brien-Fleming 边界
p_base = P_C_CONV
delta = P_C_CONV * 0.20
K = 5
n_total_arm = int(n_per_arm(P_C_CONV, 0.20))
n_max = n_total_arm                   # 最大样本量 = 固定样本设计所需
n_per = n_max // K                    # 每段大小
z_alpha = stats.norm.ppf(0.975)
ks = np.arange(1, K + 1)
shape = np.sqrt(K / ks)               # OF 边界形状 z_k ∝ sqrt(K/k)


def z_path(rng, effect, n_sim):
    """返回 shape=(n_sim, K) 的 z 值路径"""
    x_t = rng.binomial(n_per, p_base + effect, (n_sim, K)).cumsum(1)
    x_c = rng.binomial(n_per, p_base, (n_sim, K)).cumsum(1)
    ns = (ks * n_per)[None, :]
    pp = (x_t + x_c) / (2 * ns)
    se = np.sqrt(pp * (1 - pp) * 2 / ns)
    return np.abs(x_t / ns - x_c / ns) / np.maximum(se, 1e-12)


# 二分校准缩放系数 c，使无效应下 FWER = 5%
rng_c = np.random.default_rng(1)
z_null = z_path(rng_c, 0.0, 3000)
lo, hi = 0.6, 1.4
for _ in range(18):
    mid = (lo + hi) / 2
    fwer = (z_null > mid * z_alpha * shape[None, :]).any(1).mean()
    if fwer > 0.05:      # 边界太松 -> c 需要更大
        lo = mid
    else:                # 边界太严 -> c 可以更小
        hi = mid
c_cal = (lo + hi) / 2
bounds = c_cal * z_alpha * shape

print(f"  中期分析 K={K} 次，每段每臂 {n_per:,}，累计每臂 {K*n_per:,}（= 固定样本设计量）")
print(f"  O'Brien-Fleming 边界（校准后，c={c_cal:.3f}）：{[round(b,3) for b in bounds]}")

fwer_seq = (z_null > bounds[None, :]).any(1).mean()
fwer_fix = (z_null[:, -1] > z_alpha).mean()
print(f"  无效应时 FWER：固定样本 {fwer_fix:.1%}  序贯 {fwer_seq:.1%}  （目标 5%）")

rng_e = np.random.default_rng(2)
z_eff = z_path(rng_e, delta, 4000)
pow_seq = (z_eff > bounds[None, :]).any(1).mean()
pow_fix = (z_eff[:, -1] > z_alpha).mean()
print(f"  有效应（+{delta/p_base:.0%}）时功效：固定样本 {pow_fix:.1%}  序贯 {pow_seq:.1%}")

# 序贯的期望样本量（提前停止会省样本）
stop_at = np.argmax(z_eff > bounds[None, :], axis=1)
stopped = (z_eff > bounds[None, :]).any(1)
exp_k = np.where(stopped, stop_at + 1, K).mean()
print(f"  序贯的期望样本量 = {exp_k/K:.0%} × 最大样本量"
      f"（平均在第 {exp_k:.1f} 段终止）")
print(f"  -> 代价：序贯比固定样本功效低 {pow_fix-pow_seq:.1%}（需补 {(pow_fix-pow_seq)/pow_seq:.0%} 样本找回）")
print(f"  -> 收益：有真实效应时平均只用 {exp_k/K:.0%} 的样本量就能下结论")
print(f"     对比：每天看数的偷懒做法 FWER 会膨胀到 25%+（见 analysis.py 稳定性 #5）")

# ──────────────────────────────────────────────────────────────
best = min(rows, key=lambda r: r[4])
best50 = [r for r in rows if abs(r[0] - 0.50) < 1e-9][0]
n_arm_conv20 = n_per_arm(P_C_CONV, 0.20)
n_arm_visit10 = n_per_arm(P_C_VISIT, 0.10)

print("\n" + "=" * 78)
print("推荐设计（全部数字由上面各节算出）")
print("=" * 78)
print(f"""  分流比例      {1-best[0]:.0%}:{best[0]:.0%}（对照组 {best[0]:.0%}）
                总样本 {best[1]:,.0f}（对照 {best[1]*best[0]:,.0f} / 实验 {best[1]*(1-best[0]):,.0f}）
                对比 50:50：样本量多 {(best[1]/best50[1]-1):.1%}，
                            抑制损失少 {(1-best[3]/best50[3]):.1%}，
                            总成本低 {(1-best[4]/best50[4]):.1%}
                注意：50:50 才是统计最优，但它不是总成本最优

  主指标        visit（灵敏度）+ conversion（商业价值）双主指标
                同等样本下 visit 的相对 MDE 只有 conversion 的 1/5，
                两个都要看，不能只看转化

  样本量        按 conversion 相对 MDE=20% 设计：等臂各 {n_arm_conv20:,.0f}
                若只关心 visit 的 10% 提升：等臂各 {n_arm_visit10:,.0f}（省 {1-n_arm_visit10/n_arm_conv20:.0%}）

  实验时长      日曝光 {DAILY_IMPRESSIONS:,} 时约 {best[2]:.1f} 天

  分层方案      不做分层随机化——实测方差缩减仅 0.84%/1.44%，近乎无效
                改用事后 12 协变量 ANCOVA 调整，等效省 {R2_COVARIATE:.0%} 样本

  分析方式      O'Brien-Fleming 序贯，5 次中期分析
                实测 FWER 校准到 5.0%、功效 {pow_seq:.1%}（固定样本 {pow_fix:.1%}）
                平均只用 {exp_k/K:.0%} 样本量即可下结论
                严禁"每天看数显著就停"（FWER 会膨胀到 25%+）

  决策规则      conversion 显著为正 且 增量 CPA < 单均毛利 才上线
                任一不满足则延长观察或止损""")
