-- ============================================================
-- Criteo-UPLIFT v2.1 —— 广告增量性 A/B 实验 · 拉数层
-- 每个查询独立可跑；结果被 src/analysis.py 复用
-- 运行：duckdb < sql/02_queries.sql
-- ============================================================


-- ============================================================
-- Q1. SRM：分流健康度
--     期望 85:15（数据集的原始设计），偏离说明分流管道有问题
-- ============================================================
SELECT
    treatment,
    COUNT(*)                                            AS n,
    ROUND(COUNT(*) * 1.0 / SUM(COUNT(*)) OVER (), 6)    AS share,
    ROUND(COUNT(*) * 1.0 / SUM(COUNT(*)) OVER () - (CASE WHEN treatment = 1 THEN 0.85 ELSE 0.15 END), 6)
                                                        AS deviation_vs_design
FROM raw
GROUP BY treatment
ORDER BY treatment;
-- 判读：deviation_vs_design 绝对值 > 0.01 需排查；卡方检验见 analysis.py


-- ============================================================
-- Q2. AA 检验：12 个协变量在两组间的均值差（标准化）
--     srmd = |均值差| / 合并标准差，>0.1 视为不平衡
-- ============================================================
WITH stats AS (
    SELECT
        treatment,
        AVG(f0) f0_m, AVG(f1) f1_m, AVG(f2) f2_m, AVG(f3) f3_m,
        AVG(f4) f4_m, AVG(f5) f5_m, AVG(f6) f6_m, AVG(f7) f7_m,
        AVG(f8) f8_m, AVG(f9) f9_m, AVG(f10) f10_m, AVG(f11) f11_m,
        STDDEV_SAMP(f0) f0_s, STDDEV_SAMP(f1) f1_s, STDDEV_SAMP(f2) f2_s, STDDEV_SAMP(f3) f3_s,
        STDDEV_SAMP(f4) f4_s, STDDEV_SAMP(f5) f5_s, STDDEV_SAMP(f6) f6_s, STDDEV_SAMP(f7) f7_s,
        STDDEV_SAMP(f8) f8_s, STDDEV_SAMP(f9) f9_s, STDDEV_SAMP(f10) f10_s, STDDEV_SAMP(f11) f11_s,
        COUNT(*) n
    FROM raw GROUP BY treatment
),
pivoted AS (
    SELECT
        MAX(CASE WHEN treatment=1 THEN f0_m END) t_f0,  MAX(CASE WHEN treatment=0 THEN f0_m END) c_f0,
        MAX(CASE WHEN treatment=1 THEN f1_m END) t_f1,  MAX(CASE WHEN treatment=0 THEN f1_m END) c_f1,
        MAX(CASE WHEN treatment=1 THEN f2_m END) t_f2,  MAX(CASE WHEN treatment=0 THEN f2_m END) c_f2,
        MAX(CASE WHEN treatment=1 THEN f3_m END) t_f3,  MAX(CASE WHEN treatment=0 THEN f3_m END) c_f3,
        MAX(CASE WHEN treatment=1 THEN f4_m END) t_f4,  MAX(CASE WHEN treatment=0 THEN f4_m END) c_f4,
        MAX(CASE WHEN treatment=1 THEN f5_m END) t_f5,  MAX(CASE WHEN treatment=0 THEN f5_m END) c_f5,
        MAX(CASE WHEN treatment=1 THEN f6_m END) t_f6,  MAX(CASE WHEN treatment=0 THEN f6_m END) c_f6,
        MAX(CASE WHEN treatment=1 THEN f7_m END) t_f7,  MAX(CASE WHEN treatment=0 THEN f7_m END) c_f7,
        MAX(CASE WHEN treatment=1 THEN f8_m END) t_f8,  MAX(CASE WHEN treatment=0 THEN f8_m END) c_f8,
        MAX(CASE WHEN treatment=1 THEN f9_m END) t_f9,  MAX(CASE WHEN treatment=0 THEN f9_m END) c_f9,
        MAX(CASE WHEN treatment=1 THEN f10_m END) t_f10, MAX(CASE WHEN treatment=0 THEN f10_m END) c_f10,
        MAX(CASE WHEN treatment=1 THEN f11_m END) t_f11, MAX(CASE WHEN treatment=0 THEN f11_m END) c_f11,
        MAX(CASE WHEN treatment=1 THEN f0_s END) t_f0s, MAX(CASE WHEN treatment=0 THEN f0_s END) c_f0s,
        MAX(CASE WHEN treatment=1 THEN f1_s END) t_f1s, MAX(CASE WHEN treatment=0 THEN f1_s END) c_f1s,
        MAX(CASE WHEN treatment=1 THEN f2_s END) t_f2s, MAX(CASE WHEN treatment=0 THEN f2_s END) c_f2s,
        MAX(CASE WHEN treatment=1 THEN f3_s END) t_f3s, MAX(CASE WHEN treatment=0 THEN f3_s END) c_f3s,
        MAX(CASE WHEN treatment=1 THEN f4_s END) t_f4s, MAX(CASE WHEN treatment=0 THEN f4_s END) c_f4s,
        MAX(CASE WHEN treatment=1 THEN f5_s END) t_f5s, MAX(CASE WHEN treatment=0 THEN f5_s END) c_f5s,
        MAX(CASE WHEN treatment=1 THEN f6_s END) t_f6s, MAX(CASE WHEN treatment=0 THEN f6_s END) c_f6s,
        MAX(CASE WHEN treatment=1 THEN f7_s END) t_f7s, MAX(CASE WHEN treatment=0 THEN f7_s END) c_f7s,
        MAX(CASE WHEN treatment=1 THEN f8_s END) t_f8s, MAX(CASE WHEN treatment=0 THEN f8_s END) c_f8s,
        MAX(CASE WHEN treatment=1 THEN f9_s END) t_f9s, MAX(CASE WHEN treatment=0 THEN f9_s END) c_f9s,
        MAX(CASE WHEN treatment=1 THEN f10_s END) t_f10s, MAX(CASE WHEN treatment=0 THEN f10_s END) c_f10s,
        MAX(CASE WHEN treatment=1 THEN f11_s END) t_f11s, MAX(CASE WHEN treatment=0 THEN f11_s END) c_f11s
    FROM stats
)
SELECT 'f0'  AS feature, ROUND(ABS(t_f0 -c_f0 ) / NULLIF((t_f0s +c_f0s )/2,0), 4) AS srmd FROM pivoted
UNION ALL SELECT 'f1',  ROUND(ABS(t_f1 -c_f1 ) / NULLIF((t_f1s +c_f1s )/2,0), 4) FROM pivoted
UNION ALL SELECT 'f2',  ROUND(ABS(t_f2 -c_f2 ) / NULLIF((t_f2s +c_f2s )/2,0), 4) FROM pivoted
UNION ALL SELECT 'f3',  ROUND(ABS(t_f3 -c_f3 ) / NULLIF((t_f3s +c_f3s )/2,0), 4) FROM pivoted
UNION ALL SELECT 'f4',  ROUND(ABS(t_f4 -c_f4 ) / NULLIF((t_f4s +c_f4s )/2,0), 4) FROM pivoted
UNION ALL SELECT 'f5',  ROUND(ABS(t_f5 -c_f5 ) / NULLIF((t_f5s +c_f5s )/2,0), 4) FROM pivoted
UNION ALL SELECT 'f6',  ROUND(ABS(t_f6 -c_f6 ) / NULLIF((t_f6s +c_f6s )/2,0), 4) FROM pivoted
UNION ALL SELECT 'f7',  ROUND(ABS(t_f7 -c_f7 ) / NULLIF((t_f7s +c_f7s )/2,0), 4) FROM pivoted
UNION ALL SELECT 'f8',  ROUND(ABS(t_f8 -c_f8 ) / NULLIF((t_f8s +c_f8s )/2,0), 4) FROM pivoted
UNION ALL SELECT 'f9',  ROUND(ABS(t_f9 -c_f9 ) / NULLIF((t_f9s +c_f9s )/2,0), 4) FROM pivoted
UNION ALL SELECT 'f10', ROUND(ABS(t_f10-c_f10) / NULLIF((t_f10s+c_f10s)/2,0), 4) FROM pivoted
UNION ALL SELECT 'f11', ROUND(ABS(t_f11-c_f11) / NULLIF((t_f11s+c_f11s)/2,0), 4) FROM pivoted
ORDER BY feature;
-- 判读：srmd < 0.1 视为平衡


-- ============================================================
-- Q3. 链路漏斗：L1 分配 → L2 曝光 → L3 访问 → L4 转化
--     每次率都同时给分母，避免"率的分母不透明"
-- ============================================================
SELECT
    treatment,
    COUNT(*)                                             AS l1_assigned,
    SUM(exposure)                                        AS l2_exposed,
    SUM(visit)                                           AS l3_visited,
    SUM(conversion)                                      AS l4_converted,
    ROUND(SUM(exposure)   * 1.0 / COUNT(*), 6)           AS exposure_rate,
    ROUND(SUM(visit)      * 1.0 / COUNT(*), 6)           AS visit_rate,
    ROUND(SUM(conversion) * 1.0 / COUNT(*), 6)           AS conv_rate,
    -- 注意：不要再算 visit/exposure。实测该比值 >100%，因为访问可在无曝光时自然发生，
    --       exposure 是处理依从性指标，不是 visit 的前置漏斗层，二者是部分重叠关系
    ROUND(SUM(visit)      * 1.0 / NULLIF(SUM(exposure),0), 4) AS visit_over_exposure_ratio,
    ROUND(SUM(conversion) * 1.0 / NULLIF(SUM(visit),0), 6)    AS conv_given_visit
-- 判读：conv_given_visit 才是真正嵌套的条件转化率，可跨臂比较
FROM raw
GROUP BY treatment
ORDER BY treatment;


-- ============================================================
-- Q4. 依从性：曝光分布（回答"分到 treatment 就真的被投放了吗"）
--     这是 ITT → CACE 的分母，也是自选择偏差的入口
-- ============================================================
SELECT
    treatment,
    exposure,
    COUNT(*)                                          AS n,
    ROUND(COUNT(*) * 1.0 / SUM(COUNT(*)) OVER (PARTITION BY treatment), 6) AS share_within_arm,
    ROUND(AVG(visit), 6)                              AS visit_rate,
    ROUND(AVG(conversion), 6)                         AS conv_rate
FROM raw
GROUP BY treatment, exposure
ORDER BY treatment, exposure;
-- 判读：control 组 exposure 必须恒为 0；treatment 组 exposure=1 的占比就是依从率


-- ============================================================
-- Q5. 分层效应：按 strat_value × treatment 看链路指标
--     注意：这里是描述性切片，正式异质性检验见 analysis.py 的交互项模型
-- ============================================================
SELECT
    strat_value,
    treatment,
    COUNT(*)                              AS n,
    ROUND(AVG(visit), 6)                  AS visit_rate,
    ROUND(AVG(conversion), 6)             AS conv_rate,
    ROUND(SUM(conversion) * 1.0 / NULLIF(SUM(visit),0), 6) AS conv_given_visit
FROM abt
GROUP BY strat_value, treatment
ORDER BY strat_value, treatment;


-- ============================================================
-- Q6. 增量归因：按分层拆解"增量转化从哪来"
--     incremental = (arm 转化率 - 对照转化率) × arm 样本量
--     share       = 该层增量 / 总增量
-- ============================================================
WITH agg AS (
    SELECT
        strat_value,
        COUNT(*) FILTER (WHERE treatment = 1)  AS n_t,
        COUNT(*) FILTER (WHERE treatment = 0)  AS n_c,
        AVG(conversion) FILTER (WHERE treatment = 1) AS cvr_t,
        AVG(conversion) FILTER (WHERE treatment = 0) AS cvr_c
    FROM abt
    GROUP BY strat_value
),
calc AS (
    SELECT *,
        (cvr_t - cvr_c) * n_t AS incremental,
        cvr_c * n_t            AS baseline_counterfactual
    FROM agg
)
SELECT
    strat_value,
    n_t, n_c,
    ROUND(cvr_c, 6)                                             AS cvr_control,
    ROUND(cvr_t, 6)                                             AS cvr_treated,
    ROUND(cvr_t - cvr_c, 6)                                     AS abs_lift,
    ROUND((cvr_t - cvr_c) / NULLIF(cvr_c, 0), 4)                AS rel_lift,
    ROUND(incremental, 0)                                       AS incremental_conversions,
    ROUND(incremental / NULLIF(SUM(incremental) OVER (), 0), 4) AS share_of_increment
FROM calc
ORDER BY strat_value;
-- 判读：share_of_increment 告诉你增量集中在哪层；若某层 abs_lift<0 说明该层不该投


-- ============================================================
-- Q7. 曝光子群对比：为什么不能直接比 exposed vs control
-- ============================================================
SELECT
    CASE
        WHEN treatment = 0 THEN 'control (never exposed)'
        WHEN treatment = 1 AND exposure = 1 THEN 'treated & exposed'
        ELSE 'treated & NOT exposed'
    END                                AS grp,
    COUNT(*)                           AS n,
    ROUND(AVG(visit), 6)               AS visit_rate,
    ROUND(AVG(conversion), 6)          AS conv_rate,
    ROUND(SUM(conversion) * 1.0 / NULLIF(SUM(visit),0), 6) AS conv_given_visit
FROM raw
GROUP BY grp
ORDER BY grp;
-- 关键判读：treated & NOT exposed 若与 control 不同，说明曝光分配非随机
--           此时 exposed vs control 的差值混杂了"选择效应"，不是因果效应


-- ============================================================
-- Q8. 稳定性口径 #2：各分层的效应符号一致性
-- ============================================================
WITH s AS (
    SELECT
        strat_engage,
        AVG(conversion) FILTER (WHERE treatment = 1) -
        AVG(conversion) FILTER (WHERE treatment = 0) AS lift
    FROM abt
    GROUP BY strat_engage
)
SELECT
    COUNT(*)                                              AS n_strata,
    SUM(CASE WHEN lift > 0 THEN 1 ELSE 0 END)             AS positive_strata,
    ROUND(AVG(lift), 6)                                   AS mean_lift,
    ROUND(MIN(lift), 6)                                   AS min_lift,
    ROUND(MAX(lift), 6)                                   AS max_lift,
    ROUND(STDDEV_SAMP(lift) / NULLIF(AVG(lift), 0), 4)    AS cv_across_strata
FROM s;
