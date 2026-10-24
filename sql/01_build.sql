-- ============================================================
-- Criteo-UPLIFT v2.1 —— 广告增量性 A/B 实验 · 建表层
-- 引擎：DuckDB（单机直接读 gz，无需导入数据库）
-- 运行：duckdb < sql/01_build.sql
--       或在 Python 里 con.execute(open('sql/01_build.sql').read())
-- ============================================================

-- ------------------------------------------------------------
-- 1. 原始表：13,979,592 行 × 16 列
--    粒度 = 一个用户（user），不是曝光机会。无 user_id、无 timestamp
--    注意：exposure 是「该 user 是否至少被曝光一次」的 0/1 标志，不是曝光次数
-- ------------------------------------------------------------
CREATE OR REPLACE TABLE raw AS
SELECT
    CAST(f0  AS FLOAT) AS f0,
    CAST(f1  AS FLOAT) AS f1,
    CAST(f2  AS FLOAT) AS f2,
    CAST(f3  AS FLOAT) AS f3,
    CAST(f4  AS FLOAT) AS f4,
    CAST(f5  AS FLOAT) AS f5,
    CAST(f6  AS FLOAT) AS f6,
    CAST(f7  AS FLOAT) AS f7,
    CAST(f8  AS FLOAT) AS f8,
    CAST(f9  AS FLOAT) AS f9,
    CAST(f10 AS FLOAT) AS f10,
    CAST(f11 AS FLOAT) AS f11,
    CAST(treatment  AS TINYINT) AS treatment,   -- 1=投放 0=抑制（RCT 分配）
    CAST(visit      AS TINYINT) AS visit,       -- 是否访问（链路 L2）
    CAST(conversion AS TINYINT) AS conversion,  -- 是否转化（链路 L3）
    CAST(exposure   AS TINYINT) AS exposure     -- 该 user 是否被曝光过（0/1，非随机、内生）
FROM read_csv(
    'data/criteo-uplift-v2.1.csv.gz',
    header = true,
    auto_detect = true
);

-- ------------------------------------------------------------
-- 2. 数据自检：标签自洽性（转化必须发生在访问之后）
--    预期 0 行。若非 0，说明数据源有问题，需在上游修
-- ------------------------------------------------------------
CREATE OR REPLACE TABLE qc_bad_label AS
SELECT COUNT(*) AS bad_rows
FROM raw
WHERE conversion = 1 AND visit = 0;

-- 预期：控制组 exposure 恒为 0（RCT 设计要求抑制组不得曝光）
CREATE OR REPLACE TABLE qc_ctrl_exposed AS
SELECT COUNT(*) AS bad_rows
FROM raw
WHERE treatment = 0 AND exposure = 1;

-- ------------------------------------------------------------
-- 3. 分析宽表：预计算分层变量，避免每次查询重复聚类
--    strat_value : f0 的分位分箱（价格/价值敏感度代理）
--    strat_engage: f1 的分位分箱（活跃度代理）
--    strat_prop  : 倾向分分箱（仅在对照组拟合，见 analysis.py）
-- ------------------------------------------------------------
-- 注意：f0 有大量重复值（100 万行仅 59 万个唯一值），单按 f0 排序时
--       NTILE 的 tie 分配不确定，每次跑分箱结果都会变。
--       加 hash 做 tiebreaker，保证分箱可复现。
CREATE OR REPLACE TABLE abt AS
SELECT
    r.*,
    NTILE(5) OVER (ORDER BY f0, hash(f0, f1, f2, f3)) AS strat_value,
    NTILE(5) OVER (ORDER BY f1, hash(f1, f0, f2, f3)) AS strat_engage,
    CASE WHEN treatment = 1 THEN 1 ELSE 0 END AS is_trt
FROM raw r;

-- ------------------------------------------------------------
-- 4. 常用视图：链路长表（便于统一算率）
-- ------------------------------------------------------------
CREATE OR REPLACE VIEW funnel_base AS
SELECT
    treatment,
    COUNT(*)                                        AS users,   -- 用户数，不是曝光次数
    SUM(exposure)                                   AS exposures,
    SUM(visit)                                      AS visits,
    SUM(conversion)                                 AS conversions
FROM raw
GROUP BY treatment;
