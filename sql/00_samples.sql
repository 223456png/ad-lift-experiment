-- ============================================================
-- 确定性抽样层：所有"随机子样本"一律改为 md5 分折
-- ============================================================
-- 背景：USING SAMPLE n ROWS (reservoir, seed) 的结果取决于扫描路径、
-- 线程数和 DuckDB 版本——同一个 seed 两次运行连行数都不同
--（实测 600000 vs 599915），跨脚本（raw 表 vs read_csv 直读）更是必然
-- 抽到不同样本。这曾导致：LR 检验 p 在 0.019 与 0.072 之间漂移、
-- 样本外 Qini 在 0.05 与 0.50 之间漂移，README 与 metrics.json 各说各话。
-- 方法：md5(全部 16 列拼接) 的前 8 位十六进制 → 无符号整数 → % 5 分折。
-- md5 跨 DuckDB 版本稳定（内置 hash() 不保证跨版本稳定，故不用）。
-- 每折约 280 万行，五折臂间比例、转化率均与总体一致（已验证）。
--
-- 用途分配（改抽样只改这里）：
--   fold 0                  → LR 异质性检验（sample_lr，约 280 万）
--   fold 0..4               → CATE 五折轮转交叉拟合（fold i 训练 → fold i+1 评估）
--   fold 1 按 md5key 前 150 万 → Bootstrap 稳定性（sample_boot）
--   fold 2 按 md5key 末位奇偶   → 两半稳定性
-- ============================================================

-- 基表用 abt（= raw + strat_value/strat_engage 分箱列）：LR 异质性检验需要
-- strat_value，其余脚本只用原始列，取超集两边的需求都满足。
CREATE OR REPLACE VIEW det AS
SELECT
    *,
    md5(concat_ws('|',
        CAST(f0 AS VARCHAR), CAST(f1 AS VARCHAR), CAST(f2 AS VARCHAR), CAST(f3 AS VARCHAR),
        CAST(f4 AS VARCHAR), CAST(f5 AS VARCHAR), CAST(f6 AS VARCHAR), CAST(f7 AS VARCHAR),
        CAST(f8 AS VARCHAR), CAST(f9 AS VARCHAR), CAST(f10 AS VARCHAR), CAST(f11 AS VARCHAR),
        CAST(treatment AS VARCHAR), CAST(exposure AS VARCHAR),
        CAST(visit AS VARCHAR), CAST(conversion AS VARCHAR))) AS md5key,
    (CAST('0x' || substr(md5(concat_ws('|',
        CAST(f0 AS VARCHAR), CAST(f1 AS VARCHAR), CAST(f2 AS VARCHAR), CAST(f3 AS VARCHAR),
        CAST(f4 AS VARCHAR), CAST(f5 AS VARCHAR), CAST(f6 AS VARCHAR), CAST(f7 AS VARCHAR),
        CAST(f8 AS VARCHAR), CAST(f9 AS VARCHAR), CAST(f10 AS VARCHAR), CAST(f11 AS VARCHAR),
        CAST(treatment AS VARCHAR), CAST(exposure AS VARCHAR),
        CAST(visit AS VARCHAR), CAST(conversion AS VARCHAR))), 1, 8) AS UBIGINT) % 5) AS fold
FROM abt;

-- LR 异质性检验样本：fold 0 全折
CREATE OR REPLACE VIEW sample_lr AS
SELECT * FROM det WHERE fold = 0;

-- Bootstrap 样本：fold 1 内按 md5key 排序取前 150 万（确定性）
CREATE OR REPLACE VIEW sample_boot AS
SELECT * FROM det WHERE fold = 1 ORDER BY md5key LIMIT 1500000;
