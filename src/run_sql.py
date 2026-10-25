"""
SQL 验证器
==========
跑通 sql/01_build.sql + sql/02_queries.sql 的全部查询，把结果导出成 Markdown。
用途：
  1. 自检——确保仓库里的 SQL 不是"写完没跑过"
  2. 对账——和 Python 侧（report/metrics.json）交叉验证同一批指标

运行：python src/run_sql.py
输出：report/sql_results.md
"""
import json
import re
import time

import duckdb

from dbutil import connect as duck_connect
import pandas as pd

BUILD = "sql/01_build.sql"
QUERIES = "sql/02_queries.sql"
OUT = "report/sql_results.md"


def split_sql(text):
    """按分号拆分语句，跳过纯注释块"""
    # 去掉行注释后再判断，但保留原语句用于展示
    stmts, buf = [], []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("--"):
            buf.append(line)
            continue
        buf.append(line)
        if stripped.endswith(";"):
            body = "\n".join(buf).strip()
            # 去掉注释后是否还有实质内容
            code = "\n".join(l for l in body.splitlines() if not l.strip().startswith("--"))
            code = code.replace(";", "").strip()
            if code:
                stmts.append(body.rstrip(";").strip())
            buf = []
    tail = "\n".join(buf).strip()
    if tail:
        code = "\n".join(l for l in tail.splitlines() if not l.strip().startswith("--")).strip()
        if code:
            stmts.append(tail)
    return stmts


def fmt_cell(v):
    """数字格式化：整数加千分位，小数保留有效位，避免 2.09694e+06 和 -0"""
    if v is None or (not isinstance(v, str) and pd.isna(v)):
        return ""
    if isinstance(v, (int,)) and not isinstance(v, bool):
        return f"{v:,}"
    if isinstance(v, float):
        if abs(v) < 1e-12:
            return "0"
        if abs(v - round(v)) < 1e-9 and abs(v) >= 1:
            return f"{int(round(v)):,}"
        if abs(v) >= 1:
            return f"{v:,.4f}".rstrip("0").rstrip(".")
        if abs(v) >= 1e-4:
            return f"{v:.6f}".rstrip("0").rstrip(".")
        return f"{v:.3e}"
    return str(v)


def df_to_md(df):
    """零依赖的 markdown 表格（不引入 tabulate）"""
    cols = [str(c) for c in df.columns]
    out = ["| " + " | ".join(cols) + " |",
           "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in df.iterrows():
        out.append("| " + " | ".join(fmt_cell(v) for v in row) + " |")
    return "\n".join(out)


def title_of(sql, idx):
    """从语句前面的注释里抓 Qn 标题"""
    m = re.search(r"--\s*(Q\d+\.\s*[^\n]+)", sql)
    return m.group(1).strip() if m else f"Query {idx}"


def main():
    print("=" * 70)
    print("SQL 验证：建表")
    print("=" * 70)
    con = duck_connect()
    t0 = time.time()
    con.execute(open(BUILD, encoding="utf-8").read())
    print(f"  01_build.sql  OK  ({time.time()-t0:.1f}s)")

    for tbl in ["raw", "abt"]:
        n = con.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
        print(f"  {tbl:<6} {n:>12,} 行")
    bad = con.execute("SELECT bad_rows FROM qc_bad_label").fetchone()[0]
    ctrl_exp = con.execute("SELECT bad_rows FROM qc_ctrl_exposed").fetchone()[0]
    print(f"  自检：conversion=1 且 visit=0 -> {bad} 行；"
          f"control 组 exposure=1 -> {ctrl_exp} 行（都应为 0）")

    print("\n" + "=" * 70)
    print("SQL 验证：8 个查询")
    print("=" * 70)
    stmts = split_sql(open(QUERIES, encoding="utf-8").read())
    print(f"  解析出 {len(stmts)} 条语句\n")

    md = ["# SQL 查询结果", "",
          f"> 自动生成，来源 `sql/01_build.sql` + `sql/02_queries.sql`，"
          f"数据 {con.execute('SELECT COUNT(*) FROM raw').fetchone()[0]:,} 行", ""]
    ok, fail = 0, 0
    for i, st in enumerate(stmts, 1):
        name = title_of(st, i)
        print(f"  [{i}/{len(stmts)}] {name}")
        try:
            t = time.time()
            df = con.execute(st).df()
            dt = time.time() - t
            print(f"        OK  {len(df)} 行 × {len(df.columns)} 列  ({dt:.2f}s)")
            ok += 1
            md += [f"## {name}", "", df_to_md(df), "",
                   f"*{len(df)} 行，{dt:.2f}s*", ""]
        except Exception as e:
            fail += 1
            msg = str(e).splitlines()[0][:160]
            print(f"        FAIL  {msg}")
            md += [f"## {name}", "", f"**执行失败**：`{msg}`", ""]

    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(md))

    print("\n" + "=" * 70)
    print(f"结果：{ok} 成功 / {fail} 失败   ->  {OUT}")
    return fail


def reconcile():
    """与 Python 侧交叉对账"""
    print("\n" + "=" * 70)
    print("交叉对账：SQL vs Python (report/metrics.json)")
    print("=" * 70)
    try:
        m = json.load(open("report/metrics.json", encoding="utf-8"))
    except FileNotFoundError:
        print("  未找到 metrics.json，跳过")
        return

    con = duck_connect()
    con.execute(open(BUILD, encoding="utf-8").read())

    r = con.execute("""
        SELECT treatment, COUNT(*) n, AVG(visit) v, AVG(conversion) c
        FROM raw GROUP BY treatment ORDER BY treatment
    """).df().set_index("treatment")

    checks = [
        ("control 样本量", int(r.loc[0, "n"]), m["srm"]["n_c"]),
        ("treatment 样本量", int(r.loc[1, "n"]), m["srm"]["n_t"]),
        ("control visit 率", float(r.loc[0, "v"]), m["itt"]["visit"]["p_c"]),
        ("treatment visit 率", float(r.loc[1, "v"]), m["itt"]["visit"]["p_t"]),
        ("control conv 率", float(r.loc[0, "c"]), m["itt"]["conversion"]["p_c"]),
        ("treatment conv 率", float(r.loc[1, "c"]), m["itt"]["conversion"]["p_t"]),
    ]
    all_ok = True
    for name, a, b in checks:
        same = abs(a - b) < 1e-9
        all_ok &= same
        print(f"  {'✔' if same else '✘'} {name:<24} SQL={a:<16.10g} Python={b:<16.10g}")
    print(f"\n  {'全部一致' if all_ok else '存在不一致，需排查'}")


if __name__ == "__main__":
    nfail = main()
    reconcile()
    raise SystemExit(1 if nfail else 0)
