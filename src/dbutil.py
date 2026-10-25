"""DuckDB 连接的统一入口（可移植性）。

为什么需要：旧版各脚本写死 SET memory_limit='6GB'——在 4GB 内存的
笔记本上直接被 OOM killer 杀死（本项目实测）。改成按物理内存自适应：
上限 6GB（大机器不变），小机器取总内存的 60%，DuckDB 超限自动落盘。
"""
import duckdb


def connect():
    try:
        with open("/proc/meminfo") as f:            # Linux
            total_gb = int(f.readline().split()[1]) / 1048576
    except Exception:                               # Windows/macOS 或读取失败
        total_gb = 10.0
    limit_gb = max(1.5, min(6.0, total_gb * 0.6))
    con = duckdb.connect()
    con.execute(f"SET memory_limit='{limit_gb:.1f}GB'")
    con.execute("SET temp_directory='.duckdb_tmp'")   # 超限落盘目录（已进 .gitignore）
    return con
