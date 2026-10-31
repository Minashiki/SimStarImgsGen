"""Build cache/tycho1.npz from the Tycho-1 MATLAB catalog."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from starsim.catalog import build_cache, default_cache_path, default_mat_path, sha256_file


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="清洗第谷星表并写入 npz 缓存")
    parser.add_argument("--mat", type=Path, default=default_mat_path(), help="Tycho1.mat 路径")
    parser.add_argument("--out", type=Path, default=default_cache_path(), help="输出 npz 路径")
    parser.add_argument("--log-file", type=Path, default=None, help="把日志另存为 utf-8 文件")
    args = parser.parse_args(argv)
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if args.log_file is not None:
        args.log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(args.log_file, encoding="utf-8"))
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s", handlers=handlers)
    built = build_cache(args.mat, args.out)
    cache_sha = sha256_file(args.out)
    print(
        f"kept {built['n_kept']} stars, dropped {built['n_dropped']} NaN Vmag -> {args.out}"
    )
    print(f"source_sha256 {built['meta']['source_sha256']}")
    print(f"cache_sha256 {cache_sha}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
