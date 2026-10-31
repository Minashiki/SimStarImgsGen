# Tycho-1 星表

原始文件为 `Tycho1/Tycho1.mat`（`starsim.catalog.default_mat_path()` 的默认位置）。
此文件由项目使用者提供，按项目约定不再追溯下载来源和转换过程。
原始文件保留在本地，不再由 Git 跟踪；新工作区需自行放入同一文件。

当前文件是 MATLAB `.mat`，读取变量 `Tyc`。代码使用第 1、2 列作为 TYC1、TYC2，
第 4 列作为 Vmag，第 5、6 列作为赤经、赤纬（度）；坐标采用 ICRS、J1991.25，
不传播自行。Vmag 为 NaN 的行会在构建缓存时剔除。

- 原始文件字节数：`19676044`
- 原始文件 SHA-256：`45a56070d55e294c5e628c063d14523f30acc1dc2e22b6434378a532079f585a`
- 缓存文件：`cache/tycho1.npz`，继续跟踪；运行时和 M3 基线依赖其身份。
- 预期缓存 SHA-256：`2269af793dfe87db04f2199edabf16fe87ba55ff83e18930e2fa90c5f6930ff1`

在项目根目录重建缓存：

```powershell
conda run -n genstarimg --no-capture-output python scripts/build_catalog_cache.py --mat Tycho1/Tycho1.mat
```

重建前请核对原始文件字节数与 SHA-256。
