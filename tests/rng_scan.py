"""AST scan that keeps random construction inside ``starsim/rng.py``.

``tests/`` is not part of the scan. ``filename`` must be a repo-relative
POSIX path. Only ``starsim/rng.py`` is exempt, and the match is the whole
path: ``PurePosixPath(filename).parts == ("starsim", "rng.py")``. An
absolute path is not exempt, so scanning ``rng.py`` itself fails loudly
instead of being quietly skipped.

定位：防的是无意中的误用，不防存心绕过。

受监控包：``numpy`` 的 ``random``、``scipy`` 的 ``stats`` 适用于 ``starsim/`` 和
``scripts/``；``os`` 的 ``urandom``、``uuid`` 的 ``uuid1`` / ``uuid4`` 只适用于
``starsim/``。``uuid3`` / ``uuid5`` 放行。numpy 的例外仍是未被调用的
``.random.Generator`` 和 ``.random.SeedSequence``。``secrets`` 与内置 ``hash()``
也只作用于 ``starsim/``。

``__import__`` 不带 ``fromlist``，或 ``fromlist`` 为 ``None``、``[]``、``()`` 时返回顶层包；
``fromlist`` 为非空字面量列表或元组时返回子模块。``import_module`` 按完整模块名返回。
位置参数和 ``name=`` / ``fromlist=`` 都识别。

``stream`` 只有调用目标确实指向 ``starsim.rng.stream`` 才报错。
静态 ``ImportFrom`` 按文件路径解析成绝对模块名。``stream`` 的绑定和从
``starsim.rng`` 星号导入的检查只调用 ``_resolve_import_from``，不再直接比较
``node.module``。当前包取文件路径的父目录：``starsim/x.py`` 和
``starsim/__init__.py`` 都是 ``starsim``，
``starsim/sub/x.py`` 是 ``starsim.sub``。``level == 0`` 时结果就是
``node.module``。``level`` 大于当前包的层数时越过了顶层包，返回 ``None``，
不退回 ``node.module``。否则把当前包往上退 ``level - 1`` 层作为基准包；
``module`` 不为空时接在基准包后面，为空时结果就是基准包。解析结果是
``starsim.rng`` 且导入 ``stream``（可带 ``as``）时记下本地名字；解析结果是
``starsim`` 且导入 ``rng``（可带 ``as``）时把本地名字记为 rng 模块别名；
其他结果或 ``None`` 不处理。``import starsim.rng as r`` 与 ``import starsim.rng``
仍按 ``Import`` 处理。从 ``starsim.rng`` 星号导入会在该导入语句上直接报错，
包括解析后指向 ``starsim.rng`` 的相对星号导入，``starsim/`` 和 ``scripts/``
都适用，不依赖后面有没有调用 ``stream``。
``SeedSequence``、``PCG64``、``Generator``、``default_rng`` 按名字直接禁止。

已知不覆盖：
- 属性赋值（``self.np = np``）；
- 作为函数参数传递；
- 元组解包；
- 从容器里取值；
- 动态导入的参数不是字面量，包括 ``name``、``fromlist``、``level``；
- 动态相对导入（``import_module`` 带 ``package=``，或 ``__import__`` 的 ``level`` 不为 0）；
- 别名按整个文件分析，不区分作用域，也不处理名字被重新绑定，因此可能误报，``hash`` 被重新绑定的情况同理。

M6：放开 ``scipy.stats`` 的导入；继续全面禁止 ``.rvs(``；另外禁止 ``bootstrap``、``permutation_test``、``monte_carlo_test``、``qmc``。``scipy.special`` 不在禁令内。

写法约定：类型注解统一写成 ``np.random.Generator``。``from numpy.random import Generator`` 会被拦截，这是有意为之。
"""

from __future__ import annotations

import ast
from pathlib import PurePosixPath

_BANNED_CALLS = frozenset({"SeedSequence", "PCG64", "Generator", "default_rng"})
_SEED_ATTRS = frozenset({"Generator", "SeedSequence"})
_WATCHED: dict[str, frozenset[str]] = {
    "numpy": frozenset({"random"}),
    "scipy": frozenset(),
    "os": frozenset({"urandom"}),
    "uuid": frozenset({"uuid1", "uuid4"}),
}
_STARSIM_ONLY = frozenset({"os", "uuid"})


def scan_source(text: str, filename: str) -> list[str]:
    """Return ``文件:行号: 原因`` for random-number violations in ``text``.

    numpy and scipy rules apply to every scanned file. ``os``, ``uuid``,
    ``secrets``, and ``hash()`` apply only when ``filename`` is under ``starsim/``.
    """
    if _is_rng_module(filename):
        return []
    try:
        tree = ast.parse(text)
    except SyntaxError as exc:
        return [f"{filename}:{exc.lineno or 1}: 无法解析"]
    parents = _parents(tree)
    imported = _import_module_aliases(tree)
    aliases = _package_aliases(tree, imported)
    functions, modules, package, star_rng = _stream_bindings(tree, filename)
    in_starsim = _in_starsim(filename)
    active = _active_packages(in_starsim)
    found: list[str] = _stats_violations(tree, filename, imported)
    for node in ast.walk(tree):
        reason = _reason(
            node,
            parents,
            aliases,
            imported,
            functions,
            modules,
            package,
            star_rng,
            in_starsim,
            active,
        )
        if reason is not None:
            found.append(f"{filename}:{node.lineno}: {reason}")
    return found


def _is_rng_module(filename: str) -> bool:
    """True only for the repo-relative path ``starsim/rng.py``.

    ``filename`` 必须是仓库相对的 POSIX 路径。如果传了绝对路径，``rng.py``
    不会被豁免，扫描它自己就会报错。这正是期望的失败方向：出错时大声报错，
    而不是悄悄放过。
    """
    return PurePosixPath(filename).parts == ("starsim", "rng.py")


def _in_starsim(filename: str) -> bool:
    parts = PurePosixPath(filename).parts
    return bool(parts) and parts[0] == "starsim"


def _active_packages(in_starsim: bool) -> dict[str, frozenset[str]]:
    if in_starsim:
        return _WATCHED
    return {pkg: attrs for pkg, attrs in _WATCHED.items() if pkg not in _STARSIM_ONLY}


def _parents(tree: ast.AST) -> dict[int, ast.AST]:
    parents: dict[int, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[id(child)] = node
    return parents


def _import_module_aliases(tree: ast.AST) -> set[str]:
    """Names bound by ``from importlib import import_module``."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or node.module != "importlib":
            continue
        for alias in node.names:
            if alias.name == "import_module":
                names.add(alias.asname or alias.name)
    return names


def _package_aliases(tree: ast.AST, imported: set[str]) -> dict[str, set[str]]:
    """Names that refer to a watched top-level package, propagated to a fixpoint."""
    aliases: dict[str, set[str]] = {pkg: set() for pkg in _WATCHED}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Import):
            continue
        for alias in node.names:
            root = alias.name.split(".", 1)[0]
            if root not in aliases:
                continue
            if alias.asname is None:
                # ``import P`` and ``import P.Y`` both bind the name ``P``.
                aliases[root].add(root)
            elif alias.name == root:
                aliases[root].add(alias.asname)
    changed = True
    while changed:
        changed = False
        for node in ast.walk(tree):
            targets, value = _simple_targets(node)
            if not targets or value is None:
                continue
            for pkg, names in aliases.items():
                if not _rhs_is_package(value, names, pkg, imported):
                    continue
                for target in targets:
                    if target not in names:
                        names.add(target)
                        changed = True
    return aliases


def _simple_targets(node: ast.AST) -> tuple[list[str], ast.AST | None]:
    """Every simple name on an ``Assign``, or the name of a valued ``AnnAssign``."""
    if isinstance(node, ast.Assign):
        names = [target.id for target in node.targets if isinstance(target, ast.Name)]
        if not names:
            return [], None
        return names, node.value
    if (
        isinstance(node, ast.AnnAssign)
        and node.value is not None
        and isinstance(node.target, ast.Name)
    ):
        return [node.target.id], node.value
    return [], None


def _rhs_is_package(value: ast.AST, names: set[str], pkg: str, imported: set[str]) -> bool:
    if isinstance(value, ast.Name) and value.id in names:
        return True
    if isinstance(value, ast.Call):
        return _dynamic_import_target(value, imported) == pkg
    return False


def _resolve_import_from(node: ast.ImportFrom, filename: str) -> str | None:
    """把 ImportFrom 解析成绝对模块名；无法解析时返回 None。"""
    if node.level == 0:
        return node.module
    package = PurePosixPath(filename).parts[:-1]
    if node.level > len(package):
        return None
    base = package[: len(package) - (node.level - 1)]
    base_name = ".".join(base)
    if node.module:
        return f"{base_name}.{node.module}"
    return base_name


def _stream_bindings(
    tree: ast.AST, filename: str
) -> tuple[set[str], set[str], bool, set[int]]:
    """Names bound to ``stream``, names bound to the rng module, and ``import starsim...``.

    ``package`` is true when the bare name ``starsim`` is bound, so
    ``starsim.rng.stream(...)`` resolves to this module. The last set holds
    ``id`` of ``ImportFrom`` nodes that star-import ``starsim.rng``.
    ``ImportFrom`` module names come only from ``_resolve_import_from``.
    """
    functions: set[str] = set()
    modules: set[str] = set()
    package = False
    star_rng: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.asname is None and (
                    alias.name == "starsim" or alias.name.startswith("starsim.")
                ):
                    package = True
                if alias.name == "starsim.rng" and alias.asname is not None:
                    modules.add(alias.asname)
            continue
        if not isinstance(node, ast.ImportFrom):
            continue
        resolved = _resolve_import_from(node, filename)
        if resolved == "starsim.rng":
            for alias in node.names:
                if alias.name == "*":
                    star_rng.add(id(node))
                elif alias.name == "stream":
                    functions.add(alias.asname or alias.name)
        elif resolved == "starsim":
            for alias in node.names:
                if alias.name == "rng":
                    modules.add(alias.asname or alias.name)
    return functions, modules, package, star_rng


def _forbidden_module(name: str, in_starsim: bool) -> bool:
    if (
        name == "random"
        or name.startswith("random.")
        or name.startswith("numpy.random")
    ):
        return True
    return in_starsim and (name == "secrets" or name.startswith("secrets."))


def _star_import_forbidden(module: str, in_starsim: bool) -> bool:
    if module in {"numpy", "scipy", "random"} or module.startswith("numpy.") or module.startswith("scipy."):
        return True
    return in_starsim and (module == "secrets" or module.startswith("secrets."))


def _reason(
    node: ast.AST,
    parents: dict[int, ast.AST],
    aliases: dict[str, set[str]],
    imported: set[str],
    functions: set[str],
    modules: set[str],
    package: bool,
    star_rng: set[int],
    in_starsim: bool,
    active: dict[str, frozenset[str]],
) -> str | None:
    if isinstance(node, ast.Import):
        for alias in node.names:
            if _forbidden_module(alias.name, in_starsim):
                return f"导入 {alias.name}"
        return None
    if isinstance(node, ast.ImportFrom):
        if id(node) in star_rng:
            return "星号导入 starsim.rng"
        module = node.module or ""
        if any(alias.name == "*" for alias in node.names) and _star_import_forbidden(module, in_starsim):
            return f"星号导入 {module}"
        if _forbidden_module(module, in_starsim):
            return f"导入 {module}"
        watched = active.get(module)
        if watched is not None:
            hit = sorted(alias.name for alias in node.names if alias.name in watched)
            if hit:
                return f"从 {module} 导入 {', '.join(hit)}"
        return None
    if isinstance(node, ast.Call):
        if _calls_rng_stream(node, functions, modules, package):
            return "调用 starsim.rng.stream"
        found = _getattr_watched(node, aliases, active, imported)
        if found is not None:
            pkg, attr = found
            return f"getattr 取 {pkg}.{attr}"
        literal = _dynamic_import_literal(node, imported)
        if literal is not None and _forbidden_module(literal, in_starsim):
            return f"动态导入 {literal}"
        called = _called_name(node.func)
        if called in _BANNED_CALLS:
            return f"调用 {called}"
        if called == "rvs":
            return "调用 rvs"
        if in_starsim and isinstance(node.func, ast.Name) and node.func.id == "hash":
            return "调用 hash"
        return None
    if isinstance(node, ast.Attribute):
        pkg = _watched_package(node.value, node.attr, aliases, active, imported)
        if pkg is None:
            return None
        if pkg == "numpy" and node.attr == "random" and _allows_numpy_random(node, parents):
            return None
        if isinstance(node.value, ast.Call):
            return f"动态导入 {pkg} 后取 {node.attr}"
        return f"使用 {pkg}.{node.attr}"
    return None


def _watched_package(
    receiver: ast.AST,
    attr: str,
    aliases: dict[str, set[str]],
    active: dict[str, frozenset[str]],
    imported: set[str],
) -> str | None:
    if isinstance(receiver, ast.Name):
        for pkg, names in aliases.items():
            if pkg in active and receiver.id in names and attr in active[pkg]:
                return pkg
        return None
    if isinstance(receiver, ast.Call):
        target = _dynamic_import_target(receiver, imported)
        if target in active and attr in active[target]:
            return target
    return None


def _getattr_watched(
    node: ast.Call,
    aliases: dict[str, set[str]],
    active: dict[str, frozenset[str]],
    imported: set[str],
) -> tuple[str, str] | None:
    func = node.func
    if not isinstance(func, ast.Name) or func.id != "getattr":
        return None
    target: ast.AST | None = node.args[0] if node.args else None
    attr_node: ast.AST | None = node.args[1] if len(node.args) >= 2 else None
    for keyword in node.keywords:
        if keyword.arg == "object":
            target = keyword.value
        elif keyword.arg == "name":
            attr_node = keyword.value
    if target is None or not isinstance(attr_node, ast.Constant) or not isinstance(attr_node.value, str):
        return None
    pkg = _watched_package(target, attr_node.value, aliases, active, imported)
    if pkg is None:
        return None
    return pkg, attr_node.value


def _calls_rng_stream(
    node: ast.Call,
    functions: set[str],
    modules: set[str],
    package: bool,
) -> bool:
    func = node.func
    if isinstance(func, ast.Name) and func.id in functions:
        return True
    if not isinstance(func, ast.Attribute) or func.attr != "stream":
        return False
    value = func.value
    if isinstance(value, ast.Name) and value.id in modules:
        return True
    return (
        package
        and isinstance(value, ast.Attribute)
        and value.attr == "rng"
        and isinstance(value.value, ast.Name)
        and value.value.id == "starsim"
    )


def _arg(node: ast.Call, index: int, keyword: str) -> ast.AST | None:
    if len(node.args) > index:
        return node.args[index]
    for item in node.keywords:
        if item.arg == keyword:
            return item.value
    return None


def _string_literal(node: ast.AST | None) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _is_import_module_call(node: ast.Call, imported: set[str]) -> bool:
    func = node.func
    if isinstance(func, ast.Name) and func.id in imported:
        return True
    return isinstance(func, ast.Attribute) and func.attr == "import_module"


def _is_dunder_import(node: ast.Call) -> bool:
    return isinstance(node.func, ast.Name) and node.func.id == "__import__"


def _level_is_absolute(level: ast.AST | None) -> bool:
    """Omitted or the literal integer 0. Any other level is relative or unknown."""
    if level is None:
        return True
    return isinstance(level, ast.Constant) and type(level.value) is int and level.value == 0


def _fromlist_kind(node: ast.AST | None) -> str:
    """``top`` returns the top-level package, ``full`` the named module, else unknown."""
    if node is None:
        return "top"
    if isinstance(node, ast.Constant) and node.value is None:
        return "top"
    if isinstance(node, (ast.List, ast.Tuple)):
        if not all(isinstance(elt, ast.Constant) for elt in node.elts):
            return "unknown"
        return "top" if len(node.elts) == 0 else "full"
    return "unknown"


def _dynamic_import_target(node: ast.Call, imported: set[str]) -> str | None:
    """Module name this dynamic import actually returns, or ``None`` if unknown.

    ``import_module`` returns the full module. ``__import__`` returns the
    top-level package unless ``fromlist`` is a non-empty literal list or tuple.
    """
    if _is_import_module_call(node, imported):
        if _arg(node, 1, "package") is not None:
            return None
        return _string_literal(_arg(node, 0, "name"))
    if not _is_dunder_import(node):
        return None
    name = _string_literal(_arg(node, 0, "name"))
    if name is None or not _level_is_absolute(_arg(node, 4, "level")):
        return None
    kind = _fromlist_kind(_arg(node, 3, "fromlist"))
    if kind == "top":
        return name.split(".", 1)[0]
    if kind == "full":
        return name
    return None


def _dynamic_import_literal(node: ast.Call, imported: set[str]) -> str | None:
    """String passed as the module name, whether or not the return value is known."""
    if not (_is_import_module_call(node, imported) or _is_dunder_import(node)):
        return None
    return _string_literal(_arg(node, 0, "name"))


def _allows_numpy_random(node: ast.Attribute, parents: dict[int, ast.AST]) -> bool:
    """Allow ``X.random.Generator`` and ``X.random.SeedSequence`` when uncalled.

    The parent of ``X.random`` must be that attribute. The attribute itself
    must not be the function of a ``Call``. ``isinstance(x, np.random.SeedSequence)``
    and annotations such as ``rng: np.random.Generator`` pass. ``np.random.Generator(...)``
    does not, because the grandparent call's function is the attribute.
    """
    parent = parents.get(id(node))
    if not isinstance(parent, ast.Attribute) or parent.attr not in _SEED_ATTRS:
        return False
    grand = parents.get(id(parent))
    if isinstance(grand, ast.Call) and grand.func is parent:
        return False
    return True


def _called_name(func: ast.AST) -> str | None:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


_STATS_RANDOM = frozenset({'rvs', 'bootstrap', 'permutation_test', 'monte_carlo_test', 'qmc'})


def _stats_violations(tree, filename, imported):
    """Resolve stats imports, simple aliases, getattr and literal dynamic imports."""
    bindings = {}

    def resolve(node):
        if isinstance(node, ast.Name):
            return bindings.get(node.id)
        if isinstance(node, ast.Attribute):
            base = resolve(node.value)
            return f'{base}.{node.attr}' if base else None
        if isinstance(node, ast.Call):
            dynamic = _dynamic_import_target(node, imported)
            if dynamic:
                return dynamic
            if isinstance(node.func, ast.Name) and node.func.id == 'getattr':
                base = resolve(_arg(node, 0, 'object'))
                attr = _string_literal(_arg(node, 1, 'name'))
                if base and attr:
                    return f'{base}.{attr}'
        return None

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for item in node.names:
                bindings[item.asname or item.name.split('.')[0]] = item.name if item.asname else item.name.split('.')[0]
        elif isinstance(node, ast.ImportFrom):
            for item in node.names:
                bindings[item.asname or item.name] = f'{node.module}.{item.name}'
    changed = True
    while changed:
        changed = False
        for node in ast.walk(tree):
            names, value = _simple_targets(node)
            target = resolve(value)
            if target:
                for name in names:
                    if name not in bindings:
                        bindings[name] = target
                        changed = True
    def banned(path):
        return path and path.startswith('scipy.stats.') and bool(set(path.split('.')[2:]) & _STATS_RANDOM)

    found = []
    for node in ast.walk(tree):
        hit = banned(resolve(node))
        if isinstance(node, ast.Import):
            hit = hit or any(banned(a.name) for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            hit = hit or any(banned(f'{node.module}.{a.name}') for a in node.names)
        if hit:
            found.append(f'{filename}:{node.lineno}: scipy.stats 随机接口')
    return found
