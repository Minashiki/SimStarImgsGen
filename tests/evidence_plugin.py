"""Record original test identities even when setup fails."""
import json
import os
from pathlib import Path
from types import ModuleType

import pytest


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(items):
    for item in items:
        item.user_properties.append(('node_id', item.nodeid))


def pytest_deselected(items):
    global deselected
    deselected += len(items)


deselected = 0


def pytest_sessionfinish(session):
    target = os.environ.get('STARSIM_COLLECTION')
    if target:
        config = session.config
        plugins = sorted((d.project_name, d.version) for _, d in
                         config.pluginmanager.list_plugin_distinfo())
        Path(target).write_text(json.dumps(dict(
            total=session.testscollected+deselected, deselected=deselected,
            node_ids=[item.nodeid for item in session.items],
            rootdir=str(config.rootpath), inifile=str(config.inipath),
            plugins=plugins, pytest_version=pytest.__version__,
            active_plugins=sorted((name, getattr(plugin, '__version__',
                pytest.__version__ if plugin.__name__.startswith('_pytest.') else 'repository'))
                for name, plugin in config.pluginmanager.list_name_plugin()
                if isinstance(plugin, ModuleType)))), encoding='utf-8')
