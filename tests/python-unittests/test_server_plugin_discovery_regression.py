# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from pathlib import Path

import pytest

from experimental.server.runtime import engine


@pytest.fixture
def checkout(tmp_path, monkeypatch):
    root = tmp_path / "checkout"
    module = root / "experimental" / "server" / "runtime" / "engine.py"
    module.parent.mkdir(parents=True)
    module.touch()
    monkeypatch.setattr(engine, "__file__", str(module))
    monkeypatch.delenv("EDGELLM_PLUGIN_PATH", raising=False)
    monkeypatch.delenv("BUILD_DIR", raising=False)
    work = tmp_path / "elsewhere"
    work.mkdir()
    monkeypatch.chdir(work)
    return root


def make_plugin(directory):
    directory.mkdir(parents=True, exist_ok=True)
    plugin = directory / engine._PLUGIN_LIB_NAME
    plugin.touch()
    return plugin


@pytest.mark.parametrize("subdir", ["", "core", "lib"])
def test_default_build_locations_from_another_directory(checkout, subdir):
    plugin = make_plugin(checkout / "build" / subdir)
    engine._ensure_plugin_path()
    assert os.environ.get("EDGELLM_PLUGIN_PATH") == str(plugin)


@pytest.mark.parametrize("subdir", ["", "core", "lib"])
def test_custom_build_locations(checkout, tmp_path, monkeypatch, subdir):
    build_dir = tmp_path / "custom-build"
    plugin = make_plugin(build_dir / subdir)
    monkeypatch.setenv("BUILD_DIR", str(build_dir))
    engine._ensure_plugin_path()
    assert os.environ.get("EDGELLM_PLUGIN_PATH") == str(plugin)


def test_explicit_plugin_override_is_preserved(checkout, monkeypatch):
    make_plugin(checkout / "build")
    explicit = "/configured/elsewhere/libNvInfer_edgellm_plugin.so"
    monkeypatch.setenv("EDGELLM_PLUGIN_PATH", explicit)
    engine._ensure_plugin_path()
    assert os.environ["EDGELLM_PLUGIN_PATH"] == explicit


def test_custom_build_precedes_default_build(checkout, tmp_path, monkeypatch):
    make_plugin(checkout / "build" / "core")
    custom = tmp_path / "custom-build"
    plugin = make_plugin(custom)
    monkeypatch.setenv("BUILD_DIR", str(custom))
    engine._ensure_plugin_path()
    assert os.environ.get("EDGELLM_PLUGIN_PATH") == str(plugin)


def test_legacy_location_precedence_is_preserved(checkout):
    core = make_plugin(checkout / "build" / "core")
    make_plugin(checkout / "build" / "lib")
    make_plugin(checkout / "build")
    engine._ensure_plugin_path()
    assert os.environ.get("EDGELLM_PLUGIN_PATH") == str(core)


def test_missing_custom_build_falls_back_to_default(checkout, tmp_path, monkeypatch):
    plugin = make_plugin(checkout / "build")
    monkeypatch.setenv("BUILD_DIR", str(tmp_path / "not-built"))
    engine._ensure_plugin_path()
    assert os.environ.get("EDGELLM_PLUGIN_PATH") == str(plugin)


def test_relative_build_dir_is_resolved(checkout, monkeypatch):
    plugin = make_plugin(Path.cwd() / "relative-build")
    monkeypatch.setenv("BUILD_DIR", "relative-build")
    engine._ensure_plugin_path()
    assert os.environ.get("EDGELLM_PLUGIN_PATH") == str(plugin)


def test_build_dir_expands_home(checkout, tmp_path, monkeypatch):
    plugin = make_plugin(tmp_path / "build-at-home")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("BUILD_DIR", "~/build-at-home")
    engine._ensure_plugin_path()
    assert os.environ.get("EDGELLM_PLUGIN_PATH") == str(plugin)


def test_unrelated_current_directory_is_not_searched(checkout):
    make_plugin(Path.cwd() / "build")
    engine._ensure_plugin_path()
    assert "EDGELLM_PLUGIN_PATH" not in os.environ


def test_directory_named_like_library_is_ignored(checkout):
    (checkout / "build" / engine._PLUGIN_LIB_NAME).mkdir(parents=True)
    engine._ensure_plugin_path()
    assert "EDGELLM_PLUGIN_PATH" not in os.environ


def test_absent_plugin_leaves_environment_unchanged(checkout):
    engine._ensure_plugin_path()
    assert "EDGELLM_PLUGIN_PATH" not in os.environ
