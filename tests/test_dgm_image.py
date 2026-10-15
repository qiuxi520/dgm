"""镜像存在时复用，仅在缺失或显式要求时构建。"""
from unittest.mock import Mock

import docker
import pytest

from utils.docker_utils import build_dgm_container


def test_existing_latest_image_is_reused(monkeypatch):
    monkeypatch.setattr('utils.docker_utils.safe_log', lambda *args: None)
    client = Mock()
    container = build_dgm_container(client, container_name='test-container')
    client.images.get.assert_called_once_with('dgm:latest')
    client.images.build.assert_not_called()
    client.images.list.assert_not_called()
    client.containers.run.assert_called_once_with(
        image='dgm:latest', name='test-container', detach=True,
    )
    assert container is client.containers.run.return_value


@pytest.mark.parametrize('force_rebuild', [False, True])
def test_builds_when_missing_or_forced(monkeypatch, force_rebuild):
    monkeypatch.setattr('utils.docker_utils.safe_log', lambda *args: None)
    client = Mock()
    client.images.get.side_effect = docker.errors.ImageNotFound('missing')
    client.images.build.return_value = (Mock(), [])
    container = build_dgm_container(client, repo_path='repo', force_rebuild=force_rebuild)
    client.images.build.assert_called_once_with(path='repo', tag='dgm:latest', rm=True)
    if force_rebuild:
        client.images.get.assert_not_called()
    assert container is client.containers.run.return_value
