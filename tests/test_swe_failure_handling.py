"""评估环境失败必须暴露原始原因，且不会报告成功。"""
from types import SimpleNamespace
from unittest.mock import Mock

import docker
import pytest

import swe_bench.harness as harness
import swe_bench.report as report
from swe_bench.build_config import configure_build_client
from swe_bench.utils import setup_logger


def test_logger_supports_swebench_build_errors(tmp_path):
    from swebench.harness.docker_build import BuildImageError
    path = tmp_path / 'docker.log'
    logger = setup_logger(path)
    error = BuildImageError('missing-image', '环境镜像不存在', logger)
    assert '环境镜像不存在' in str(error)
    assert str(path) in str(error)


def test_failed_container_creation_preserves_error(tmp_path, monkeypatch, capsys):
    client = Mock()
    monkeypatch.setattr(harness.docker, 'from_env', lambda: client)
    monkeypatch.setattr(harness, 'configure_build_client', lambda value: value)
    monkeypatch.setattr(harness, 'make_test_spec', lambda _: SimpleNamespace(
        repo_script_list=[], get_instance_container_name=lambda _: 'test-container',
    ))
    monkeypatch.setattr(harness, 'remove_existing_container', lambda *args: None)
    cleanup = Mock()
    monkeypatch.setattr(harness, 'cleanup_container', cleanup)
    monkeypatch.setattr(harness, 'build_container', Mock(side_effect=RuntimeError('原始构建错误')))
    result = harness.process_entry(
        {'instance_id': 'test', 'problem_statement': 'test', 'base_commit': 'test'},
        tmp_path, 'model', [],
    )
    assert result['error'] == '原始构建错误'
    cleanup.assert_not_called()
    client.close.assert_called_once()
    assert 'referenced before assignment' not in capsys.readouterr().out


def test_failed_env_build_stops_before_agent(tmp_path, monkeypatch):
    monkeypatch.setattr(harness, 'load_dataset', lambda _: {'test': [{'instance_id': 'test'}]})
    client = Mock()
    monkeypatch.setattr(harness.docker, 'from_env', lambda: client)
    monkeypatch.setattr(harness, 'configure_build_client', lambda value: value)
    build = Mock(return_value=([], ['failed-env']))
    monkeypatch.setattr(harness, 'build_env_images', build)
    process = Mock()
    monkeypatch.setattr(harness, 'process_entry', process)
    with pytest.raises(RuntimeError, match='failed-env'):
        harness.harness(pred_dname=tmp_path, max_workers=5)
    assert build.call_args.kwargs['max_workers'] == 1
    process.assert_not_called()


def test_report_thread_errors_propagate(tmp_path, monkeypatch):
    monkeypatch.setattr(report, 'load_predictions', lambda _: {})
    with pytest.raises(RuntimeError, match='为空'):
        report.make_report([tmp_path], output_dir=tmp_path)


def test_build_proxy_reaches_loopback_via_host_network(monkeypatch):
    monkeypatch.delenv('SWE_BUILD_NETWORK', raising=False)
    monkeypatch.setenv('HTTP_PROXY', 'http://127.0.0.1:7897')
    client = Mock()
    original = client.api.build
    original.return_value = iter([{'stream': 'built'}])
    configure_build_client(client)
    assert list(client.api.build(tag='test', decode=True)) == [{'stream': 'built'}]
    options = original.call_args.kwargs
    assert options['network_mode'] == 'host'
    assert options['buildargs']['HTTP_PROXY'] == 'http://127.0.0.1:7897'
    assert 'DEEPSEEK_API_KEY' not in options['buildargs']


def test_default_swe_evaluation_is_serial(tmp_path, monkeypatch):
    import self_improve_step
    run = Mock(return_value=[tmp_path / 'predictions'])
    make_report = Mock()
    monkeypatch.setattr(self_improve_step, 'harness', run)
    monkeypatch.setattr(self_improve_step, 'make_report', make_report)
    monkeypatch.setattr(self_improve_step, 'get_all_performance', lambda *a, **k: ([], {'accuracy_score': 0}))
    monkeypatch.setattr(self_improve_step, 'safe_log', lambda *args: None)
    self_improve_step.run_harness_swe('task', 'run', [], 1, str(tmp_path), {}, 'run', None, ['task'], None)
    assert run.call_args.kwargs['max_workers'] == 1
    assert run.call_args.kwargs['num_evals_parallel'] == 1
    assert make_report.call_args.kwargs['num_eval_procs'] == 1


def test_official_eval_does_not_change_process_directory(tmp_path, monkeypatch):
    from pathlib import Path
    run = Mock()
    monkeypatch.setattr(report.subprocess, 'run', run)
    original = Path.cwd()
    report.run_evals('predictions.jsonl', 'run', 'dataset', str(original), str(tmp_path), num_eval_procs=1)
    assert Path.cwd() == original
    assert run.call_args.kwargs['cwd'] == str(tmp_path)
