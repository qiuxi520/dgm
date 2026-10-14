"""验证环境文件加载与日志中文化后的兼容性。"""
import importlib.util
from pathlib import Path

import pytest

import DGM_outer
import llm
from prompts.self_improvement_prompt import read_mdlog_file


@pytest.mark.parametrize("prefix", [
    "Error in get_response_withtools: ",
    "工具调用模型请求失败：",
])
@pytest.mark.parametrize("repetitions, expected", [(1, False), (2, True)])
def test_context_limit_detection(monkeypatch, prefix, repetitions, expected):
    error = prefix + "Error code: 400 - {'message': 'Input is too long for requested model.'}"
    log = "\n".join([error] * repetitions)
    monkeypatch.setattr(
        DGM_outer, "find_selfimprove_eval_logs",
        lambda *args, **kwargs: ([log], None, None, None),
    )
    assert DGM_outer.any_exceeding_context_length("unused", "initial", ["task"]) is expected


def test_log_filter_accepts_both_languages(tmp_path):
    path = tmp_path / "chat_history.md"
    content = "正常内容\nError in get_response_withtools: legacy\n工具调用模型请求失败：new\n保留内容\n"
    path.write_text(content, encoding="utf-8")
    assert read_mdlog_file(path) == "正常内容\n保留内容"
    assert read_mdlog_file(path, filter=False) == content.strip()


@pytest.mark.parametrize("existing", [None, "explicit-environment-value"])
def test_env_file_loaded_from_module_directory(tmp_path, monkeypatch, existing):
    # 使用临时模块和测试凭据，不读取或修改项目中的真实密钥。
    module_path = tmp_path / "llm.py"
    module_path.write_text(Path(llm.__file__).read_text(encoding="utf-8"), encoding="utf-8")
    (tmp_path / ".env").write_text("OPENAI_API_KEY=test-env-file-value\n", encoding="utf-8")
    if existing is None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    else:
        monkeypatch.setenv("OPENAI_API_KEY", existing)
    monkeypatch.chdir(tmp_path.parent)
    spec = importlib.util.spec_from_file_location("llm_configuration_test", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.os.getenv("OPENAI_API_KEY") == (existing or "test-env-file-value")
