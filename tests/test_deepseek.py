"""无需网络的 DeepSeek 接口与工具对话回归测试。"""
import copy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from openai.types.chat import ChatCompletion

import llm
import llm_withtools
from utils.eval_utils import msg_history_to_report


MODEL = 'deepseek-v4-flash'


def completion(content=None, calls=None, reasoning='思考内容', finish='stop'):
    return ChatCompletion.model_validate({
        'id': 'test', 'object': 'chat.completion', 'created': 0, 'model': MODEL,
        'choices': [{
            'index': 0, 'finish_reason': finish,
            'message': {
                'role': 'assistant', 'content': content,
                'tool_calls': calls, 'reasoning_content': reasoning,
            },
        }],
    })


def call(identifier, arguments):
    return {'id': identifier, 'type': 'function',
            'function': {'name': 'test_tool', 'arguments': arguments}}


def test_client_uses_deepseek_configuration(monkeypatch):
    monkeypatch.setenv('DEEPSEEK_API_KEY', 'test-key')
    monkeypatch.setenv('DEEPSEEK_BASE_URL', 'https://api.deepseek.com')
    factory = Mock(return_value=object())
    monkeypatch.setattr(llm.openai, 'OpenAI', factory)
    client, model = llm.create_client(MODEL)
    assert client is factory.return_value
    assert model == MODEL
    factory.assert_called_once_with(api_key='test-key', base_url='https://api.deepseek.com')


def test_diagnosis_accepts_v4_model(monkeypatch):
    monkeypatch.setenv('DEEPSEEK_THINKING', 'enabled')
    create = Mock(return_value=completion('诊断结果'))
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    text, history = llm.get_response_from_llm('任务', client, MODEL, '诊断说明')
    assert text == '诊断结果'
    assert history[-1] == {'role': 'assistant', 'content': '诊断结果'}
    assert create.call_args.kwargs['messages'][0]['role'] == 'system'
    assert create.call_args.kwargs['extra_body']['thinking']['type'] == 'enabled'


@pytest.mark.parametrize('bad_arguments', [False, True])
def test_native_tools_keep_reasoning_and_every_call_id(monkeypatch, bad_arguments):
    requests = []
    responses = iter([
        completion(calls=[call('one', 'not-json' if bad_arguments else '{"value":1}'),
                          call('two', '{"value":2}')], finish='tool_calls'),
        completion('完成'),
    ])

    def create(**kwargs):
        requests.append(copy.deepcopy(kwargs))
        return next(responses)

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    tool_function = Mock(side_effect=lambda value: value * 2)
    monkeypatch.setattr(llm_withtools, 'create_client', lambda model: (client, model))
    monkeypatch.setattr(llm_withtools, 'load_all_tools', lambda **kwargs: [{
        'info': {'name': 'test_tool', 'description': 'test', 'input_schema': {
            'type': 'object', 'properties': {'value': {'type': 'integer'}},
            'required': ['value'],
        }}, 'function': tool_function,
    }])
    history = llm_withtools.chat_with_agent('测试', model=MODEL, logging=lambda _: None)
    assert history[-1]['content'] == '完成'
    assert history[1]['reasoning_content'] == '思考内容'
    assert [message['tool_call_id'] for message in history if message['role'] == 'tool'] == ['one', 'two']
    assert requests[1]['messages'][1]['reasoning_content'] == '思考内容'
    assert requests[0]['tools'][0]['function']['name'] == 'test_tool'
    assert 'tool_choice' not in requests[0]
    assert 'input' not in requests[0]
    assert tool_function.call_count == (1 if bad_arguments else 2)
    if bad_arguments:
        assert '工具参数解析失败' in history[2]['content']
    # 下一次用户请求继续传回原生历史，包括上一轮最终响应的推理内容。
    responses = iter([completion('下一轮完成')])
    continued = llm_withtools.chat_with_agent('继续', model=MODEL, msg_history=history, logging=lambda _: None)
    assert requests[-1]['messages'][-2]['reasoning_content'] == '思考内容'
    assert continued[:len(history)] == history


def test_tool_results_still_feed_test_report_parser(monkeypatch):
    monkeypatch.setattr('utils.eval_utils.parse_eval_output', lambda *args: {'test': 'PASSED'})
    history = [{'role': 'tool', 'tool_call_id': 'one', 'content': 'test PASSED'}]
    assert msg_history_to_report('dgm', history, model=MODEL) == {'test': 'PASSED'}


def test_container_receives_model_and_key(monkeypatch):
    monkeypatch.setenv('DEEPSEEK_API_KEY', 'test-key')
    monkeypatch.setenv('DEEPSEEK_MODEL', MODEL)
    monkeypatch.setenv('DEEPSEEK_BASE_URL', 'https://api.deepseek.com')
    environment = llm.llm_environment()
    assert environment['DEEPSEEK_API_KEY'] == 'test-key'
    assert environment['DEEPSEEK_MODEL'] == MODEL
    assert environment['DEEPSEEK_BASE_URL'] == 'https://api.deepseek.com'


def test_all_dgm_roles_share_default_model():
    import self_improve_step
    assert llm_withtools.CLAUDE_MODEL == llm.DEFAULT_MODEL
    assert llm_withtools.OPENAI_MODEL == llm.DEFAULT_MODEL
    assert self_improve_step.diagnose_model == llm.DEFAULT_MODEL
