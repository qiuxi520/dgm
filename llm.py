# Code adapted from https://github.com/SakanaAI/AI-Scientist/blob/main/ai_scientist/llm.py.
import json
# 中文导读：基础模型客户端与纯文本请求封装，主要用于失败诊断和报告整理。
# 这里提供 API 调用而非自进化算法；真正执行编辑工具的循环在 llm_withtools.py。
import os
import re
from pathlib import Path

import anthropic
import backoff
import openai
from dotenv import load_dotenv

# 调试器和命令行均读取项目配置；显式设置的环境变量优先。
load_dotenv(Path(__file__).resolve().parent / '.env', override=False)

MAX_OUTPUT_TOKENS = 4096
DEFAULT_MODEL = os.getenv("DEEPSEEK_MODEL") or "deepseek-v4-flash"


def deepseek_request_options():
    """纯文本与工具调用使用相同的 DeepSeek 配置。"""
    return {
        "max_tokens": int(os.getenv("DEEPSEEK_MAX_TOKENS") or "8192"),
        "extra_body": {
            "thinking": {"type": os.getenv("DEEPSEEK_THINKING") or "enabled"},
        },
    }


def llm_environment():
    """将模型配置显式传给容器；.env 不进入 Docker 镜像。"""
    names = (
        "DEEPSEEK_API_KEY", "DEEPSEEK_BASE_URL", "DEEPSEEK_MODEL",
        "DEEPSEEK_THINKING", "DEEPSEEK_MAX_TOKENS",
        "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "AWS_REGION",
        "AWS_REGION_NAME", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY",
    )
    return {name: os.environ[name] for name in names if os.getenv(name)}

AVAILABLE_LLMS = [
    # Anthropic models
    "claude-3-5-sonnet-20240620",
    "claude-3-5-sonnet-20241022",
    # OpenAI models
    "gpt-4o-mini-2024-07-18",
    "gpt-4o-2024-05-13",
    "gpt-4o-2024-08-06",
    "o1-preview-2024-09-12",
    "o1-mini-2024-09-12",
    "o1-2024-12-17",
    "o3-mini-2025-01-31",
    # OpenRouter models
    "llama3.1-405b",
    # Anthropic Claude models via Amazon Bedrock
    "bedrock/anthropic.claude-3-sonnet-20240229-v1:0",
    "bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0",
    "bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0",
    "bedrock/anthropic.claude-3-haiku-20240307-v1:0",
    "bedrock/anthropic.claude-3-opus-20240229-v1:0",
    "bedrock/us.anthropic.claude-3-5-sonnet-20241022-v2:0",
    # Anthropic Claude models Vertex AI
    "vertex_ai/claude-3-opus@20240229",
    "vertex_ai/claude-3-5-sonnet@20240620",
    "vertex_ai/claude-3-5-sonnet-v2@20241022",
    "vertex_ai/claude-3-sonnet@20240229",
    "vertex_ai/claude-3-haiku@20240307",
    # DeepSeek models
    "deepseek-chat",
    "deepseek-coder",
    "deepseek-reasoner",
    "deepseek-v4-flash",
]

def create_client(model: str):
    # 返回 (客户端, 供应商接受的模型名)；例如 bedrock/ 前缀只用于选择客户端，
    # 发请求前会去掉路由前缀。环境变量提供凭据，不在源码中保存密钥。
    """
    Create and return an LLM client based on the specified model.
    Args:
        model (str): The name of the model to use.
    Returns:
        Tuple[Any, str]: A tuple containing the client instance and the client model name.
    """
    if model.startswith("claude-"):
        print(f"使用 Anthropic API，模型：{model}.")
        return anthropic.Anthropic(), model
    elif model.startswith("bedrock") and "claude" in model:
        client_model = model.split("/")[-1]
        print(f"使用 Amazon Bedrock，模型：{client_model}.")
        client = anthropic.AnthropicBedrock(
            aws_access_key=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            aws_region=os.getenv("AWS_REGION_NAME"),
        )
        return client, client_model
    elif model.startswith("vertex_ai") and "claude" in model:
        client_model = model.split("/")[-1]
        print(f"使用 Vertex AI，模型：{client_model}.")
        return anthropic.AnthropicVertex(), client_model
    elif 'gpt' in model or model.startswith("o1-") or model.startswith("o3-"):
        print(f"使用 OpenAI API，模型：{model}.")
        return openai.OpenAI(), model
    elif model.startswith("deepseek-"):
        print(f"使用 DeepSeek API，模型：{model}。")
        client = openai.OpenAI(
            api_key=os.environ["DEEPSEEK_API_KEY"],
            base_url=os.getenv("DEEPSEEK_BASE_URL") or "https://api.deepseek.com"
        )
        return client, model
    elif model == "llama3.1-405b":
        print(f"使用 OpenAI API，模型：{model}.")
        client = openai.OpenAI(
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url="https://openrouter.ai/api/v1"
        ), model
    else:
        raise ValueError(f"Model {model} not supported.")

# Get N responses from a single message, used for ensembling.
@backoff.on_exception(backoff.expo, (openai.RateLimitError, openai.APITimeoutError))
def get_batch_responses_from_llm(
        msg,
        client,
        model,
        system_message,
        print_debug=False,
        msg_history=None,
        temperature=0.75,
        n_responses=1,
):
    if msg_history is None:
        msg_history = []

    if model in [
        "gpt-4o-2024-05-13",
        "gpt-4o-mini-2024-07-18",
        "gpt-4o-2024-08-06",
    ]:
        new_msg_history = msg_history + [{"role": "user", "content": msg}]
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                *new_msg_history,
            ],
            temperature=temperature,
            max_tokens=MAX_OUTPUT_TOKENS,
            n=n_responses,
            stop=None,
            seed=0,
        )
        content = [r.message.content for r in response.choices]
        new_msg_history = [
            new_msg_history + [{"role": "assistant", "content": c}] for c in content
        ]
    elif model == "llama-3-1-405b-instruct":
        new_msg_history = msg_history + [{"role": "user", "content": msg}]
        response = client.chat.completions.create(
            model="meta-llama/llama-3.1-405b-instruct",
            messages=[
                {"role": "system", "content": system_message},
                *new_msg_history,
            ],
            temperature=temperature,
            max_tokens=MAX_OUTPUT_TOKENS,
            n=n_responses,
            stop=None,
        )
        content = [r.message.content for r in response.choices]
        new_msg_history = [
            new_msg_history + [{"role": "assistant", "content": c}] for c in content
        ]
    else:
        content, new_msg_history = [], []
        for _ in range(n_responses):
            c, hist = get_response_from_llm(
                msg,
                client,
                model,
                system_message,
                print_debug=False,
                msg_history=None,
                temperature=temperature,
            )
            content.append(c)
            new_msg_history.append(hist)

    if print_debug:
        print()
        print("*" * 20 + " 模型对话开始 " + "*" * 20)
        for j, msg in enumerate(new_msg_history[0]):
            print(f'{j}, {msg["role"]}: {msg["content"]}')
        print(content)
        print("*" * 21 + " 模型对话结束 " + "*" * 21)
        print()

    return content, new_msg_history

@backoff.on_exception(
    backoff.expo,
    (openai.RateLimitError, openai.APITimeoutError, anthropic.RateLimitError, anthropic.APIStatusError),
    max_time=120,
)
def get_response_from_llm(
        msg,
        client,
        model,
        system_message,
        print_debug=False,
        msg_history=None,
        temperature=0.7,
):
    # 返回 (响应文本, 追加后的对话历史)，各供应商的历史格式在分支中适配。
    # 默认诊断走 DeepSeek；保留 o1/o3 的兼容分支（将 system_message 拼接进 user）。
    if msg_history is None:
        msg_history = []

    if "claude" in model:
        new_msg_history = msg_history + [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": msg,
                    }
                ],
            }
        ]
        response = client.messages.create(
            model=model,
            max_tokens=MAX_OUTPUT_TOKENS,
            temperature=temperature,
            system=system_message,
            messages=new_msg_history,
        )
        content = response.content[0].text
        new_msg_history = new_msg_history + [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": content,
                    }
                ],
            }
        ]
    elif model.startswith("gpt-4o-"):
        new_msg_history = msg_history + [{"role": "user", "content": msg}]
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                *new_msg_history,
            ],
            temperature=temperature,
            max_tokens=MAX_OUTPUT_TOKENS,
            n=1,
            stop=None,
            seed=0,
        )
        content = response.choices[0].message.content
        new_msg_history = new_msg_history + [{"role": "assistant", "content": content}]
    elif model.startswith("o1-") or model.startswith("o3-"):
        new_msg_history = msg_history + [{"role": "user", "content": system_message + msg}]
        response = client.chat.completions.create(
            model=model,
            messages=[
                # {"role": "user", "content": system_message},
                *new_msg_history,
            ],
            temperature=1,
            # max_completion_tokens=MAX_OUTPUT_TOKENS,
            n=1,
            # stop=None,
            seed=0,
        )
        content = response.choices[0].message.content
        new_msg_history = new_msg_history + [{"role": "assistant", "content": content}]
    elif model.startswith("deepseek-") and model != "deepseek-reasoner":
        new_msg_history = msg_history + [{"role": "user", "content": msg}]
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                *new_msg_history,
            ],
            temperature=temperature,
            **deepseek_request_options(),
        )
        content = response.choices[0].message.content
        new_msg_history = new_msg_history + [{"role": "assistant", "content": content}]
    elif model in ["deepseek-reasoner"]:
        new_msg_history = msg_history + [{"role": "user", "content": msg}]
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                *new_msg_history,
            ],
            n=1,
            stop=None,
        )
        content = response.choices[0].message.content
        new_msg_history = new_msg_history + [{"role": "assistant", "content": content}]
        reasoning_content = response.choices[0].message.reasoning_content
    elif model.startswith("llama3.1-"):
        llama_size = model.split("-")[-1]
        client_model = f"meta-llama/llama-3.1-{llama_size}-instruct"
        new_msg_history = msg_history + [{"role": "user", "content": msg}]
        response = client.chat.completions.create(
            model=client_model,
            messages=[
                {"role": "system", "content": system_message},
                *new_msg_history,
            ],
            temperature=temperature,
            max_tokens=MAX_OUTPUT_TOKENS,
            n=1,
            stop=None,
        )
        content = response.choices[0].message.content
        new_msg_history = new_msg_history + [{"role": "assistant", "content": content}]
        resoning_content = response.choices[0].message.reasoning_content
    else:
        raise ValueError(f"Model {model} not supported.")
    if print_debug:
        print()
        print("*" * 20 + " 模型对话开始 " + "*" * 20)
        print(f'用户：{new_msg_history[-2]["content"]}')
        print(f'助手：{new_msg_history[-1]["content"]}')
        print("*" * 21 + " 模型对话结束 " + "*" * 21)
        print()
    return content, new_msg_history

def extract_json_between_markers(llm_output):
    # 优先解析 ```json 代码块，再尝试从普通文本中抽取 JSON，并清理非法控制字符。
    # 解析失败返回 None，由诊断函数重试；兜底非贪婪正则不能可靠解析任意嵌套对象。
    inside_json_block = False
    json_lines = []
    
    # Split the output into lines and iterate
    for line in llm_output.split('\n'):
        striped_line = line.strip()
        
        # Check for start of JSON code block
        if striped_line.startswith("```json"):
            inside_json_block = True
            continue
        
        # Check for end of code block
        if inside_json_block and striped_line.startswith("```"):
            # We've reached the closing triple backticks.
            inside_json_block = False
            break
        
        # If we're inside the JSON block, collect the lines
        if inside_json_block:
            json_lines.append(line)
    
    # If we never found a JSON code block, fallback to any JSON-like content
    if not json_lines:
        # Fallback: Try a regex that finds any JSON-like object in the text
        fallback_pattern = r"\{.*?\}"
        matches = re.findall(fallback_pattern, llm_output, re.DOTALL)
        for candidate in matches:
            candidate = candidate.strip()
            if candidate:
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    # Attempt to clean control characters and re-try
                    candidate_clean = re.sub(r"[\x00-\x1F\x7F]", "", candidate)
                    try:
                        return json.loads(candidate_clean)
                    except json.JSONDecodeError:
                        continue
        return None

    # Join all lines in the JSON block into a single string
    json_string = "\n".join(json_lines).strip()
    
    # Try to parse the collected JSON lines
    try:
        return json.loads(json_string)
    except json.JSONDecodeError:
        # Attempt to remove invalid control characters and re-parse
        json_string_clean = re.sub(r"[\x00-\x1F\x7F]", "", json_string)
        try:
            return json.loads(json_string_clean)
        except json.JSONDecodeError:
            return None
