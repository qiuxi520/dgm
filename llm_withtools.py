# 中文导读：代理的“观察 → 工具行动 → 获取结果 → 再决策”循环。
# 模型给出结构化工具请求，Python 调用真实函数并回传结果；模型文本不会自动变成文件修改。
# DeepSeek 使用 Chat Completions 原生函数调用；Claude 使用 tool_use/tool_result；
# o3 使用 Responses，其余模型走文本标签解析。工具通过 tools.load_all_tools 动态发现。
import ast
import json
import re
import anthropic
import backoff
import openai
import copy

from llm import DEFAULT_MODEL, create_client, deepseek_request_options, get_response_from_llm
from prompts.tooluse_prompt import get_tooluse_prompt
from tools import load_all_tools

# 保留旧常量名供历史代理补丁使用，两个编程入口统一使用配置中的模型。
CLAUDE_MODEL = DEFAULT_MODEL
OPENAI_MODEL = DEFAULT_MODEL

def process_tool_call(tools_dict, tool_name, tool_input):
    # 用模型返回的工具名查注册表，再以关键字参数执行对应函数。
    # 将工具异常转为字符串反馈给模型，使其有机会调整下一步操作。
    try:
        if tool_name in tools_dict:
            return tools_dict[tool_name]['function'](**tool_input)
        else:
            return f"Error: Tool '{tool_name}' not found"
    except Exception as e:
        return f"Error executing tool '{tool_name}': {str(e)}"

@backoff.on_exception(
    backoff.expo,
    (openai.RateLimitError, openai.APITimeoutError, anthropic.RateLimitError, anthropic.APIStatusError),
    max_time=600,
    max_value=60,
)
def get_response_withtools(
    client, model, messages, tools, tool_choice,
    logging=None, max_retry=3
):
    # 分派到 DeepSeek Chat Completions、Claude Messages 或 o3 Responses。
    # 递归重试处理本函数捕获的异常，装饰器对最终向外传播的指定 API 异常做指数退避。
    # 上下文溢出分支目前只有 pass，并未裁剪历史；这也成为外层可选的进化方向。
    try:
        if model.startswith('deepseek-'):
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=tools,
                **deepseek_request_options(),
            )
        elif 'claude' in model:
            response = client.messages.create(
                model=model,
                messages=messages,
                max_tokens=4096,
                tool_choice=tool_choice,
                tools=tools,
            )
        elif model.startswith('o3-'):
            response = client.responses.create(
                model=model,
                input=messages,
                tool_choice=tool_choice,
                tools=tools,
                parallel_tool_calls=False
            )
            response = response
        else:
            raise ValueError(f"Unsupported model: {model}")
        return response
    except Exception as e:
        logging(f"工具调用模型请求失败：{str(e)}")
        if max_retry > 0:
            return get_response_withtools(client, model, messages, tools, tool_choice, logging, max_retry - 1)

        # Hitting the context window limit
        if 'Input is too long for requested model' in str(e):
            pass

        raise  # Re-raise the exception after logging

def check_for_tool_use(response, model=''):
    # 将供应商响应归一化为 tool_id/name/input，供后续执行与回传关联。
    # 当前路径一次只取一个工具请求；o3 的 API 请求也禁用了并行工具调用。
    # 注意 o3 分支没有稳妥处理空 output 或完全没有 function_call 的情况。
    """
    Checks if the response contains a tool call.
    """
    if 'claude' in model:
        # Claude, check for stop_reason in response
        if response.stop_reason == "tool_use":
            tool_use_block = next(block for block in response.content if block.type == "tool_use")
            return {
                'tool_id': tool_use_block.id,
                'tool_name': tool_use_block.name,
                'tool_input': tool_use_block.input,
            }

    elif model.startswith('o3-'):
        # OpenAI, check for tool_calls in response
        for tool_call in response.output:
            if tool_call.type == "function_call":
                break

        if tool_call:
            return {
                'tool_id': tool_call.call_id,
                'tool_name': tool_call.name,
                'tool_input': json.loads(tool_call.arguments),
            }

    else:
        # Any other LLM, response is str, check for <tool_use> tag in response
        pattern = r'<tool_use>(.*?)</tool_use>'
        match = re.search(pattern, response, re.DOTALL)
        if match:
            tool_use_str = match.group(1).strip()
            try:
                tool_use_dict = ast.literal_eval(tool_use_str)
                if isinstance(tool_use_dict, dict) and 'tool_name' in tool_use_dict and 'tool_input' in tool_use_dict:
                    return tool_use_dict
            except Exception:
                pass

    # No tool use found
    return None

def convert_tool_info(tool_info, model=None):
    # 把统一的 Claude 风格 schema 转成供应商格式；o3 strict 模式要求所有字段 required。
    # 原可选字段改为允许 null，以表达“显式缺省”；本函数会原地修改 input_schema。
    """
    Converts tool_info from Claude format to the given model's format.
    """
    if 'claude' in model:
        # should have no change
        return {
            'name': tool_info['name'],
            'description': tool_info['description'],
            'input_schema': tool_info['input_schema'],
        }
    elif model.startswith('o3-'):
        def add_additional_properties(d):
            if isinstance(d, dict):
                if 'properties' in d:
                    d['additionalProperties'] = False
                for k, v in d.items():
                    add_additional_properties(v)
        add_additional_properties(tool_info['input_schema'])
        for p in tool_info['input_schema']['properties'].keys():
            if not p in tool_info['input_schema']['required']:
                tool_info['input_schema']['required'].append(p)
                t = copy.deepcopy(tool_info['input_schema']['properties'][p]["type"])
                if isinstance(t, str):
                    tool_info['input_schema']['properties'][p]["type"] = [t, "null"]
                elif isinstance(t, list):
                    tool_info['input_schema']['properties'][p]["type"] = t + ["null"]
                
        return {
            'type': 'function',
            'name': tool_info['name'],
            'description': tool_info['description'],
            'parameters': tool_info['input_schema'],
            "strict": True,
        }
    else:
        return tool_info

def convert_block_claude(block):
    """
    Convert a single block of content from Claude into a standard format.
    """
    if isinstance(block, dict):
        block_type = block.get('type')
        text = block.get('text')
        tool_name = block.get('name')
        tool_input = block.get('input')
        tool_result = block.get('content')
    else:
        block_type = getattr(block, 'type', None)
        text = getattr(block, 'text', None)
        tool_name = getattr(block, 'name', None)
        tool_input = getattr(block, 'input', None)
        tool_result = getattr(block, 'content', None)

    text = text or ""

    if block_type == "text":
        return {
            "type": "text",
            "text": text
        }
    elif block_type == "tool_use":
        # Convert to the manual tool calling format
        return {
            "type": "text",
            "text": f"<tool_use>\n{{'tool_name': {tool_name}, 'tool_input': {tool_input}}}\n</tool_use>"
        }
    elif block_type == "tool_result":
        return {
            "type": "text",
            "text": f"Tool Result: {tool_result}"
        }
    else:
        # Fallback if we ever encounter an unknown block type
        return {
            "type": "text",
            "text": str(block)
        }

def convert_msg_history_claude(msg_history):
    """
    Convert Claude-style message history into a generic format.
    """
    new_msg_history = []

    for msg in msg_history:
        role = msg.get('role', '')
        content_blocks = msg.get('content', [])
        new_content = []

        for block in content_blocks:
            new_content.append(convert_block_claude(block))

        new_msg_history.append({
            "role": role,
            "content": new_content
        })

    return new_msg_history

def convert_msg_history_openai(msg_history):
    """
    Convert OpenAI-style message history into a generic format.
    """
    new_msg_history = []

    for msg in msg_history:
        if isinstance(msg, dict):
            role = msg.get('role', '')
            content = msg.get('content', '')

            if role == 'tool':
                new_msg = {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Tool Result: {content}",
                        }
                    ],
                }
            else:
                new_msg = {
                    "role": role,
                    "content": content,
                }
        else:
            role = getattr(msg, 'role', None)
            content = getattr(msg, 'content', None)
            tool_calls = getattr(msg, 'tool_calls', None)

            if tool_calls:
                tool_call = tool_calls[0]
                function_name = getattr(tool_call.function, 'name', '')
                function_args = getattr(tool_call.function, 'arguments', '')
                # Convert to the manual tool calling format
                new_msg = {
                    "role": role,
                    "content": [
                        {
                            "type": "text",
                            "text": f"<tool_use>\n{{'tool_name': {function_name}, 'tool_input': {function_args}}}\n</tool_use>",
                        }
                    ],
                }
            else:
                new_msg = {
                    "role": role,
                    "content": [
                        {
                            "type": "text",
                            "text": content,
                        }
                    ],
                }

        new_msg_history.append(new_msg)

    return new_msg_history

def convert_msg_history(msg_history, model=None):
    # 为日志或跨模型复用整理内容块；主分派函数里 o3 的跨模型转换目前被注释掉。
    """
    Convert message history from the model-specific format to a generic format.
    """
    if model and model.startswith('deepseek-'):
        history = []
        for message in msg_history:
            role = message['role']
            content = message.get('content') or ''
            if role == 'tool':
                role = 'user'
                content = f'Tool Result: {content}'
            elif message.get('tool_calls'):
                content += '\n' + json.dumps(message['tool_calls'], ensure_ascii=False)
            history.append({'role': role, 'content': [{'type': 'text', 'text': content}]})
        return history
    elif 'claude' in model:
        return convert_msg_history_claude(msg_history)
    elif model.startswith('o3-'):
        return convert_msg_history_openai(msg_history)
    else:
        return msg_history

def chat_with_agent_deepseek(msg, model=DEFAULT_MODEL, msg_history=None, logging=print):
    """原生函数调用循环，保留推理字段及所有工具调用的 ID。"""
    client, client_model = create_client(model)
    all_tools = load_all_tools(logging=logging)
    tools_dict = {tool['info']['name']: tool for tool in all_tools}
    tools = [
        {
            'type': 'function',
            'function': {
                'name': tool['info']['name'],
                'description': tool['info']['description'],
                'parameters': copy.deepcopy(tool['info']['input_schema']),
            },
        }
        for tool in all_tools
    ]
    messages = copy.deepcopy(msg_history or [])
    messages.append({'role': 'user', 'content': msg})
    logging(f'用户指令：{msg}')
    while True:
        response = get_response_withtools(
            client, client_model, messages, tools, 'auto', logging=logging,
        )
        choice = response.choices[0]
        message = choice.message
        if choice.finish_reason == 'length':
            raise RuntimeError('DeepSeek 输出达到长度上限，请增大 DEEPSEEK_MAX_TOKENS。')
        assistant = {'role': 'assistant', 'content': message.content}
        reasoning = getattr(message, 'reasoning_content', None)
        if reasoning is not None:
            assistant['reasoning_content'] = reasoning
        calls = message.tool_calls or []
        if calls:
            assistant['tool_calls'] = [call.model_dump(exclude_none=True) for call in calls]
        messages.append(assistant)
        logging(f'智能体响应：{message.content or ""}')
        if not calls:
            return messages
        for call in calls:
            try:
                arguments = json.loads(call.function.arguments)
                if not isinstance(arguments, dict):
                    raise ValueError('工具参数必须为 JSON 对象')
                result = process_tool_call(tools_dict, call.function.name, arguments)
            except (ValueError, TypeError) as error:
                result = f'工具参数解析失败：{error}'
            logging(f'调用工具：{call.function.name}\n工具结果：{result}')
            messages.append({
                'role': 'tool', 'tool_call_id': call.id, 'content': str(result),
            })


def chat_with_agent_manualtools(msg, model, msg_history=None, logging=print):
    # 无原生工具调用时，提示词展示工具源码，并要求模型输出 <tool_use> 标签。
    # 标签通过 ast.literal_eval 解析为数据，再由注册表分派，避免直接 eval 模型输出。
    # Construct message
    if msg_history is None:
        msg_history = []
    system_message = f'You are a coding agent.\n\n{get_tooluse_prompt()}'
    new_msg_history = msg_history

    try:
        # Load all tools
        all_tools = load_all_tools(logging=logging)
        tools_dict = {tool['info']['name']: tool for tool in all_tools}
        
        # Create client
        client, client_model = create_client(model)

        # Call API
        logging(f"输入：{msg}")
        response, new_msg_history = get_response_from_llm(
            msg=msg,
            client=client,
            model=client_model,
            system_message=system_message,
            print_debug=False,
            msg_history=new_msg_history,
        )
        logging(f"输出：{response}")

        # Tool use
        tool_use = check_for_tool_use(response, model=client_model)
        while tool_use:
            # Process tool call
            tool_name = tool_use['tool_name']
            tool_input = tool_use['tool_input']
            tool_result = process_tool_call(tools_dict, tool_name, tool_input)

            # Get tool response
            tool_msg = f'Tool Used: {tool_name}\nTool Input: {tool_input}\nTool Result: {tool_result}'
            # 对话中的工具标记保持原格式，仅将展示用标签改为中文。
            logging(f'调用工具：{tool_name}\n工具输入：{tool_input}\n工具结果：{tool_result}')
            response, new_msg_history = get_response_from_llm(
                msg=tool_msg,
                client=client,
                model=client_model,
                system_message=system_message,
                print_debug=False,
                msg_history=new_msg_history,
            )
            logging(f"输出：{response}")

            # Check for next tool use
            tool_use = check_for_tool_use(response, model=client_model)

    except Exception:
        pass

    return new_msg_history

def chat_with_agent_claude(
        msg,
        model='bedrock/us.anthropic.claude-3-5-sonnet-20241022-v2:0',
        msg_history=None,
        logging=print,
    ):
    # 核心循环：把 response.content 加为 assistant 消息，把实际执行结果加为
    # user/tool_result 消息；tool_use_id 把执行结果关联到原工具调用。
    # 每次请求带上全部历史；模型不再请求工具时，记录最终文本并结束。
    # 当前 except Exception: pass 会吞掉失败并返回已有历史，调用方还需检查补丁/日志。
    # Construct message
    if msg_history is None:
        msg_history = []
    new_msg_history = [
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

    try:
        # Create client
        client, client_model = create_client(model)

        # Load all tools
        all_tools = load_all_tools(logging=logging)
        tools_dict = {tool['info']['name']: tool for tool in all_tools}
        tools = [convert_tool_info(tool['info'], model=client_model) for tool in all_tools]

        # Call API
        response = get_response_withtools(
            client=client,
            model=client_model,
            messages=msg_history + new_msg_history,
            tool_choice={"type": "auto"},
            tools=tools,
            logging=logging,
        )

        # Check for tool use
        tool_use = check_for_tool_use(response, model=client_model)
        while tool_use:
            # Process tool call
            tool_name = tool_use['tool_name']
            tool_input = tool_use['tool_input']
            tool_result = process_tool_call(tools_dict, tool_name, tool_input)

            # Get tool response
            new_msg_history.append({"role": "assistant", "content": response.content})
            new_msg_history.append({
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_use['tool_id'],
                        "content": tool_result,
                    }
                ],
            })
            response = get_response_withtools(
                client=client,
                model=client_model,
                messages=msg_history + new_msg_history,
                tool_choice={"type": "auto"},
                tools=tools,
                logging=logging,
            )

            # Check for next tool use
            tool_use = check_for_tool_use(response, model=client_model)

        # Get final response
        final_response = next((block.text for block in response.content if hasattr(block, "text")), None)
        new_msg_history.append({
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": final_response,
                }
            ],
        })

    except Exception:
        pass

    return new_msg_history

def chat_with_agent_openai(
        msg,
        model='o3-mini-2025-01-31',
        msg_history=None,
        logging=print,
    ):
    # o3 使用 Responses 消息格式：先记录 function_call，再记录带相同 call_id
    # 的 function_call_output，回传执行结果以继续模型决策。
    # 当前循环主要保存工具调用与结果，最终保存 response；不要把它视为完整的推理轨迹。
    # Construct message
    if msg_history is None:
        msg_history = []
    new_msg_history = [
        {
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": msg,
                }
            ],
        }
    ]
    separator = '=' * 10
    logging(f"\n{separator} 用户指令 {separator}\n{msg}")
    try:
        # Create client
        client, client_model = create_client(model)

        # Load all tools
        all_tools = load_all_tools(logging=logging)
        tools_dict = {tool['info']['name']: tool for tool in all_tools}
        tools = [convert_tool_info(tool['info'], model=client_model) for tool in all_tools]

        # Call API
        response = get_response_withtools(
            client=client,
            model=client_model,
            messages=msg_history + new_msg_history,
            tool_choice="auto",
            tools=tools,
            logging=logging,
        )
        logging(f"\n{separator} 智能体响应 {separator}\n{response}")

        # Check for tool use
        tool_use = check_for_tool_use(response, model=client_model)
        logging(tool_use)
        while tool_use:
            # Process tool call
            tool_name = tool_use['tool_name']
            tool_input = tool_use['tool_input']
            tool_result = process_tool_call(tools_dict, tool_name, tool_input)

            logging(f"调用工具：{tool_name}")
            logging(f"工具输入：{tool_input}")
            logging(f"工具结果：{tool_result}")

            # Get tool response
            for tool_call in response.output:
                if tool_call.type == "function_call":
                    break
            new_msg_history.append(tool_call)
            new_msg_history.append({
                "type": "function_call_output",
                "call_id": tool_use['tool_id'],
                "output": tool_result,
            })
            response = get_response_withtools(
                client=client,
                model=client_model,
                messages=msg_history + new_msg_history,
                tool_choice="auto",
                tools=tools,
                logging=logging,
            )

            # Check for next tool use
            tool_use = check_for_tool_use(response, model=client_model)

            logging(f"工具响应：{response}")

        # Get final response
        new_msg_history.append(response)

    except Exception:
        pass

    return new_msg_history

def chat_with_agent(
    msg,
    model=CLAUDE_MODEL,
    msg_history=None,
    logging=print,
    convert=False,  # Convert the message history to a generic format, so that msg_history can be used across models
):
    # 代理统一入口；根据模型名选择工具协议，最终返回对话历史，文件修改由工具完成。
    # Claude/o3 子函数返回本轮新增历史，本函数再拼上传入的历史。
    if msg_history is None:
        msg_history = []

    if model.startswith('deepseek-'):
        new_msg_history = chat_with_agent_deepseek(
            msg, model=model, msg_history=msg_history, logging=logging,
        )
        if convert:
            new_msg_history = convert_msg_history(new_msg_history, model=model)
    elif 'claude' in model:
        # Claude models
        new_msg_history = chat_with_agent_claude(msg, model=model, msg_history=msg_history, logging=logging)
        conv_msg_history = convert_msg_history(new_msg_history, model=model)
        logging(conv_msg_history)
        if convert:
            new_msg_history = conv_msg_history
        new_msg_history = msg_history + new_msg_history

    elif model.startswith('o3-'):
        # OpenAI models
        new_msg_history = chat_with_agent_openai(msg, model=model, msg_history=msg_history, logging=logging)
        # Current version does not support cross-model conversion
        # new_msg_history = convert_msg_history(new_msg_history, model=model)
        new_msg_history = msg_history + new_msg_history

    else:
        # Models without in-built tool calling
        new_msg_history = chat_with_agent_manualtools(msg, model=model, msg_history=msg_history, logging=logging)
        conv_msg_history = convert_msg_history(new_msg_history, model=model)
        if convert:
            new_msg_history = conv_msg_history

    return new_msg_history


if __name__ == "__main__":
    # Test the tool calling functionality
    msg = "hello!"
    chat_with_agent(msg)
