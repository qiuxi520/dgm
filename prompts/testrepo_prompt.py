

def get_test_command(eval_script):
    # 从 SWE 评估脚本倒数第二行提取测试命令，并去掉特定路径参数以生成通用测试提示。
    # 这依赖脚本格式约定，不是任意 shell 脚本的通用解析器。
    test_hint = ''
    # test_command is the 2nd last line in eval_script
    lines = eval_script.strip().split('\n')
    test_command = lines[-2].strip()
    # Remove trailing arguments specifying filepaths
    parts = test_command.split()
    if '.' in parts[-1] and not parts[-1].endswith('.py'):
        # Get the test hint
        test_hint = 'If the target test file path is tests/some_folder/some_file.py, then <specific test files> should be `some_folder.some_file`.'
    while parts and '.' in parts[-1]:
        parts.pop()
    # Reconstruct the command
    test_command = ' '.join(parts)
    return f'cd /testbed/ && {test_command} <specific test files>', test_hint

def get_test_description(eval_script='', swerepo=False, polyglot=False):
    # 区分 SWE 仓库、多语言题和 DGM 自身。自修改时只建议测试 tools/utils，
    # 提示禁止运行 forward，避免测试时再次触发模型调用和代理执行链。
    assert not (swerepo and polyglot), "swerepo and polyglot cannot both be True"
    if swerepo:  # SWE repo
        swe_prompt = '''The tests in the repository can be run with the bash command `{test_command}`. If no specific test files are provided, all tests will be run. The given command-line options must be used EXACTLY as specified. Do not use any other command-line options. {test_hint}'''
        test_command, test_hint = get_test_command(eval_script)
        description = swe_prompt.format(test_command=test_command, test_hint=test_hint)
    elif polyglot:
        description = f"In the repository folder, the tests can be run with the following bash command(s):\n\n```{eval_script}```\n"
    else:  # dgm repo
        description = 'The tests in the repository can be run with the bash command `cd /dgm/ && pytest -rA <specific test files>`. If no specific test files are provided, all tests will be run. The given command-line options must be used EXACTLY as specified. Do not use any other command-line options. ONLY test tools and utils. NEVER try to test or run agentic_system.forward().'

    return description.strip()
