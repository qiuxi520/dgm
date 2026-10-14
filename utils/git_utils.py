import os
# 中文导读：Git 补丁是 DGM 的版本载体；外层节点 ID 与实际 Git 提交哈希是两套标识。
import git
import subprocess


def get_git_commit_hash(repo_path='.'):
    try:
        # Load the repository
        repo = git.Repo(repo_path)
        # Get the current commit hash
        commit_hash = repo.head.commit.hexsha
        return commit_hash
    except Exception as e:
        print("获取 Git 提交哈希失败：", e)
        return None

def apply_patch(git_dname, patch_str):
    """
    Apply a patch to the repository at `git_dname`.
    """
    cmd = ["git", "-C", git_dname, "apply", "--reject", "-"]
    result = subprocess.run(
        cmd,
        input=patch_str,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False
    )
    # Check if the patch was applied successfully
    if result.returncode != 0:
        print(f"应用补丁失败：补丁未完全应用，返回码：{result.returncode}, 标准输出： {result.stdout}, 标准错误： {result.stderr}")
    else:
        print("补丁应用成功")

def diff_versus_commit(git_dname, commit):
    # 已跟踪文件用 git diff <基线提交>；未跟踪新文件与空文件比较生成新增文件 diff。
    # 这保证新增工具也能遗传给子代，同时避免为了导出 diff 而修改 Git 暂存区。
    """
    Take a diff of `git_dname` current contents versus the `commit`, including untracked files,
    without modifying the repository state.
    """
    # Get diff of tracked files
    diff_cmd = ["git", "-C", git_dname, "diff", commit]
    result = subprocess.run(diff_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    diff_output = result.stdout.decode()

    # Get list of untracked files
    untracked_files_cmd = ["git", "-C", git_dname, "ls-files", "--others", "--exclude-standard"]
    result = subprocess.run(untracked_files_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    untracked_files = result.stdout.decode().splitlines()

    # Generate diffs for untracked files
    for file in untracked_files:
        # Diff untracked file against /dev/null (empty file)
        file_path = os.path.join(git_dname, file)
        devnull = '/dev/null'
        if os.name == 'nt':  # Handle Windows
            devnull = 'NUL'
        diff_file_cmd = ["git", "-C", git_dname, "diff", "--no-index", devnull, file]
        result = subprocess.run(
            diff_file_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=git_dname,
            check=False
        )
        diff_file_output = result.stdout.decode('utf-8', errors='replace')
        diff_output += diff_file_output

    return diff_output

def reset_to_commit(git_dname, commit):
    # 恢复独立工作区的基线，并清除未跟踪文件；会丢弃该工作区尚未保存的修改。
    """
    Reset the repository at `git_dname` to the given `commit`.
    """
    # Step 1: Hard-reset tracked files
    reset_cmd = ["git", "-C", git_dname, "reset", "--hard", commit]
    result_reset = subprocess.run(
        reset_cmd,
        capture_output=True,
        text=True,
        check=False
    )
    if result_reset.returncode != 0:
        print(f"重置提交失败：无法将 {git_dname} 重置到提交 '{commit}'. 标准输出： {result_reset.stdout} 标准错误： {result_reset.stderr}")
    else:
        print(f"提交重置成功：{commit}")

    # Step 2: Clean untracked files (the "new files") and directories
    clean_cmd = ["git", "-C", git_dname, "clean", "-fd"]
    result_clean = subprocess.run(
        clean_cmd,
        capture_output=True,
        text=True,
        check=False
    )
    if result_clean.returncode != 0:
        print(f"重置提交后清理目录失败：{git_dname}. 标准输出： {result_clean.stdout} 标准错误： {result_clean.stderr}")
    else:
        print(f"重置提交后清理目录成功：{commit}")


def filter_patch_by_files(patch_str, target_files):
    # 按 diff --git 文件块筛选，只保留目标文件对应的改动；不是按函数或代码行筛选。
    """
    Filters out the diff blocks related to any of the target_files in a patch string.

    Args:
        patch_str (str): The complete patch text.
        target_files (list[str]): A list of filenames for which to extract changes (e.g. ['affine_cipher.py', 'other.py']).

    Returns:
        str: A string containing only the diff blocks for the specified target files.
    """
    lines = patch_str.splitlines()
    filtered_lines = []
    include_block = False

    for line in lines:
        # When we encounter a new diff block header, check if the block is for any of the target files.
        if line.startswith("diff --git"):
            include_block = any(f"a/{target}" in line and f"b/{target}" in line for target in target_files)
        if include_block:
            filtered_lines.append(line)
    return "\n".join(filtered_lines)


def remove_patch_by_files(patch_str, keyword='polyglot'):
    # 排除文件头包含关键字的补丁块；Polyglot harness 会用它清理代理补丁。
    """
    Removes diff blocks related to files containing the keyword from a patch string.

    Args:
        patch_str (str): The complete patch text.
        keyword (str): Keyword to match in filenames for removal (default: 'polyglot').

    Returns:
        str: A string containing the patch with diff blocks for matching files removed.
    """
    lines = patch_str.splitlines()
    filtered_lines = []
    include_block = True

    for line in lines:
        # When we encounter a new diff block header, check if the block contains the keyword
        if line.startswith("diff --git"):
            include_block = keyword.lower() not in line.lower()
        if include_block:
            filtered_lines.append(line)

    return "\n".join(filtered_lines)
