# 中文导读：一次 DGM 变异的完整闭环。本文件衔接“诊断模型”和“可修改代码的代理”。
# 先重建父代 → 诊断失败 → 在容器中让父代编辑 /dgm → 导出增量补丁 → 基准评估。
# 诊断模型生成改进任务，coding_agent 执行任务，基准测试提供分数；三者职责不同。
import argparse
import datetime
import json
import os
import docker

from llm import DEFAULT_MODEL, create_client, get_response_from_llm, extract_json_between_markers, llm_environment
from prompts.self_improvement_prompt import get_diagnose_prompt_polyglot, get_diagnose_prompt_swe, get_problem_description_prompt
from prompts.diagnose_improvement_prompt import get_diagnose_improvement_prompt
from prompts.testrepo_prompt import get_test_description
from swe_bench.harness import harness
from polyglot.harness import harness as polyglot_harness
from swe_bench.report import make_report
from utils.common_utils import load_json_file
from utils.evo_utils import get_model_patch_paths, get_all_performance, is_compiled_self_improve
from utils.docker_utils import (
    build_dgm_container,
    cleanup_container,
    copy_from_container,
    copy_to_container,
    log_container_output,
    remove_existing_container,
    setup_logger,
    safe_log,
)

dataset = None
diagnose_model = DEFAULT_MODEL

def diagnose_problem(entry, commit, root_dir, out_dir, patch_files=[], max_attempts=3, polyglot=False):
    # 从失败日志、代理源码和评测证据构造上下文，请模型提出一个通用能力改进。
    # 解析 JSON 后只把 implementation_suggestion 和 problem_description 等内容
    # 整理为代理可以实施的任务；解析失败递归重试，初次调用之外最多再试 max_attempts 次。
    client = create_client(diagnose_model)
    if polyglot:
        diagnose_sys_message, diagnose_prompt = get_diagnose_prompt_polyglot(
            entry, commit, root_dir, out_dir, dataset,
            patch_files=patch_files,
        )
    else:
        diagnose_sys_message, diagnose_prompt = get_diagnose_prompt_swe(
            entry, commit, root_dir, out_dir, dataset,
            patch_files=patch_files,
        )
    try:
        response, msg_history = get_response_from_llm(
            msg=diagnose_prompt,
            client=client[0],
            model=client[1],
            system_message=diagnose_sys_message,
            print_debug=False,
            msg_history=None,
        )
        safe_log(f"消息历史：{msg_history}")
        response_json = extract_json_between_markers(response)
        assert response_json, "empty response json"
        problem_statement = get_problem_description_prompt(response_json, polyglot)
    except Exception as e:
        # Exception most probably due to not having json in the response
        safe_log(f"问题诊断失败：{e}")
        if max_attempts > 0:
            return diagnose_problem(
                entry, commit, root_dir, out_dir,
                patch_files=patch_files,
                max_attempts=max_attempts-1,
                polyglot=polyglot,
            )
        else:
            return None
    return problem_statement

def diagnose_improvement(
        entry, parent_commit, root_dir, model_patch_file, out_dir, run_id,
        patch_files=[], max_attempts=3,
    ):
    # 可选的事后文字分析，比较父代与子代解题轨迹；不参与外层的分数抽样公式。
    # 注意：对应提示构造器目前与日志函数的参数/返回值不一致，见该文件中文注释。
    """
    Diagnose the improvement of the model patch.

    Args:
        entry (str): The task entry to improve.
        parent_commit (str): The commit hash of the parent commit.
        root_dir (str): The root directory of the repository.
        model_patch_file (str): The path to the model patch file.
        out_dir (str): The output directory.
        run_id (str): The run id of the self-improvement attempt.
        patch_files (list): The list of patch files before self-improvement.
        max_attempts (int): The maximum number of attempts to diagnose the improvement.
    
    Returns:
        dict: The improvement diagnosis.
    """
    client = create_client(diagnose_model)
    diagnose_sys_message, diagnose_prompt = get_diagnose_improvement_prompt(
        entry, parent_commit, root_dir, model_patch_file, out_dir, run_id, dataset,
        patch_files=patch_files,
    )
    safe_log(f"正在诊断改进：{diagnose_prompt}")
    try:
        response, msg_history = get_response_from_llm(
            msg=diagnose_prompt,
            client=client[0],
            model=client[1],
            system_message=diagnose_sys_message,
            print_debug=False,
            msg_history=None,
        )
        safe_log(f"消息历史：{msg_history}")
        response_json = extract_json_between_markers(response)
        assert response_json, "empty response json"
        improvement_diagnosis = response_json
    except Exception as e:
        # Exception most probably due to not having json in the response
        safe_log(f"改进诊断失败：{e}")
        if max_attempts > 0:
            return diagnose_improvement(
                entry, parent_commit, root_dir, model_patch_file, out_dir, run_id,
                patch_files=patch_files, max_attempts=max_attempts-1,
            )
        else:
            return None
    return improvement_diagnosis

def save_metadata(metadata, output_dir):
    # 节点元数据连接谱系与评测：run_id、parent_commit、entry、改进任务、性能等。
    metadata_file = os.path.join(output_dir, "metadata.json")
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=4)

def run_harness_swe(entry, model_name_or_path, patch_files, num_evals, output_dir, metadata, run_id, test_more_threshold, test_task_list, test_task_list_more, eval_workers=1):
    # 两种补丁需要区分：patch_files 改的是代理自身；harness 输出改的是基准题仓库。
    # harness 先让新代理解题，make_report 再用 SWE-bench 官方测试判断题目是否解决。
    # num_evals 表示重复解题次数，用于观察随机性；结果按提交实例总数聚合为解题率。
    safe_log('开始运行评估框架')
    test_task_list = [entry] if test_task_list is None else test_task_list
    dnames = harness(
        test_task_list=test_task_list,
        num_samples=-1,
        max_workers=min(eval_workers, len(test_task_list)),
        model_name_or_path=model_name_or_path,
        model_patch_paths=patch_files,
        num_evals=num_evals,
        num_evals_parallel=1,
        pred_dname=os.path.join(output_dir, "predictions"),
    )
    metadata['swe_dnames'] = [str(dn) for dn in dnames]
    safe_log('开始生成评估报告')
    make_report(
        dnames,
        run_ids=[f"{run_id}_{i}" for i in range(len(dnames))],
        dataset_name="princeton-nlp/SWE-bench_Verified",
        output_dir=output_dir,
        dnames_workers=1,
        num_eval_procs=eval_workers,
    )
    safe_log('开始统计性能')
    performances, overall_performance = get_all_performance(model_name_or_path, results_dir=output_dir)
    metadata['overall_performance'] = overall_performance
    if overall_performance is None:
        raise RuntimeError('未生成 SWE-bench 性能报告，评估未完成。')
    safe_log("评估结束")

    # Check if additional evaluation should be run
    # 达到阈值才扩展到 medium 子集以控制评估成本；这里比较成功实例数与题数乘阈值，
    # 多次重复评估时成功数也累加，因此并不等价于比较 accuracy_score 与阈值。
    if (overall_performance and \
        test_more_threshold is not None and test_task_list_more is not None and \
            overall_performance.get('total_resolved_instances', 0) >= len(test_task_list) * test_more_threshold):
        safe_log("开始额外评估轮次")
        dnames = harness(
            test_task_list=test_task_list_more,
            num_samples=-1,
            max_workers=min(eval_workers, len(test_task_list_more)),
            model_name_or_path=model_name_or_path,
            model_patch_paths=patch_files,
            num_evals=num_evals,
            num_evals_parallel=1,
            pred_dname=os.path.join(output_dir, "predictions"),
        )
        safe_log('开始生成额外评估报告')
        make_report(
            dnames,
            run_ids=[f"{run_id}_{i}" for i in range(len(dnames))],
            dataset_name="princeton-nlp/SWE-bench_Verified",
            output_dir=output_dir,
            dnames_workers=1,
            num_eval_procs=eval_workers,
        )
        safe_log('开始统计性能')
        performances, overall_performance = get_all_performance(model_name_or_path, results_dir=output_dir)
        metadata['overall_performance'] = overall_performance
        if overall_performance is None:
            raise RuntimeError('未生成额外 SWE-bench 性能报告，评估未完成。')
        safe_log("额外评估结束")

def run_harness_polyglot(entry, model_name_or_path, patch_files, num_evals, output_dir, metadata, run_id, test_more_threshold, test_task_list, test_task_list_more):
    # Polyglot 在 harness 内直接测试和写报告，无需再调用 SWE 的 make_report。
    # 追加评估写入 overall_performance_deep；外层抽样仍读取 overall_performance。
    # 当前 Polyglot harness 显式限制 num_evals <= 1。
    safe_log('开始运行评估框架')
    test_task_list = [entry] if test_task_list is None else test_task_list
    safe_log(f'工作线程数：{min(10, len(test_task_list))}')
    dnames = polyglot_harness(
        test_task_list=test_task_list,
        num_samples=-1,
        max_workers=min(10, len(test_task_list)),
        model_name_or_path=model_name_or_path,
        model_patch_paths=patch_files,
        num_evals=num_evals,
        num_evals_parallel=min(5, num_evals),
        pred_dname=os.path.join(output_dir, "predictions"),
        output_dir=output_dir
    )
    metadata['swe_dnames'] = [str(dn) for dn in dnames]
    safe_log('开始统计性能')
    performances, overall_performance = get_all_performance(model_name_or_path, results_dir=output_dir)
    metadata['overall_performance'] = overall_performance
    safe_log("评估结束")

    # Check if additional evaluation should be run
    if (overall_performance and \
        test_more_threshold is not None and test_task_list_more is not None and \
            overall_performance.get('total_resolved_instances', 0) >= len(test_task_list) * test_more_threshold):
        safe_log("开始额外评估轮次")
        dnames = polyglot_harness(
            test_task_list=test_task_list_more,
            num_samples=-1,
            max_workers=50,
            model_name_or_path=model_name_or_path,
            model_patch_paths=patch_files,
            num_evals=num_evals,
            num_evals_parallel=min(5, num_evals),
            pred_dname=os.path.join(output_dir, "predictions"),
            output_dir=output_dir
        )
        # metadata['swe_dnames'] = [str(dn) for dn in dnames]
        safe_log('开始统计性能')
        performances, overall_performance = get_all_performance(model_name_or_path, results_dir=output_dir)
        metadata['overall_performance_deep'] = overall_performance
        safe_log("额外评估结束")

def self_improve(
    parent_commit='initial',  # 'initial' if starting from original dgm, else the run_id
    output_dir='output_selfimprove/',
    force_rebuild=False,
    num_evals=1,
    post_improve_diagnose=True,
    entry=None,
    test_task_list=None,  # None means the entry above only
    # Additional evaluation parameters
    test_more_threshold=None,
    test_task_list_more=None,
    full_eval_threshold=None,
    # Run baseline
    run_baseline=None,
    polyglot=False,
    eval_workers=1,
):  
    # parent_commit 实际是档案节点 ID（initial 或时间戳），不是 Git commit hash。
    # 每个节点目录保存相对父代的增量，重建时必须按祖先到父代的顺序应用整条补丁链。
    # full_eval_threshold 当前未被使用；下面虽加载 big 子集，也没有执行 big 评估。

    global dataset
    # 数据供诊断使用；此处是模块全局变量，外层线程之间共享同一数据引用。
    if polyglot:
        with open("polyglot/polyglot_benchmark_metadata.json") as f:
            dataset = json.loads(f.read())
    else:
        from datasets import load_dataset
        dataset = load_dataset("princeton-nlp/SWE-bench_Verified")
        dataset = dataset['test']

    # Variables for this self-improvement attempt
    metadata = {}
    root_dir = os.path.abspath('./')  # root_dir should be /dgm
    run_id = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    out_dir_base = output_dir  # out_dir_base should be /dgm/output_selfimprove/ or /dgm/output_dgm/{dgm_run_id}/
    output_dir = os.path.join(root_dir, f"{output_dir}/{run_id}/")
    os.makedirs(output_dir, exist_ok=True)
    metadata['run_id'] = run_id
    metadata['parent_commit'] = parent_commit
    test_task_list_big = load_json_file("./swe_bench/subsets/big.json")

    # Set up logger
    logger = setup_logger(os.path.join(output_dir, "self_improve.log"))

    # Create and start the Docker container
    image_name = "dgm:latest"
    container_name = f"dgm-container-{run_id}"
    client = docker.from_env()
    # Remove any existing container with the same name
    remove_existing_container(client, container_name)
    # Now create and start the container
    container = build_dgm_container(
        client, root_dir, image_name, container_name,
        force_rebuild=force_rebuild,
    )
    container.start()

    if polyglot:
        # remove the swe version of coding_agent.py
        # 在容器里统一入口名：Polyglot 版本替换为 /dgm/coding_agent.py，
        # 之后的自修改补丁和评估器都针对这个统一路径工作。
        exec_result = container.exec_run("rm /dgm/coding_agent.py", workdir='/')
        log_container_output(exec_result)
        # rename coding_agent_polyglot.py to coding_agent.py
        exec_result = container.exec_run("mv /dgm/coding_agent_polyglot.py /dgm/coding_agent.py", workdir='/')
        log_container_output(exec_result)
        # remove swe-specific files utils/eval_utils.py and utils/swe_log_parsers.py
        exec_result = container.exec_run("rm /dgm/utils/eval_utils.py", workdir='/')
        log_container_output(exec_result)
        exec_result = container.exec_run("rm /dgm/utils/swe_log_parsers.py", workdir='/')
        log_container_output(exec_result)
    else:
        # remove the polyglot version of coding_agent.py
        exec_result = container.exec_run("rm /dgm/coding_agent_polyglot.py", workdir='/')

    # Find all parent patches and apply them
    patch_files = get_model_patch_paths(root_dir, os.path.join(output_dir, '../'), parent_commit)
    # no_selfimprove 消融在变异容器里跳过祖先补丁，编辑初始代码；普通路径恢复父代。
    if run_baseline not in ['no_selfimprove']:
        for patch_file in patch_files:
            copy_to_container(container, patch_file, '/dgm/parent_patch.txt')
            exec_result = container.exec_run("/bin/sh -c 'patch -p1 < /dgm/parent_patch.txt'", workdir='/dgm')
            log_container_output(exec_result)
            exec_result = container.exec_run("rm /dgm/parent_patch.txt", workdir='/dgm')
            log_container_output(exec_result)

    # Commit this version of dgm, so that irrelevant changes are not included in the patch
    # 父代代码作为新的 Git 基线：最终 diff 仅记录本次变异，不重复包含祖先改动。
    # 这里的 commit_hash 才是真正传给 git diff 的提交标识。
    exec_result = container.exec_run("git add --all", workdir='/dgm/')
    log_container_output(exec_result)
    exec_result = container.exec_run("git -c user.name='user' -c user.email='you@example.com' commit -m 'a nonsense commit message'", workdir='/dgm/')
    log_container_output(exec_result)
    commit_output = exec_result.output.decode('utf-8')
    # Git commit output format: `[master (root-commit) <hash>] a nonsense commit message`
    commit_hash = commit_output.split()[1].strip("[]")  # Extract the hash part

    # Install requirements again in case of any changes
    exec_result = container.exec_run("python -m pip install -r /dgm/requirements.txt", workdir='/')
    log_container_output(exec_result)

    # Get tasks to improve
    # 先解释“代理为什么失败”，再把改进任务交给代理实现；不是直接照抄题目答案。
    if entry:
        safe_log(f"待改进任务：{entry}")
        problem_statement = diagnose_problem(entry, parent_commit, root_dir, out_dir_base, patch_files=patch_files, polyglot=polyglot)
        safe_log(f"问题描述：{problem_statement}")
    else:
        safe_log("未提供任务，退出。")
        cleanup_container(container)
        save_metadata(metadata, output_dir)
        return metadata

    metadata['entry'] = entry
    metadata['problem_statement'] = problem_statement
    # If problem statement is not found, exit
    if not problem_statement:
        safe_log("问题描述诊断失败，退出。")
        cleanup_container(container)
        save_metadata(metadata, output_dir)
        return metadata

    # Run self-improvement
    safe_log("正在执行自改进")
    chat_history_file_container = "/dgm/self_evo.md"
    test_description = get_test_description(swerepo=False)
    env_vars = llm_environment()
    cmd = [
        # 同一个编程代理也能编辑自己的实现：--git_dir 从解题时的 /testbed/
        # 换成 /dgm/，目标变为工具、提示词和工作流；30 分钟超时由 Linux timeout 执行。
        "timeout", "1800",  # 30min timeout
        "python", "/dgm/coding_agent.py",
        "--problem_statement", problem_statement,
        "--git_dir", "/dgm/",
        "--chat_history_file", chat_history_file_container,
        "--base_commit", commit_hash,
        "--outdir", "/dgm/",
        "--test_description", test_description,
        "--self_improve",
    ]
    exec_result = container.exec_run(cmd, environment=env_vars, workdir='/')
    log_container_output(exec_result)

    # Copy output files back to host
    chat_history_file = os.path.join(output_dir, "self_evo.md")
    copy_from_container(container, chat_history_file_container, chat_history_file)
    model_patch_file = os.path.join(output_dir, "model_patch.diff")
    copy_from_container(container, "/dgm/model_patch.diff", model_patch_file)

    # Try reading the patch file to validate it
    try:
        # Check if patch file exists and is not empty
        if not os.path.exists(model_patch_file):
            raise Exception("Model patch file is empty or does not exist")
        with open(model_patch_file, 'r') as f:
            patch_content = f.read()
            if not patch_content.strip():
                raise Exception("Model patch file is empty")
    except Exception as e:
        safe_log(f"读取模型补丁文件失败：{str(e)}")
        save_metadata(metadata, output_dir)
        return metadata

    patch_files.append(model_patch_file)
    # 补丁链 = 祖先补丁 + 本次增量；评估时在新容器中重放它，测试得到的子代代理。

    # Stop and remove the container
    cleanup_container(container)

    # Evaluate the performance of the self-improvement
    model_patch_exists = os.path.exists(model_patch_file)
    metadata['model_patch_exists'] = model_patch_exists
    model_patch_notempty = os.path.getsize(model_patch_file) > 0
    metadata['model_patch_notempty'] = model_patch_notempty
    model_name_or_path = run_id
    if model_patch_exists and model_patch_notempty:
        try:
            if not polyglot:
                run_harness_swe(entry, model_name_or_path, patch_files, num_evals, output_dir, metadata, run_id, test_more_threshold, test_task_list, test_task_list_more, eval_workers=eval_workers)
            else:
                run_harness_polyglot(entry, model_name_or_path, patch_files, num_evals, output_dir, metadata, run_id, test_more_threshold, test_task_list, test_task_list_more)
        except Exception as e:
            metadata['evaluation_error'] = str(e)
            safe_log(f"自改进评估失败：{e}")

    # Post-self-improvement diagnosis
    # 性能结论由基准结果提供；这里的模型文字诊断是辅助解释，不能替代测试分数。
    # 直接调用 is_compiled_self_improve 未传 logger 和非空题数列表，当前实现存在
    # None.info / 空列表索引风险；外层 filter_compiled 的调用则提供了这些参数。
    if post_improve_diagnose:
        safe_log("正在诊断自改进")
        metadata['is_compiled'] = is_compiled_self_improve(metadata)
        if metadata['is_compiled']:
            safe_log("自改进代码通过编译")
            improvement_diagnosis = diagnose_improvement(
                entry, parent_commit, root_dir,
                model_patch_file, out_dir_base, run_id,
                patch_files=patch_files,
            )
            metadata['improvement_diagnosis'] = improvement_diagnosis
            safe_log(f"改进诊断：{improvement_diagnosis}")
        else:
            safe_log("自改进代码未通过编译")
            metadata['improvement_diagnosis'] = "Fail to complied. Ignore this."

    # Save metadata of this self-improvement attempt
    save_metadata(metadata, output_dir)
    return metadata

def main():
    parser = argparse.ArgumentParser(description="Self-improvement step for the repository.")
    parser.add_argument('--parent_commit', default="initial", type=str, help='Current commit to find the eval results, "initial" if starting from original dgm, else the run_id')
    parser.add_argument('--output_dir', default="./output_selfimprove", type=str, help='Directory to store the output')
    parser.add_argument('--force_rebuild', default=False, action='store_true', help='Force rebuild of the Docker image')
    parser.add_argument('--num_evals', default=1, type=int, help='Repeated number of swe evaluations after self-improvement')
    parser.add_argument('--no_post_improve_diagnose', default=False, action='store_true', help='Skip diagnosing the self-improvement after evaluation')
    parser.add_argument('--entry', default="django__django-10999", type=str, help='Task entry to improve')
    parser.add_argument('--test_task_list', default=None, type=str, help='List of tasks to evaluate the self-improvement')
    args = parser.parse_args()

    # Copy cached initial version into experiment dir
    os.system(f"cp -r initial/ {args.output_dir}")

    metadata = self_improve(
        parent_commit=args.parent_commit,
        output_dir=args.output_dir,
        force_rebuild=args.force_rebuild,
        num_evals=args.num_evals,
        post_improve_diagnose=not args.no_post_improve_diagnose,
        entry=args.entry,
        test_task_list=args.test_task_list,
    )

if __name__ == "__main__":
    main()
