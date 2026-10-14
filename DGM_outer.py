# 中文导读：DGM 的外层进化调度器（先读本文件，再读 self_improve_step.py）。
# archive 保存可继续繁衍的代理版本 ID；每个版本用父节点 ID、增量 diff 和评估结果表示。
# 一代的流程：选择父代与诊断入口 → 并行自修改 → 过滤无效候选 → 更新档案 → 保存状态。
# “自进化”发生在代理的 Python 代码、提示词和工具上，本模块没有训练基础模型权重。
import argparse
import datetime
import json
import math
import os
import random
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed, TimeoutError

from prompts.self_improvement_prompt import find_selfimprove_eval_logs
from self_improve_step import self_improve
from utils.common_utils import load_json_file
from utils.docker_utils import setup_logger
from utils.evo_utils import load_dgm_metadata, is_compiled_self_improve

def initialize_run(output_dir, prevrun_dir=None, polyglot=False):
    # 新实验使用仓库中缓存的 initial 评估结果；续跑取最后一代档案，从下一代开始。
    # initial 是版本 ID，即使使用 Polyglot，后续代码仍以 output_dir/initial 寻找元数据。
    # Initialize archive
    start_gen_num = 0
    if not prevrun_dir:
        archive = ['initial']
    else:
        # Load previous run's archive
        metadata_path = os.path.join(prevrun_dir, "dgm_metadata.jsonl")
        metadata = load_dgm_metadata(metadata_path, last_only=True)
        archive = metadata['archive']
        start_gen_num = metadata['generation'] + 1

    # Copy cached initial version into experiment dir
    initial_folder_name = 'initial' if not polyglot else 'initial_polyglot'
    if not prevrun_dir and not os.path.exists(f"{output_dir}/{initial_folder_name}"):
        if os.path.exists(initial_folder_name):
            os.system(f"cp -r {initial_folder_name}/ {output_dir}/initial")
        else:
            raise RuntimeError("Error: Need to properly configure evaluation results for the initial version.")
    
    return archive, start_gen_num

def any_exceeding_context_length(output_dir, commit_id, instance_ids):
    # 这里根据历史日志中的特定报错判断上下文溢出，而不是计算 token 数。
    # 连续出现两次报错被视为父代未妥善处理该问题；filter=False 保留原始错误行。
    """
    Check if any of the issues have exceeded the context length.
    """
    for instance_id in instance_ids:
        md_logs, _, _, _ = find_selfimprove_eval_logs(instance_id, output_dir, commit_id, filter=False)
        md_log = md_logs[0]
        # 兼容历史英文日志与当前中文日志，保留供应商返回的原始错误内容。
        error_detail = "Error code: 400 - {'message': 'Input is too long for requested model.'}"
        error_strings = [
            f"{prefix}{error_detail}"
            for prefix in ("Error in get_response_withtools: ", "工具调用模型请求失败：")
        ]
        if any(f'{error_str}\n{error_str}' in md_log for error_str in error_strings):
            return True
    return False

def choose_selfimproves(output_dir, archive, selfimprove_size, method='random', run_baseline=None, polyglot=False):
    # 返回 [(父版本 ID, 改进入口), ...]；入口可为失败题目 ID，也可为通用缺陷标签。
    # 一道失败题只提供诊断线索，诊断提示要求提出适用于多个仓库的代理能力改进。
    """
    Choose self-improve attempts for the current generation.
    """
    selfimprove_entries = []

    # Get parent candidates
    candidates = {}
    # 按档案顺序读取节点，并统计直接子代数；假设父节点已先进入 candidates。
    # 不具备可读评估元数据的节点会被跳过，不能成为下一代父代。
    for commit in archive:
        try:
            metadata_path = os.path.join(output_dir, commit, "metadata.json")
            metadata = load_json_file(metadata_path)
            candidates[commit] = {
                'accuracy_score': metadata['overall_performance']['accuracy_score'],
                'total_unresolved_ids': metadata['overall_performance']['total_unresolved_ids'],
                'total_emptypatch_ids': metadata['overall_performance']['total_emptypatch_ids'],
                'total_resolved_ids': metadata['overall_performance']['total_resolved_ids'],
                'children_count': 0,
            }
            # update children count, parent should already be in the archive
            if commit != 'initial':
                parent_commit = metadata['parent_commit']
                candidates[parent_commit]['children_count'] += 1
        except Exception as e:
            # probably because swe-eval failed, generated code did not compile, etc.
            print(f"{commit} 不满足父代条件：{e}")
            continue

    # Choose parents based on method and baseline
    if run_baseline == 'no_darwin':
        # 消融实验：只延续最后一个候选，不从历史分支中搜索；此分支只取一个父代。
        # Always take the last commit
        commits = list(candidates.keys())
        parent_commits = commits[-1:]
    elif method == 'score_prop':
        # 将解题率经 sigmoid 映射到正权重，再归一化抽样；并非按原始分数直接抽样。
        # Choose parents based on score
        commits = list(candidates.keys())
        scores = [candidates[commit]['accuracy_score'] for commit in commits]
        scores = [1 / (1 + math.exp(-10*(score-0.5))) for score in scores]
        probabilities = [score / sum(scores) for score in scores]
        print(commits)
        parent_commits = random.choices(commits, probabilities, k=selfimprove_size)
    elif method == 'score_child_prop':
        # 默认策略：w = sigmoid(10 * (解题率 - 0.5)) / (1 + 直接子代数)。
        # 高分节点更易入选，繁衍较少的节点也有机会；random.choices 是有放回抽样，
        # 因而同一个父代可以在一代内产生多个独立的改进尝试。
        # Choose parents based on score and the number of children
        commits = list(candidates.keys())
        scores = [candidates[commit]['accuracy_score'] for commit in commits]
        scores = [1 / (1 + math.exp(-10*(score-0.5))) for score in scores]
        children_counts = [candidates[commit]['children_count'] for commit in commits]
        children_counts = [1 / (1 + count) for count in children_counts]
        probabilities = [score * count for score, count in zip(scores, children_counts)]
        probabilities = [prob / sum(probabilities) for prob in probabilities]
        parent_commits = random.choices(commits, probabilities, k=selfimprove_size)
    elif method == 'best':
        # 阅读注意：实际 sorted 未设置 reverse=True，当前代码取的是最低分节点。
        # 这是原实现与“best”命名的不一致；中文注释仅记录现状，不改动算法。
        # Choose parents with the best score
        sorted_commits = sorted(candidates, key=lambda x: candidates[x]['accuracy_score'])
        parent_commits = sorted_commits[:min(selfimprove_size, len(sorted_commits))]
        if len(parent_commits) < selfimprove_size:
            parent_commits.extend(random.choices(parent_commits, k=selfimprove_size - len(parent_commits)))
    else:
        # Choose parents randomly
        parent_commits = random.choices(list(candidates.keys()), k=selfimprove_size)

    # Choose entries for each parent
    for parent_commit in parent_commits:
        empty_ids = candidates[parent_commit]['total_emptypatch_ids']
        resolved_ids = candidates[parent_commit]['total_resolved_ids']
        unresolved_ids = candidates[parent_commit]['total_unresolved_ids']
        
        if polyglot:
            # Polyglot 优先诊断空补丁和失败题；没有这些题时才回退到全部题目。
            entry_ids = empty_ids + unresolved_ids
            if not entry_ids:
                entry_ids = resolved_ids + empty_ids + unresolved_ids
        else:
            num_total_ids = len(empty_ids) + len(resolved_ids) + len(unresolved_ids)
            # SWE 分支依次尝试空补丁、随机性、上下文溢出三个通用改进方向。
            # 各次随机判断有先后依赖，并不是三个互斥方向各占固定的 25% 概率。

            # Solve empty patches
            if len(empty_ids) >= 0.1 * num_total_ids and random.random() < 0.25:
                entry = 'solve_empty_patches'
                selfimprove_entries.append((parent_commit, entry))
                continue

            # Solve stochasticity
            if random.random() < 0.25:
                entry = 'solve_stochasticity'
                selfimprove_entries.append((parent_commit, entry))
                continue

            # Solve context length
            if any_exceeding_context_length(output_dir, parent_commit, empty_ids + unresolved_ids) and \
                random.random() < 0.25:
                entry = 'solve_contextlength'
                selfimprove_entries.append((parent_commit, entry))
                continue

            # Choose a random unresolved entry
            if unresolved_ids == 0:
                # 注意 unresolved_ids 来自列表字段，列表与整数 0 的比较不能判断空列表；
                # 若列表为空，后面的 random.choice 会报错。此处保留原行为供学习对照。
                continue
            entry_ids = unresolved_ids
        entry = random.choice(entry_ids)
        selfimprove_entries.append((parent_commit, entry))

    return selfimprove_entries

def filter_compiled(run_ids, output_dir, num_swe_issues=[], logger=None):
    # “compiled”是评估结果完整性检查：有分数字段、有非空解题补丁、提交数够用。
    # 它既不直接调用 Python 编译器，也不要求子代分数高于父代。
    """
    Filter out runs that did not compile or have all empty patches.
    """
    run_ids_compiled = []

    logger.info(f"SWE 任务数量：{num_swe_issues}")
    for run_id in run_ids:
        metadata_path = os.path.join(output_dir, run_id, "metadata.json")
        metadata = load_json_file(metadata_path)
        logger.info(f"{run_id} 元数据：{metadata}")
        if is_compiled_self_improve(metadata, num_swe_issues=num_swe_issues, logger=logger):
            run_ids_compiled.append(run_id)
    return run_ids_compiled

def get_original_score(output_dir):
    """
    Get the original score from the initial version.
    """
    metadata = load_json_file(os.path.join(output_dir, "initial", "metadata.json"))
    return metadata["overall_performance"]["accuracy_score"]

def update_archive(output_dir, archive, new_ids, method='keep_all', noise_leeway=0.1):
    # 默认保留所有通过有效性检查的子代，包括暂时退步的分支，支持后续继续探索。
    # keep_better 比较的是“初始代理分数 - 容差”，不是该子代自己的父代分数。
    """
    Update the archive with the new self-improve runs.
    """
    if method == 'keep_better':
        # keep only better ones
        original_score = get_original_score(output_dir) - noise_leeway
        for run_id in new_ids:
            metadata = load_json_file(os.path.join(output_dir, run_id, "metadata.json"))
            score = metadata["overall_performance"]["accuracy_score"]
            if score >= original_score:
                archive.append(run_id)
    else:
        # keep everything
        archive += new_ids

    return archive

def get_full_eval_threshold(output_dir, archive):
    # 在近乎完成 small+medium+big 的历史节点中，取第二高分并设 0.4 下限。
    # 阅读注意：调用方把这个阈值传给 self_improve，但目前内层未使用该参数，
    # 所以计算阈值并不意味着代码已经触发 big 子集的完整评估。
    """
    Get the threshold for full evaluation.
    """
    archive_scores = []
    num_full_eval = sum(len(load_json_file(f"./swe_bench/subsets/{size}.json"))
                       for size in ['small', 'medium', 'big'])

    # Get original score
    original_score = get_original_score(output_dir)
    archive_scores.append(original_score)

    # Get scores from the archive
    for run_id in archive:
        metadata = load_json_file(os.path.join(output_dir, run_id, "metadata.json"))
        total_submitted_instances = metadata["overall_performance"]["total_submitted_instances"]
        # Skip if node did not have full evaluation
        if total_submitted_instances < num_full_eval * 0.9:
            continue
        score = metadata["overall_performance"]["accuracy_score"]
        archive_scores.append(score)

    # Get threshold, second highest score
    threshold = sorted(archive_scores, reverse=True)[1] if len(archive_scores) > 1 else archive_scores[0]
    # Ensure threshold is at least 0.4
    threshold = max(threshold, 0.4)

    return threshold

def main():
    # 本入口负责多代调度；单次自修改的容器生命周期和评估交给 self_improve。
    parser = argparse.ArgumentParser(description="Darwin Godel Machine!")
    parser.add_argument("--max_generation", type=int, default=80, help="Maximum number of evolution iterations.")
    parser.add_argument("--selfimprove_size", type=int, default=2, help="Number of self-improvements attempts per DGM generation.")
    parser.add_argument("--selfimprove_workers", type=int, default=2, help="Number of parallel workers for self-improvement attempts.")
    parser.add_argument(
        "--choose_selfimproves_method", type=str, default='score_child_prop',
        choices=['random', 'score_prop', 'score_child_prop' 'best'],
        # 注意：上面相邻字符串缺少逗号，会被 Python 拼接为 score_child_propbest。
        # 默认值仍是 score_child_prop，但显式传入同名参数会被 argparse 拒绝。
        help="Method to choose self-improve attempts.",
    )
    parser.add_argument("--continue_from", type=str, default=None, help="Directory to continue the run from.")
    parser.add_argument("--update_archive", type=str, default='keep_all', choices=['keep_better', 'keep_all'], help="Method to update the archive.")
    # self-improve arguments
    parser.add_argument("--num_swe_evals", type=int, default=1, help="Number of repeated SWE evaluations to run for each self-improve attempt.")
    parser.add_argument('--post_improve_diagnose', default=False, action='store_true', help='Diagnose the self-improvement after evaluation')
    parser.add_argument("--shallow_eval", default=False, action='store_true', help="Run single shallow evaluation for self-improvement on swe.")
    parser.add_argument("--polyglot", default=False, action='store_true', help="Run single shallow evaluation for self-improvement on swe.")
    parser.add_argument("--eval_noise", type=float, default=0.1, help="Noise leeway for evaluation.")
    parser.add_argument("--no_full_eval", default=False, action='store_true', help="Do not run full evaluation on swe if a node is the top N highest performing.")
    # baselines
    parser.add_argument("--run_baseline", type=str, default=None, choices=['no_selfimprove', 'no_darwin'], help="Baseline to run.")
    args = parser.parse_args()

    # Variables for this DGM run
    if not args.continue_from:
        run_id = datetime.datetime.now().strftime("%Y%m%d%H%M%S_%f")
    else:
        run_id = os.path.basename(args.continue_from)
        
    output_dir = os.path.join("./output_dgm", run_id)
    os.makedirs(output_dir, exist_ok=True)

    # Initialize
    archive, start_gen_num = initialize_run(output_dir, prevrun_dir=args.continue_from, polyglot=args.polyglot)

    # SWE issues to consider
    if not args.polyglot:
        swe_issues_sm = load_json_file("./swe_bench/subsets/small.json")
        swe_issues_med = load_json_file("./swe_bench/subsets/medium.json")
    else:
        swe_issues_sm = load_json_file("./polyglot/subsets/small.json")
        swe_issues_med = load_json_file("./polyglot/subsets/medium.json")

    # Set up logger
    logger = setup_logger(os.path.join(output_dir, "dgm_outer.log"))
    logger.info(f"开始 DGM 实验 {run_id}，参数：{vars(args)}")
    logger.info(f"档案：{archive}")
    test_more_threshold = 0.4
    # Run the DGM
    for gen_num in range(start_gen_num, args.max_generation):
        # 代数从 0 开始；本代所有父代均从更新前的同一份历史档案中选择。
        # Choose self-improve attempts
        selfimprove_entries = choose_selfimproves(
            output_dir, archive, args.selfimprove_size,
            method=args.choose_selfimproves_method,
            run_baseline=args.run_baseline,
            polyglot=args.polyglot,
        )
        logger.info(f"第 {gen_num} 代的自改进任务：{selfimprove_entries}")

        # Run self-improvement processes
        selfimprove_ids = []
        with ThreadPoolExecutor(max_workers=args.selfimprove_workers) as executor:
            futures = [
                executor.submit(
                    self_improve,
                    parent_commit=parent_commit,
                    output_dir=output_dir,
                    force_rebuild=False,
                    num_evals=args.num_swe_evals,
                    post_improve_diagnose=args.post_improve_diagnose,
                    entry=entry,
                    test_task_list=swe_issues_sm,
                    test_more_threshold=None if args.shallow_eval else test_more_threshold,
                    test_task_list_more=None if args.shallow_eval else swe_issues_med,
                    polyglot=args.polyglot,
                    full_eval_threshold=None if args.no_full_eval else get_full_eval_threshold(output_dir, archive),
                    run_baseline=args.run_baseline,
                )
                for parent_commit, entry in selfimprove_entries
            ]

            for future in as_completed(futures):
                # as_completed 先等待 future 完成，再调用 result；因此此处的 timeout
                # 并不构成运行任务的 1.5 小时硬期限，容器内 timeout 才实际约束执行时间。
                try:
                    # Added timeout to avoid hanging indefinitely (1.5 h here)
                    metadata = future.result(timeout=1.5*60*60)
                    selfimprove_ids.append(metadata['run_id'])
                except TimeoutError:
                    logger.error("自改进尝试超时。")
                    future.cancel()  # Optionally cancel the future if it's still running
                except Exception as e:
                    import traceback
                    logger.error(f"自改进步骤失败：{e}")
                    logger.error(f"异常堆栈：\n{traceback.format_exc()}")

        # Update archive
        logger.info(f"正在更新档案，代数：{gen_num}")
        selfimprove_ids_compiled = filter_compiled(
            selfimprove_ids,
            output_dir,
            num_swe_issues=[len(swe_issues_sm)] if args.shallow_eval else [len(swe_issues_sm), len(swe_issues_med)], logger=logger
        )
        archive = update_archive(output_dir, archive, selfimprove_ids_compiled, method=args.update_archive, noise_leeway=args.eval_noise)

        # Save DGM state
        with open(os.path.join(output_dir, "dgm_metadata.jsonl"), "a") as f:
            # 每代追加档案快照、尝试入口及子代列表，供恢复运行和进化树可视化使用。
            # indent=2 使单条记录跨多行；load_dgm_metadata 按记录边界读取，
            # 不能简单地把这里当作“每行一个 JSON 对象”的标准 JSONL 来解析。
            f.write(json.dumps({
                "generation": gen_num,
                "selfimprove_entries": selfimprove_entries,
                "children": selfimprove_ids,
                "children_compiled": selfimprove_ids_compiled,
                "archive": archive,
            }, indent=2) + "\n")

if __name__ == "__main__":
    main()

