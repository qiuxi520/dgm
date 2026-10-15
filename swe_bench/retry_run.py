"""复用已生成的 DGM 补丁，准备环境或重试 SWE-bench 评估。"""
import argparse
import json
from pathlib import Path

import docker
from datasets import load_dataset
from swebench.harness.docker_build import build_env_images, build_instance_image
from swebench.harness.test_spec import make_test_spec

from self_improve_step import run_harness_swe
from swe_bench.build_config import configure_build_client
from swe_bench.utils import setup_logger as setup_build_logger
from utils.common_utils import load_json_file
from utils.docker_utils import setup_logger
from utils.evo_utils import get_model_patch_paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--node', required=True, help='含 metadata.json 和 model_patch.diff 的节点目录')
    parser.add_argument('--eval_workers', type=int, default=1)
    parser.add_argument('--prepare-only', action='store_true', help='仅构建评估镜像，不调用模型')
    args = parser.parse_args()
    if args.eval_workers < 1:
        parser.error('--eval_workers 必须大于等于 1')
    root = Path(__file__).resolve().parents[1]
    node = Path(args.node).resolve()
    metadata = load_json_file(node / 'metadata.json')
    tasks = load_json_file(root / 'swe_bench/subsets/small.json')
    if args.prepare_only:
        dataset = load_dataset('princeton-nlp/SWE-bench_Verified')['test']
        specs = [make_test_spec(entry) for entry in dataset if entry['instance_id'] in tasks]
        client = configure_build_client(docker.from_env())
        try:
            _, failed = build_env_images(client, specs, force_rebuild=False, max_workers=1)
            if failed:
                raise RuntimeError(f'环境镜像构建失败：{failed}')
            for spec in specs:
                spec.repo_script_list.insert(0, 'git config --global http.version HTTP/1.1')
                logger = setup_build_logger(node / f'{spec.instance_id}_prepare.log')
                build_instance_image(spec, client, logger, nocache=False)
        finally:
            client.close()
        print('全部评估镜像准备成功，可以复用原补丁重试评估。')
        return
    patches = get_model_patch_paths(str(root), str(node.parent), metadata['parent_commit'])
    patches.append(str(node / 'model_patch.diff'))
    setup_logger(node / 'reevaluate.log')
    try:
        run_harness_swe(
            metadata['entry'], metadata['run_id'], patches, 1, str(node),
            metadata, metadata['run_id'], None, tasks, None,
            eval_workers=args.eval_workers,
        )
    except Exception as error:
        metadata['evaluation_error'] = str(error)
        raise
    else:
        metadata.pop('evaluation_error', None)
    finally:
        (node / 'metadata.json').write_text(json.dumps(metadata, indent=4, ensure_ascii=False))
    print('原补丁评估完成，结果已保存到节点 metadata.json；进化档案未自动变更。')


if __name__ == '__main__':
    main()
