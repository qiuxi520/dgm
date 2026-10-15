"""SWE-bench 构建共享的内存和网络配置，不修改外部 SWE-bench 仓库。"""
import os
import threading
from urllib.parse import urlsplit


# 外层有多个自改进线程时，也只允许一个镜像构建流运行。
_BUILD_LOCK = threading.Lock()


def configure_build_client(client):
    """配置流式构建：串行执行，并将宿主机代理传给构建步骤。"""
    original_build = client.api.build
    proxies = {}
    for upper in ('HTTP_PROXY', 'HTTPS_PROXY', 'NO_PROXY'):
        value = os.getenv(upper) or os.getenv(upper.lower())
        if value:
            proxies[upper] = value
            proxies[upper.lower()] = value
    network = os.getenv('SWE_BUILD_NETWORK')
    if not network and any(
        urlsplit(proxies.get(key, '')).hostname in ('localhost', '127.0.0.1', '::1')
        for key in ('HTTP_PROXY', 'HTTPS_PROXY')
    ):
        # Linux 中容器的 localhost 默认并非宿主机；host 模式可访问本机代理。
        network = 'host'

    def build(**kwargs):
        options = dict(kwargs)
        options['buildargs'] = {**proxies, **options.get('buildargs', {})}
        if network:
            options.setdefault('network_mode', network)

        def stream():
            with _BUILD_LOCK:
                yield from original_build(**options)
        return stream()

    client.api.build = build
    return client
