"""
Sandbox 完整流程学习测试
打断点建议位置已标注 # <<< BREAKPOINT

运行方式：
    cd backend
    PYTHONPATH=. uv run python learning-test/test_sandbox_flow.py
"""

import sys
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

# ============================================================
# Step 0: 构造最小依赖，不需要真实 config.yaml
# ============================================================

# Mock config，避免必须有 config.yaml
from unittest.mock import patch

# ============================================================
# Step 1: 直接测试 LocalSandbox（最底层）
# ============================================================

def test_local_sandbox_directly():
    """直接测 LocalSandbox，不经过 Middleware，看路径翻译如何工作"""
    print("\n=== Step 1: 直接测试 LocalSandbox ===")

    from deerflow.sandbox.local.local_sandbox import LocalSandbox, PathMapping

    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir) / "workspace"
        workspace.mkdir()

        # 创建路径映射：虚拟路径 /mnt/user-data/workspace → 真实临时目录
        mappings = [
            PathMapping(
                container_path="/mnt/user-data/workspace",
                local_path=str(workspace),
                read_only=False,
            )
        ]

        sandbox = LocalSandbox("local", path_mappings=mappings)  # <<< BREAKPOINT: 看 sandbox 初始化

        # 测试路径翻译
        virtual_path = "/mnt/user-data/workspace/hello.txt"
        resolved = sandbox._resolve_path(virtual_path)           # <<< BREAKPOINT: 看路径翻译结果
        print(f"  虚拟路径: {virtual_path}")
        print(f"  真实路径: {resolved}")
        assert resolved == str(workspace / "hello.txt"), f"路径翻译错误: {resolved}"

        # 测试写文件（通过虚拟路径）
        sandbox.write_file("/mnt/user-data/workspace/hello.txt", "Hello Sandbox!")  # <<< BREAKPOINT
        content = sandbox.read_file("/mnt/user-data/workspace/hello.txt")           # <<< BREAKPOINT
        print(f"  写入并读取: {content}")
        assert content == "Hello Sandbox!"

        # 测试执行命令（命令里包含虚拟路径）
        command_with_virtual = "cat /mnt/user-data/workspace/hello.txt"
        resolved_command = sandbox._resolve_paths_in_command(command_with_virtual)  # <<< BREAKPOINT
        print(f"  原始命令:   {command_with_virtual}")
        print(f"  翻译后命令: {resolved_command}")

        output = sandbox.execute_command(command_with_virtual)  # <<< BREAKPOINT: 看执行结果
        print(f"  执行输出:   {output}")
        assert "Hello Sandbox!" in output

        print("  ✓ LocalSandbox 基础功能正常")


# ============================================================
# Step 2: 测试 SandboxProvider 生命周期
# ============================================================

def test_sandbox_provider_lifecycle():
    """测试 Provider 的 acquire/get/release 流程"""
    print("\n=== Step 2: 测试 SandboxProvider 生命周期 ===")

    from deerflow.sandbox.local.local_sandbox import LocalSandbox, PathMapping
    from deerflow.sandbox.local.local_sandbox_provider import LocalSandboxProvider
    import deerflow.sandbox.local.local_sandbox_provider as provider_module

    # 重置单例，确保干净状态
    provider_module._singleton = None

    with patch("deerflow.sandbox.local.local_sandbox_provider.LocalSandboxProvider._setup_path_mappings", return_value=[]):
        provider = LocalSandboxProvider()

        # acquire：第一次调用创建沙箱
        sandbox_id = provider.acquire(thread_id="thread-001")  # <<< BREAKPOINT: 看 acquire 返回什么
        print(f"  acquire() 返回 sandbox_id: {sandbox_id}")
        assert sandbox_id == "local"

        # acquire 再次调用：返回同一个 id（单例）
        sandbox_id_2 = provider.acquire(thread_id="thread-002")  # <<< BREAKPOINT: 单例，同一个 id
        print(f"  再次 acquire() 返回: {sandbox_id_2}（应该相同）")
        assert sandbox_id == sandbox_id_2, "LocalSandbox 应该是单例"

        # get：通过 id 取回实例
        sandbox = provider.get(sandbox_id)                      # <<< BREAKPOINT
        print(f"  get() 返回: {sandbox}")
        assert isinstance(sandbox, LocalSandbox)

        # release：Local 实现是空操作
        provider.release(sandbox_id)                            # <<< BREAKPOINT: release 不做任何事
        sandbox_after_release = provider.get(sandbox_id)
        print(f"  release 后 get() 仍然返回: {sandbox_after_release}（Local 不释放）")

        print("  ✓ Provider 生命周期正常")

    # 清理单例
    provider_module._singleton = None


# ============================================================
# Step 3: 模拟 SandboxMiddleware 的 before_agent / after_agent
# ============================================================

def test_sandbox_middleware_lifecycle():
    """模拟 Middleware 钩子，看沙箱何时被 acquire"""
    print("\n=== Step 3: 模拟 SandboxMiddleware 生命周期 ===")

    import deerflow.sandbox.local.local_sandbox_provider as provider_module
    from deerflow.sandbox.sandbox_provider import set_sandbox_provider, reset_sandbox_provider
    from deerflow.sandbox.local.local_sandbox_provider import LocalSandboxProvider
    from deerflow.sandbox.middleware import SandboxMiddleware

    # 重置
    provider_module._singleton = None
    reset_sandbox_provider()

    with patch("deerflow.sandbox.local.local_sandbox_provider.LocalSandboxProvider._setup_path_mappings", return_value=[]):
        with patch("deerflow.sandbox.sandbox_provider.get_app_config") as mock_config:
            # 让 get_sandbox_provider() 使用 LocalSandboxProvider
            mock_config.return_value.sandbox.use = "deerflow.sandbox.local.local_sandbox_provider:LocalSandboxProvider"

            provider = LocalSandboxProvider()
            set_sandbox_provider(provider)

            # lazy_init=False: before_agent 时就 acquire
            middleware = SandboxMiddleware(lazy_init=False)  # <<< BREAKPOINT

            # 模拟 state 和 runtime
            state = {}  # 初始没有 sandbox
            runtime = MagicMock()
            runtime.context = {"thread_id": "test-thread-123"}

            # 调用 before_agent（相当于一次对话开始）
            result = middleware.before_agent(state, runtime)  # <<< BREAKPOINT: 看返回值
            print(f"  before_agent 返回: {result}")
            assert result == {"sandbox": {"sandbox_id": "local"}}

            # after_agent（对话结束）
            state_with_sandbox = {"sandbox": {"sandbox_id": "local"}}
            result2 = middleware.after_agent(state_with_sandbox, runtime)  # <<< BREAKPOINT
            print(f"  after_agent 返回: {result2}（None 表示 Local 不释放）")

            print("  ✓ SandboxMiddleware 生命周期正常")

    provider_module._singleton = None
    reset_sandbox_provider()


# ============================================================
# Step 4: 测试 ensure_sandbox_initialized（lazy init 核心逻辑）
# ============================================================

def test_ensure_sandbox_initialized():
    """测试 bash_tool 内部的懒初始化逻辑"""
    print("\n=== Step 4: 测试 ensure_sandbox_initialized（lazy init）===")

    import deerflow.sandbox.local.local_sandbox_provider as provider_module
    from deerflow.sandbox.sandbox_provider import set_sandbox_provider, reset_sandbox_provider
    from deerflow.sandbox.local.local_sandbox_provider import LocalSandboxProvider
    from deerflow.sandbox.tools import ensure_sandbox_initialized

    provider_module._singleton = None
    reset_sandbox_provider()

    with patch("deerflow.sandbox.local.local_sandbox_provider.LocalSandboxProvider._setup_path_mappings", return_value=[]):
        provider = LocalSandboxProvider()
        set_sandbox_provider(provider)

        # 模拟 runtime（state 里没有 sandbox）
        runtime = MagicMock()
        runtime.state = {}  # 没有 sandbox → 触发 lazy acquire
        runtime.context = {"thread_id": "lazy-thread-001"}
        runtime.config = {}

        # 第一次调用：触发 lazy acquire
        sandbox = ensure_sandbox_initialized(runtime)           # <<< BREAKPOINT: 看 lazy acquire 过程
        print(f"  lazy acquire 结果: {sandbox}")
        print(f"  sandbox.id: {sandbox.id}")

        # 第二次调用（state 里已经有了）：直接复用
        runtime.state = {"sandbox": {"sandbox_id": "local"}}
        sandbox2 = ensure_sandbox_initialized(runtime)         # <<< BREAKPOINT: 复用，不重新 acquire
        print(f"  复用 sandbox: {sandbox2}（应与上面相同）")
        assert sandbox is sandbox2, "应该复用同一个 sandbox 实例"

        print("  ✓ ensure_sandbox_initialized 正常")

    provider_module._singleton = None
    reset_sandbox_provider()


# ============================================================
# 运行所有步骤
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Sandbox 完整流程学习测试")
    print("在关键行打断点，逐步跟踪调用链")
    print("=" * 60)

    try:
        test_local_sandbox_directly()
        test_sandbox_provider_lifecycle()
        test_sandbox_middleware_lifecycle()
        test_ensure_sandbox_initialized()
        print("\n✓ 所有步骤通过！")
    except Exception as e:
        import traceback
        print(f"\n✗ 失败: {e}")
        traceback.print_exc()
        sys.exit(1)