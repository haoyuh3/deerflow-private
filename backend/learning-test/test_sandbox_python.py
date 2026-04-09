"""
直接使用 LocalSandbox 运行 Python 代码
不需要 Agent，不需要 LLM，直接测试沙箱底层能力

运行：
    cd backend
    PYTHONPATH=. uv run python learning-test/test_sandbox_python.py
"""

import tempfile
from pathlib import Path
from deerflow.sandbox.local.local_sandbox import LocalSandbox, PathMapping


def main():
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir) / "workspace"
        workspace.mkdir()

        # 创建沙箱，映射虚拟路径
        sandbox = LocalSandbox(
            "local",
            path_mappings=[
                PathMapping(
                    container_path="/mnt/user-data/workspace",
                    local_path=str(workspace),
                    read_only=False,
                )
            ],
        )

        print("=== 1. 执行简单 Python 代码 ===")
        output = sandbox.execute_command("python3 -c \"print('Hello from Sandbox!')\"")
        print(output)

        print("=== 2. 写文件再读取 ===")
        sandbox.write_file("/mnt/user-data/workspace/hello.py", """
x = [i ** 2 for i in range(5)]
print("squares:", x)
""")
        output = sandbox.execute_command("python3 /mnt/user-data/workspace/hello.py")
        print(output)

        print("=== 3. 读文件内容 ===")
        content = sandbox.read_file("/mnt/user-data/workspace/hello.py")
        print(content)

        print("=== 4. 列出目录 ===")
        entries = sandbox.list_dir("/mnt/user-data/workspace")
        print(entries)


if __name__ == "__main__":
    main()