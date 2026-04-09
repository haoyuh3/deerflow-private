"""
用户输入一段话 → 模型调用 bash tool → Sandbox 执行 的完整流程演示

运行前提：
    - 项目根目录有 config.yaml 并配置好模型
    - cd backend
    - PYTHONPATH=. uv run python learning-test/test_sandbox_via_chat.py

关键断点位置（在 IDE 里打）：
    1. deerflow/sandbox/tools.py          bash_tool()              ← 工具被调用
    2. deerflow/sandbox/tools.py          ensure_sandbox_initialized()  ← lazy acquire
    3. deerflow/sandbox/local/local_sandbox.py  execute_command()  ← 实际执行命令
    4. deerflow/sandbox/local/local_sandbox.py  _resolve_paths_in_command()  ← 路径翻译
"""

import logging

from deerflow.client import DeerFlowClient

# 打开 DEBUG 可以看到 SandboxMiddleware 的 acquire/release 日志
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s [%(name)s] %(message)s",
)

# 只关注沙箱相关日志
logging.getLogger("deerflow.sandbox").setLevel(logging.DEBUG)

# ============================================================
# 方式1：stream 模式，可以看到模型一步步的工具调用过程
# ============================================================
def run_with_stream():
    print("\n" + "=" * 60)
    print("方式1: stream 模式（可看到工具调用过程）")
    print("=" * 60)

    client = DeerFlowClient()

    # 这句话会强制模型调用 bash 工具
    user_input = "用 bash 命令输出当前日期，命令：date"

    print(f"\n用户输入: {user_input}\n")
    print("-" * 40)

    for event in client.stream(user_input, thread_id="sandbox-demo-001"):
        event_type = event.get("event")

        if event_type == "messages-tuple":
            data = event.get("data", {})
            msg_type = data.get("type")
            content = data.get("content", "")

            if msg_type == "AIMessageChunk" and content:
                print(f"[AI] {content}", end="", flush=True)

            elif msg_type == "ToolMessage":
                tool_name = data.get("name", "")
                print(f"\n[Tool Result: {tool_name}]\n{content}")

            elif msg_type == "AIMessage":
                tool_calls = data.get("tool_calls", [])
                for tc in tool_calls:
                    print(f"\n[Tool Call: {tc.get('name')}]")
                    print(f"  args: {tc.get('args')}")

        elif event_type == "end":
            print("\n\n[Stream 结束]")


# ============================================================
# 方式2：chat 模式（同步，只返回最终结果）
# ============================================================
def run_with_chat():
    print("\n" + "=" * 60)
    print("方式2: chat 模式（同步，只看最终结果）")
    print("=" * 60)

    client = DeerFlowClient()

    user_input = "帮我在工作区创建一个 hello.txt 文件，内容是 Hello from Sandbox，然后读出来给我看"

    print(f"\n用户输入: {user_input}\n")
    print("-" * 40)

    # chat() 只收集 type=="ai" 的消息，如果模型调用了 ask_clarification
    # 会被 ClarificationMiddleware 拦截成 ToolMessage，chat() 拿不到，返回空字符串
    # 所以改用 stream() 收集所有类型的消息
    ai_text = ""
    clarification = ""

    for event in client.stream(user_input, thread_id="sandbox-demo-003"):
        if event.type != "messages-tuple":
            continue
        msg_type = event.data.get("type", "")
        content = event.data.get("content", "")

        if msg_type == "ai" and content:
            ai_text = content  # 正常 AI 回复

        elif msg_type == "tool" and event.data.get("name") == "ask_clarification":
            clarification = content  # 模型发起的澄清问题

    if clarification:
        print(f"\n[模型在问你]: {clarification}")
        print("提示: 换一个更明确的指令，避免模型反问，例如：")
        print('  user_input = "直接用 bash 执行：echo Hello from Sandbox > /mnt/user-data/workspace/hello.txt && cat /mnt/user-data/workspace/hello.txt"')
    elif ai_text:
        print(f"\n最终回复:\n{ai_text}")
    else:
        print("\n[无回复]")


if __name__ == "__main__":
    # 选择运行哪种方式
    run_with_chat()   # 简单，只看结果
    # run_with_stream()   # 推荐，可以看到工具调用过程ß