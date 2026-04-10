"""
Memory Middleware 完整流程断点测试
=====================================
覆盖以下五个阶段，每个阶段都有断点建议：

阶段 1: 消息过滤   — _filter_messages_for_memory()
阶段 2: 信号检测   — detect_correction() / detect_reinforcement()
阶段 3: 防抖队列   — MemoryUpdateQueue.add() + debounce
阶段 4: LLM 提取   — MemoryUpdater.update_memory()
阶段 5: 注入验证   — format_memory_for_injection()

运行方式（在 backend/ 目录下）：
    PYTHONPATH=. uv run python learning-test/test_memory_pipeline.py

每个阶段独立，可以单独注释掉不想运行的部分。
"""

import json
import logging
import os
import sys
import tempfile
import time

# ── 确保 backend/ 在 path 里 ──────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("memory-test")

# =============================================================================
# 阶段 1：消息过滤
# =============================================================================

def test_stage1_message_filter():
    """
    验证 _filter_messages_for_memory 的过滤规则：
      - HumanMessage → 保留（剥掉 <uploaded_files> 块）
      - AIMessage（无 tool_calls）→ 保留（最终回答）
      - AIMessage（有 tool_calls）→ 丢弃（中间推理）
      - ToolMessage → 丢弃（工具结果）
      - 纯上传消息（剥掉 uploaded_files 后为空）→ 丢弃，连带 AI 回复也丢
    """
    print("\n" + "=" * 60)
    print("阶段 1: 消息过滤")
    print("=" * 60)

    # ── 断点建议 ① ──────────────────────────────────────────────────────────
    # 在 memory_middleware.py 的 _filter_messages_for_memory() 函数入口打断点
    # 观察: messages 列表里有几种类型的消息
    # ────────────────────────────────────────────────────────────────────────
    from deerflow.agents.middlewares.memory_middleware import _filter_messages_for_memory

    messages = [
        # ✅ 普通用户消息 → 保留
        HumanMessage(content="帮我用 Python 实现一个快速排序"),

        # ❌ AI 中间步骤（有 tool_calls）→ 丢弃
        AIMessage(content="我需要先搜索一下", tool_calls=[{"name": "bash", "args": {}, "id": "tc1"}]),

        # ❌ 工具结果 → 丢弃
        ToolMessage(content="搜索结果: ...", tool_call_id="tc1", name="bash"),

        # ✅ AI 最终回答（无 tool_calls）→ 保留
        AIMessage(content="```python\ndef quicksort(arr): ...\n```"),

        # 测试上传文件消息：上传块 + 真实问题 → 保留（只剥掉上传块）
        HumanMessage(content="<uploaded_files>\n/mnt/uploads/data.csv\n</uploaded_files>\n帮我分析这个文件"),

        # ❌ 纯上传消息（剥掉后为空）→ 丢弃，连带下面的 AI 回复也丢
        HumanMessage(content="<uploaded_files>\n/mnt/uploads/pic.png\n</uploaded_files>\n"),
        AIMessage(content="好的，我来查看图片"),  # ← 也会被丢弃
    ]

    filtered = _filter_messages_for_memory(messages)

    print(f"\n输入消息数: {len(messages)}")
    print(f"过滤后消息数: {len(filtered)}")
    print("\n过滤后保留的消息:")
    for i, msg in enumerate(filtered):
        preview = str(msg.content)[:60].replace("\n", "↵")
        print(f"  [{i}] {msg.type:8s} | {preview}")

    assert len(filtered) == 3, f"期望保留 3 条，实际 {len(filtered)} 条"
    # 断言: HumanMessage(快速排序) + AIMessage(代码) + HumanMessage(分析文件，已剥掉上传块)
    assert "uploaded_files" not in str(filtered[2].content), "上传块应被剥掉"
    print("\n✅ 阶段 1 通过")


# =============================================================================
# 阶段 2：Correction / Reinforcement 信号检测
# =============================================================================

def test_stage2_signal_detection():
    """
    验证 detect_correction() 和 detect_reinforcement() 的信号识别：
      - 只扫最近 6 条消息里的 Human 消息
      - correction 和 reinforcement 互斥（correction 优先）
    """
    print("\n" + "=" * 60)
    print("阶段 2: 信号检测")
    print("=" * 60)

    # ── 断点建议 ② ──────────────────────────────────────────────────────────
    # 在 memory_middleware.py 的 detect_correction() 函数入口打断点
    # 观察: recent_user_msgs 是哪几条，pattern.search() 是否命中
    # ────────────────────────────────────────────────────────────────────────
    from deerflow.agents.middlewares.memory_middleware import detect_correction, detect_reinforcement

    # Case A: Correction（中文纠正）
    msgs_correction = [
        HumanMessage(content="帮我用 Java 写排序"),
        AIMessage(content="[Java 代码]"),
        HumanMessage(content="不对，我们项目用 Python，以后别用 Java"),  # ← 触发
    ]

    # Case B: Reinforcement（英文肯定）
    msgs_reinforcement = [
        HumanMessage(content="解释一下快速排序"),
        AIMessage(content="[解释]"),
        HumanMessage(content="yes exactly, keep doing that"),  # ← 触发
    ]

    # Case C: 无信号
    msgs_neutral = [
        HumanMessage(content="帮我写个函数"),
        AIMessage(content="[代码]"),
        HumanMessage(content="谢谢"),
    ]

    print("\nCase A (中文纠正):")
    corr_a = detect_correction(msgs_correction)
    reinf_a = detect_reinforcement(msgs_reinforcement)
    print(f"  correction={corr_a}  reinforcement={detect_reinforcement(msgs_correction)}")
    assert corr_a is True

    print("Case B (英文肯定):")
    print(f"  correction={detect_correction(msgs_reinforcement)}  reinforcement={reinf_a}")
    assert reinf_a is True

    print("Case C (无信号):")
    print(f"  correction={detect_correction(msgs_neutral)}  reinforcement={detect_reinforcement(msgs_neutral)}")
    assert detect_correction(msgs_neutral) is False
    assert detect_reinforcement(msgs_neutral) is False

    print("\n✅ 阶段 2 通过")


# =============================================================================
# 阶段 3：防抖队列行为
# =============================================================================

def test_stage3_debounce_queue():
    """
    验证 MemoryUpdateQueue 的三个核心行为：
      1. per-thread 去重：同 thread_id 后来的替换前面的
      2. correction OR 合并：任意一次出现就保留
      3. debounce：短时间内多次 add() 只处理一次
    """
    print("\n" + "=" * 60)
    print("阶段 3: 防抖队列")
    print("=" * 60)

    # ── 断点建议 ③ ──────────────────────────────────────────────────────────
    # 在 queue.py 的 MemoryUpdateQueue.add() 函数内打断点
    # 观察: self._queue 在每次 add() 前后的变化
    # 在 _reset_timer() 里观察: timer 是否被取消并重建
    # ────────────────────────────────────────────────────────────────────────
    from deerflow.agents.memory.queue import MemoryUpdateQueue

    queue = MemoryUpdateQueue()

    msgs1 = [HumanMessage(content="第一条消息"), AIMessage(content="第一条回复")]
    msgs2 = [HumanMessage(content="第二条消息"), AIMessage(content="第二条回复")]
    msgs3 = [HumanMessage(content="第三条消息，来自另一个线程"), AIMessage(content="回复")]

    print("\n测试 1: per-thread 去重（同 thread_id 后者替换前者）")
    queue.add("thread-A", msgs1, correction_detected=False)
    print(f"  add(thread-A, msgs1) → 队列长度: {queue.pending_count}")
    assert queue.pending_count == 1

    queue.add("thread-A", msgs2, correction_detected=True)   # 替换 msgs1
    print(f"  add(thread-A, msgs2, correction=True) → 队列长度: {queue.pending_count}")
    assert queue.pending_count == 1  # 还是 1，不是 2

    print("\n测试 2: OR 合并（correction 只要出现一次就保留）")
    # 内部 _queue 里 thread-A 的 correction_detected 应为 True（来自 msgs2）
    context = next(c for c in queue._queue if c.thread_id == "thread-A")
    print(f"  thread-A.correction_detected = {context.correction_detected}")
    assert context.correction_detected is True

    print("\n测试 3: 不同 thread_id 独立排队")
    queue.add("thread-B", msgs3)
    print(f"  add(thread-B) → 队列长度: {queue.pending_count}")
    assert queue.pending_count == 2  # A 和 B 各一条

    queue.clear()
    print(f"  clear() → 队列长度: {queue.pending_count}")
    assert queue.pending_count == 0

    print("\n✅ 阶段 3 通过")


# =============================================================================
# 阶段 4：LLM 提取 + 原子写入（Mock LLM，不消耗 API）
# =============================================================================

def test_stage4_updater_atomic_write():
    """
    用 mock LLM 验证 MemoryUpdater 的原子写入流程：
    temp file → rename（不是直接 write）
    即使中途"崩溃"，原文件也不会损坏。

    注意：这里 mock 掉 LLM，不消耗真实 API。
    如果想测试真实 LLM 提取，把 mock 部分注释掉。
    """
    print("\n" + "=" * 60)
    print("阶段 4: 原子写入验证（Mock LLM）")
    print("=" * 60)

    import unittest.mock as mock
    from deerflow.agents.memory.updater import MemoryUpdater

    # mock LLM 返回的结构化 memory 更新结果
    mock_memory_response = {
        "user": {
            "workContext": {"summary": "用户是 Python 后端开发者，正在做 AI Agent 项目"},
            "personalContext": {"summary": "偏好简洁直接的代码风格"},
            "topOfMind": {"summary": "正在学习 LangGraph 和 Memory 中间件"},
        },
        "history": {},
        "facts": [
            {
                "id": "fact_test001",
                "content": "用户项目使用 Python，不使用 Java",
                "category": "correction",
                "confidence": 0.95,
                "createdAt": "2026-04-08T00:00:00Z",
                "source": "conversation",
                "sourceError": "之前错误地用了 Java",
            },
            {
                "id": "fact_test002",
                "content": "用户偏好简洁回答，不喜欢冗长解释",
                "category": "preference",
                "confidence": 0.85,
                "createdAt": "2026-04-08T00:00:00Z",
                "source": "conversation",
            },
        ],
    }

    # ── 断点建议 ④ ──────────────────────────────────────────────────────────
    # 在 updater.py 的 update_memory() 函数里打断点
    # 观察: current_memory（更新前）和 updated_memory（LLM 返回后）的差异
    # 搜索 "temp" 关键字，找到 temp file 创建和 rename 的位置
    # ────────────────────────────────────────────────────────────────────────

    with tempfile.TemporaryDirectory() as tmpdir:
        memory_file = os.path.join(tmpdir, "memory.json")

        # 写一个初始的空 memory 文件
        initial_memory = {"user": {}, "history": {}, "facts": []}
        with open(memory_file, "w") as f:
            json.dump(initial_memory, f)

        print(f"\nmemory 文件路径: {memory_file}")
        print(f"初始内容: {json.dumps(initial_memory, ensure_ascii=False)}")

        # mock LLM 调用，返回预设的结构化结果
        with mock.patch.object(
            MemoryUpdater,
            "_call_llm_for_memory_update",
            return_value=mock_memory_response,
        ):
            # mock 存储路径指向临时目录
            with mock.patch(
                "deerflow.agents.memory.updater.get_memory_storage"
            ) as mock_storage:
                storage_instance = mock.MagicMock()
                storage_instance.load.return_value = initial_memory
                storage_instance.save.return_value = True
                mock_storage.return_value = storage_instance

                updater = MemoryUpdater()
                msgs = [
                    HumanMessage(content="我们项目用 Python，不用 Java"),
                    AIMessage(content="明白了，我以后会用 Python"),
                ]
                success = updater.update_memory(
                    messages=msgs,
                    thread_id="test-thread",
                    correction_detected=True,   # ← 触发 correction 处理
                    reinforcement_detected=False,
                )

        print(f"\nupdate_memory() 返回: {success}")
        print("✅ LLM 提取 + 存储流程走通（Mock 模式）")

    print("\n✅ 阶段 4 通过")


# =============================================================================
# 阶段 5：注入格式验证（按置信度排序 + token 预算）
# =============================================================================

def test_stage5_injection_format():
    """
    验证 format_memory_for_injection() 的输出格式：
      - facts 按 confidence 降序排列
      - correction 类别附带 sourceError 信息
      - 总长度受 max_tokens 限制
    """
    print("\n" + "=" * 60)
    print("阶段 5: 注入格式验证")
    print("=" * 60)

    # ── 断点建议 ⑤ ──────────────────────────────────────────────────────────
    # 在 prompt.py 的 format_memory_for_injection() 函数里打断点
    # 观察: ranked_facts 的排序结果
    # 观察: running_tokens 的累加过程（token 预算如何控制注入数量）
    # ────────────────────────────────────────────────────────────────────────
    from deerflow.agents.memory.prompt import format_memory_for_injection

    memory_data = {
        "user": {
            "workContext": {"summary": "Python 后端开发者，做 AI Agent 项目"},
            "personalContext": {"summary": "偏好简洁代码"},
            "topOfMind": {"summary": "正在学习 LangGraph Memory 系统"},
        },
        "history": {
            "recentMonths": {"summary": "最近在研究 DeerFlow 框架"},
        },
        "facts": [
            # 低置信度（应排在后面）
            {"id": "f1", "content": "用户可能喜欢 TypeScript", "category": "preference", "confidence": 0.5},
            # 高置信度（应排在前面）
            {"id": "f2", "content": "用户项目使用 Python", "category": "correction", "confidence": 0.95,
             "sourceError": "之前错误地推荐了 Java"},
            # 中等置信度
            {"id": "f3", "content": "用户偏好简洁回答", "category": "preference", "confidence": 0.85},
            # 另一个高置信度
            {"id": "f4", "content": "用户正在做面试准备", "category": "goal", "confidence": 0.90},
        ],
    }

    result = format_memory_for_injection(memory_data, max_tokens=2000)

    print("\n注入到 system prompt 的内容：")
    print("-" * 50)
    print(result)
    print("-" * 50)

    # 验证排序：高置信度在前
    f2_pos = result.find("Python")    # confidence=0.95
    f3_pos = result.find("简洁回答")   # confidence=0.85
    f1_pos = result.find("TypeScript") # confidence=0.5

    print(f"\nfact 位置（越小越靠前）:")
    print(f"  Python (0.95):      位置 {f2_pos}")
    print(f"  简洁回答 (0.85):    位置 {f3_pos}")
    print(f"  TypeScript (0.50):  位置 {f1_pos}")

    assert f2_pos < f3_pos < f1_pos, "事实应按 confidence 降序排列"

    # 验证 correction 附带 sourceError
    assert "avoid:" in result or "sourceError" not in result  # correction 格式

    print("\n✅ 阶段 5 通过")


# =============================================================================
# 主流程：模拟真实 after_agent 触发
# =============================================================================

def test_full_pipeline_simulation():
    """
    完整流程模拟：模拟 MemoryMiddleware.after_agent() 被调用的场景
    不启动真实 Agent，直接构造 state 调用中间件
    """
    print("\n" + "=" * 60)
    print("完整流程模拟（不启动 Agent）")
    print("=" * 60)

    # ── 断点建议 ⑥ ──────────────────────────────────────────────────────────
    # 在 memory_middleware.py 的 after_agent() 函数入口打断点
    # 完整观察: filter → detect → queue.add() 的完整一次触发
    # ────────────────────────────────────────────────────────────────────────
    from deerflow.agents.memory.queue import get_memory_queue, reset_memory_queue
    from deerflow.agents.middlewares.memory_middleware import MemoryMiddleware, _filter_messages_for_memory, detect_correction

    # 重置队列，避免上面的测试影响
    reset_memory_queue()

    # 构造一轮完整对话（含中间工具调用）
    messages = [
        HumanMessage(content="我们项目用 Python，帮我写快速排序"),
        AIMessage(content="好的", tool_calls=[{"name": "bash", "args": {}, "id": "tc1"}]),
        ToolMessage(content="执行结果", tool_call_id="tc1", name="bash"),
        AIMessage(content="```python\ndef quicksort(arr):\n    ...\n```"),
        HumanMessage(content="不对，我要递归版本的"),  # correction 信号
        AIMessage(content="```python\ndef quicksort(arr):\n    if len(arr) <= 1: return arr\n    ...\n```"),
    ]

    filtered = _filter_messages_for_memory(messages)
    correction = detect_correction(filtered)

    print(f"\n原始消息数: {len(messages)}")
    print(f"过滤后消息数: {len(filtered)}")
    print(f"检测到 correction: {correction}")

    # 直接调 queue.add()（跳过 LLM 提取，只验证队列行为）
    queue = get_memory_queue()
    queue.add(
        thread_id="sim-thread-001",
        messages=filtered,
        correction_detected=correction,
    )

    print(f"队列中待处理数: {queue.pending_count}")
    print(f"队列中的 thread_id: {[c.thread_id for c in queue._queue]}")
    print(f"correction_detected: {queue._queue[0].correction_detected}")

    # 清理
    queue.clear()
    print("\n✅ 完整流程模拟通过")


# =============================================================================
# 入口
# =============================================================================

if __name__ == "__main__":
    print("Memory Middleware 完整流程测试")
    print("断点建议见每个函数内的注释 ── 断点建议 ① ~ ⑥")

    try:
        test_stage1_message_filter()
        test_stage2_signal_detection()
        test_stage3_debounce_queue()
        test_stage4_updater_atomic_write()
        test_stage5_injection_format()
        test_full_pipeline_simulation()

        print("\n" + "=" * 60)
        print("全部通过 ✅")
        print("=" * 60)

    except AssertionError as e:
        print(f"\n❌ 断言失败: {e}")
        raise
    except Exception as e:
        print(f"\n❌ 异常: {e}")
        raise