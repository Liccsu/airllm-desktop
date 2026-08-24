"""引擎结构化事件日志。

引擎进程统一把事件以单行 JSON 写到 stdout，桌面应用逐行解析并转发到界面：

.. code-block:: text

    {"event":"serve.started","data":{...}}

非事件行（库自身打印、警告）由调用方按原文转发，不中断解析。
"""

from __future__ import annotations

import json
import sys
import time
from typing import Any


def emit(event: str, **fields: Any) -> None:
    """输出一个结构化事件到 stdout。"""

    record: dict[str, Any] = {"event": event, "ts": time.time()}
    record.update(fields)
    print(json.dumps(record, ensure_ascii=False, separators=(",", ":")), flush=True)


def emit_error(event: str, message: str, **fields: Any) -> None:
    """输出一个失败事件，message 为可读取的中文描述。"""

    emit(event, message=message, **fields)
