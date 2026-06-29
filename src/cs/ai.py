from dataclasses import dataclass
from cs.my_types import AIMode


@dataclass
class AI:
    mode: AIMode = AIMode.ASSISTANT
    status: str = "running"  # 状态
    mood: float = 70.0  # 心情指数
    interst: float = 60.0  # 兴致：兴趣指数
    with_internet: bool = False  # 是否联网

    def say(self, msg: str):
        print(msg)

    def run(self, command: str):
        print(command)
