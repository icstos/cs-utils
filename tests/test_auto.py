from utils.auto import Auto, KeyboardEvent
import time

# ============ 模块级 __main__ 演示 ============
if __name__ == "__main__":
    # print(f"鼠标位置: {Auto.Mouse.get_position()}")

    # # 简单演示：显示一个 alert 对话框

    # Auto.Msg.alert("Auto 重构完成", "pynput + flet 自动化工具已就绪")
    # # 键盘监听演示 (3 秒)
    # print("\n开始键盘监听 (3 秒)，请按键...")
    # events: list[KeyboardEvent] = []
    # with Auto.KeyBoard.listen(on_press=events.append):
    #     time.sleep(3)b
    # print(f"监听到 {len(events)} 个键盘事件： {events}")
    # Auto.show_msg('你好')
    time.sleep(2)
    Auto.KeyBoard.hotkey('win', 'd')
