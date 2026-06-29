from cs.auto import Auto, KeyboardEvent
import time
import asyncio


def test_keyboard_ctrl_triggers_callback():
    """Start a keyboard listener, simulate pressing Ctrl, and expect 'test' printed."""

    def on_press(evt: KeyboardEvent):
        if "ctrl" in evt.key.lower():
            print("test")

    with Auto.KeyBoard.listen(on_press=on_press) as events:
        # Auto.KeyBoard.press("ctrl")
        time.sleep(5)

    # captured = capfd.readouterr()
    # assert "test" in captured.out


import threading


def cb(evt: KeyboardEvent):
    print("evt:", evt)


def run_listener(stop_evt: threading.Event):
    with Auto.KeyBoard.listen(on_press=cb):
        stop_evt.wait()  # 阻塞直到 stop_evt.set()


stop = threading.Event()
t = threading.Thread(target=run_listener, args=(stop,), daemon=True)
t.start()

# # 主程序继续运行
# # ...
# # 停止监听：
# stop.set()
# t.join()
# =========s=== 模块级 __main__ 演示 ============
if __name__ == "__main__":
    test_keyboard_ctrl_triggers_callback()
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
    # time.sleep(2)
    # Auto.KeyBoard.hotkey('win', 'd')
    # def on_ctrl_press(key):
    #     if key == 'ctrl':
    #         print('test')

    # listener = Auto.KeyBoard.listen(
    #     on_press=on_ctrl_press,
    # )
