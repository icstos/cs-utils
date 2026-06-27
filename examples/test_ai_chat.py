from utils.ai_chat import AiChat


def main() -> None:
    chat = AiChat()
    print(chat.list_models())
    print(chat.chat("Who are you?"))
    print(chat.chat("你是谁？"))
    for chunk in chat.chat("Hello again", stream=True):
        print(chunk, end="", flush=True)
    print()


if __name__ == "__main__":
    main()
