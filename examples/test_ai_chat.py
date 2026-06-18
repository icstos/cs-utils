from utils.ai_chat import AiChat


def main() -> None:
    chat = AiChat()
    print(chat.list_models())
    print(chat.chat('Hello, how are you?'))
    for chunk in chat.chat('Hello again', stream=True):
        print(chunk, end='', flush=True)
    print()


if __name__ == '__main__':
    main()
