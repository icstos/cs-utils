import difflib


class Text:
    @staticmethod
    def diff(text1: str, text2: str, ignore_list):
        # quick_ratio()比ratio()计算效率更高，计算结果一致
        # return difflib.SequenceMatcher(ignore_list, str_1, str_2).quick_ratio()
        return difflib.SequenceMatcher(lambda x: x in ignore_list, text1, text2).ratio()


if __name__ == "__main__":
    str_1 = "hello world"
    str_2 = "hello python"
    str_3 = "hello world2"
    ignore_list = [" "]
    print(Text.diff(str_1, str_2, ignore_list))
    print(Text.diff(str_1, str_3, ignore_list))
