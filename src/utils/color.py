from dataclasses import dataclass
from enum import Enum, EnumMeta
from typing import cast, overload


@dataclass
class ColorItem:
    name: str
    name_zh: str
    hex: str
    rgb: tuple[int, int, int]


class ColorMeta(EnumMeta):
    @overload
    def __getitem__(cls, item: int | str) -> ColorItem: ...
    @overload
    def __getitem__(cls, item: slice) -> list[ColorItem]: ...
    def __getitem__(cls, item: int | str | slice) -> ColorItem | list[ColorItem]:
        members = list(cls.__members__.values())
        if isinstance(item, slice):
            return [cast(ColorItem, getattr(m, "value")) for m in members[item]]
        elif isinstance(item, int):
            return cast(ColorItem, getattr(members[item], "value"))
        elif isinstance(item, str):
            return cast(ColorItem, getattr(cls.__members__[item], "value"))


class Color(Enum, metaclass=ColorMeta):
    RED = ColorItem(name="RED", name_zh="红色", hex="#f44336", rgb=(244, 67, 54))
    PINK = ColorItem(name="PINK", name_zh="粉色", hex="#e91e63", rgb=(233, 30, 99))
    PURPLE = ColorItem(name="PURPLE", name_zh="紫色", hex="#9c27b0", rgb=(156, 39, 176))
    DEEPPURPLE = ColorItem(
        name="DEEP_PURPLE", name_zh="深紫色", hex="#673ab7", rgb=(103, 58, 183)
    )
    INDIGO = ColorItem(name="INDIGO", name_zh="靛蓝", hex="#3f51b5", rgb=(63, 81, 181))
    BLUE = ColorItem(name="BLUE", name_zh="蓝色", hex="#2196f3", rgb=(33, 150, 243))
    LIGHTBLUE = ColorItem(
        name="LIGHT_BLUE", name_zh="浅蓝色", hex="#03a9f4", rgb=(3, 169, 244)
    )
    CYAN = ColorItem(name="CYAN", name_zh="青色", hex="#00bcd4", rgb=(0, 188, 212))
    TEAL = ColorItem(name="TEAL", name_zh="蓝绿色", hex="#009688", rgb=(0, 150, 136))
    GREEN = ColorItem(name="GREEN", name_zh="绿色", hex="#4caf50", rgb=(76, 175, 80))
    LIGHTGREEN = ColorItem(
        name="LIGHT_GREEN", name_zh="浅绿色", hex="#8bc34a", rgb=(139, 195, 74)
    )
    LIME = ColorItem(name="LIME", name_zh="黄绿色", hex="#cddc39", rgb=(205, 220, 57))
    YELLOW = ColorItem(name="YELLOW", name_zh="黄色", hex="#ffeb3b", rgb=(255, 235, 59))
    AMBER = ColorItem(name="AMBER", name_zh="琥珀色", hex="#ffc107", rgb=(255, 193, 7))
    ORANGE = ColorItem(name="ORANGE", name_zh="橙色", hex="#ff9800", rgb=(255, 152, 0))
    DEEPORANGE = ColorItem(
        name="DEEPORANGE", name_zh="深橙色", hex="#ff5722", rgb=(255, 87, 34)
    )
    BROWN = ColorItem(name="BROWN", name_zh="棕色", hex="#795548", rgb=(121, 85, 72))
    GREY = ColorItem(name="GREY", name_zh="灰色", hex="#9e9e9e", rgb=(158, 158, 158))
    BLUEGREY = ColorItem(
        name="BLUE_GREY", name_zh="蓝灰色", hex="#607d8b", rgb=(96, 125, 139)
    )

    @property
    def name(self) -> str:
        return self.value.name

    @property
    def name_zh(self) -> str:
        return self.value.name_zh

    @property
    def hex(self) -> str:
        return self.value.hex

    @property
    def rgb(self) -> tuple[int, int, int]:
        return self.value.rgb

    @staticmethod
    def rgb_to_hex(rgb: tuple[int, int, int]) -> str:
        """Convert RGB tuple (0-255) to hex string."""
        r, g, b = rgb
        return f"#{r:02x}{g:02x}{b:02x}"

    @staticmethod
    def hex2rgb(hex_str: str) -> tuple[int, int, int]:
        """Convert hex color string to RGB tuple."""
        value = hex_str.lstrip("#")
        lv = len(value)
        r = int(value[0 : lv // 3], 16)
        g = int(value[lv // 3 : 2 * lv // 3], 16)
        b = int(value[2 * lv // 3 :], 16)
        return (r, g, b)


if __name__ == "__main__":
    print(Color[0])
    print(Color[0].rgb)
    print(Color.RED)
    print(Color.RED.name)
    print(Color.RED.name_zh)
    print(Color.RED.hex)
    print(Color.RED.rgb)
