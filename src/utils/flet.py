import flet as ft

from .color import Color


def stroke_paint(color_idx, stroke_width=1):
    return ft.Paint(
        style=ft.PaintingStyle.STROKE,
        color=Color[color_idx].name,
        stroke_width=stroke_width,
    )


def fill_paint(color_idx):
    return ft.Paint(
        style=ft.PaintingStyle.FILL,
        color=ft.Colors.with_opacity(0.2, Color[color_idx].name),
    )


home_icon = ft.IconButton(ft.Icons.HOME, tooltip="Home", icon_color=ft.Colors.WHITE)
search_icon = ft.IconButton(
    ft.Icons.SEARCH, tooltip="Search", icon_color=ft.Colors.WHITE
)
settings_icon = ft.IconButton(
    ft.Icons.SETTINGS, tooltip="Settings", icon_color=ft.Colors.WHITE
)
