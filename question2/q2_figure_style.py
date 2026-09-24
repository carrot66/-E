"""Consistent CJK typography for reproducible Problem 2 figures."""
from __future__ import annotations

import matplotlib as mpl
from matplotlib import font_manager
from matplotlib.ft2font import FT2Font


FONT_CANDIDATES = (
    "Microsoft YaHei", "Noto Sans CJK SC", "Source Han Sans SC",
    "SimHei", "WenQuanYi Micro Hei", "WenQuanYi Zen Hei",
)
GLYPH_CHECK = "问题二附件三验证集鲁棒性情感识别文本语音视觉缺失比例训练轮次分类预测强度负向中性正向准确率散点共同模态置信度"


def use_chinese_font() -> dict[str, str]:
    """Select a font with the Chinese glyphs used in Q2, or fail visibly."""
    required = {ord(char) for char in GLYPH_CHECK}
    for family in FONT_CANDIDATES:
        for font in font_manager.fontManager.ttflist:
            if font.name != family:
                continue
            if required.issubset(FT2Font(font.fname).get_charmap()):
                mpl.rcParams.update({
                    "font.family": "sans-serif",
                    "font.sans-serif": [family, "DejaVu Sans"],
                    "axes.unicode_minus": False,
                    "pdf.fonttype": 42,
                    # Paths avoid missing Chinese glyphs in SVG on other machines.
                    "svg.fonttype": "path",
                    "axes.spines.right": False,
                    "axes.spines.top": False,
                    "axes.axisbelow": True,
                    "axes.linewidth": .8,
                    "legend.frameon": False,
                })
                return {"family": family, "file": font.fname}
    raise RuntimeError("未找到覆盖问题2中文标签的字体；请安装微软雅黑、思源黑体或文泉驿微米黑")
