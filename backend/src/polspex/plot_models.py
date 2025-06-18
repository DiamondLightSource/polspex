"""
Plot models for Davidia plotting

https://github.com/DiamondLightSource/davidia/blob/main/client/component/src/LinePlot.tsx
https://diamondlightsource.github.io/davidia/?path=/docs/plots-line--docs
"""
import numpy as np
from itertools import cycle
from typing import Optional
from pydantic import BaseModel

COLORS = cycle(['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])


class GlyphType:
    """
    h5web line markers (glyphs)
    https://github.com/silx-kit/h5web/blob/main/packages/lib/src/vis/line/models.ts
    """
    Circle = 'Circle'
    Cross = 'Cross'
    Square = 'Square'
    Cap = 'Cap'


class lineParams(BaseModel):
    colour: Optional[str] = None
    pointSize: int
    lineOn: bool
    glyphType: Optional[str] = None


class lineData(BaseModel):
    key: str
    lineParams: lineParams
    x: list 
    xDomain: list[float, float]
    y: list 
    yDomain: list[float, float]
    defaultIndices: bool


class plotConfig(BaseModel):
    title: str 
    xLabel: str 
    yLabel: str


class lineProps(BaseModel):
    plotConfig: plotConfig
    lineData: tuple[lineData]
    xDomain: tuple[float, float]
    yDomain: tuple[float, float]


def getGlyph(fmt: str):
    if 'x' in fmt or '+' in fmt:
        return GlyphType.Cross
    if 'o' in fmt:
        return GlyphType.Circle
    if '^' in fmt:
        return GlyphType.Cap
    if 's' in fmt:
        return GlyphType.Square
    return None


def gen_line_data(xdata: np.ndarray, ydata: np.ndarray, fmt: str = '', label: str = '', colour: str = '') -> lineData:
    return {
        'key': label,
        'lineParams': {
            'colour': colour if colour else next(COLORS),
            'pointSize': 0,
            'lineOn': True, #'-' in fmt or ':' in fmt,
            # 'glyphType': getGlyph(fmt),  
        },
        'x': xdata,
        'xDomain': (xdata.min(), xdata.max()),
        'y': ydata,
        'yDomain': (ydata.min(), ydata.max()),
        'defaultIndices': False,
    }


def gen_plot_props(title: str, xlabel: str, ylabel: str, 
                  xlim: tuple[float, float] | None, ylim: tuple[float, float] | None, 
                  *lines: lineData) -> lineProps:
    return {
        'plotConfig': {
            'title': title,
            'xLabel': xlabel,
            'yLabel': ylabel,
        },
        'lineData': lines,
        'xDomain': xlim if xlim else (
            min(line['xDomain'][0] for line in lines),
            max(line['xDomain'][1] for line in lines),
        ),
        'yDomain': ylim if ylim else (
            min(line['yDomain'][0] for line in lines),
            max(line['yDomain'][1] for line in lines),
        ),
    }
