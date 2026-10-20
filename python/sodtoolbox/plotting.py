"""Figures with LaTeX text (Computer Modern) and a fixed categorical palette.

palette='matlab' uses MATLAB's default ColorOrder (dark green #055432 replaces green next to orange or yellow,
plum #67355C replaces purple next to blue); palette='nld' uses the Nonlinear Dynamics Laboratory figure series.
Reference curves are dashed grey. Save as vector PDF.
"""
import os, shutil

__all__ = ['set_style', 'PALETTES', 'REF', 'color', 'ref']

PALETTES = {
    'matlab': ['#0072BD', '#D95319', '#EDB120', '#67355C', '#055432', '#4DBEEE', '#A2142F'],
    'nld': ['#1F5A85', '#B4622A', '#588A75', '#CA863E', '#774860'],
}
REF = {'matlab': '#808080', 'nld': '#8A8578'}
_STATE = {'palette': 'matlab'}


def set_style(palette='matlab', usetex=None, size=10):
    """LaTeX text and the chosen palette; usetex=None uses LaTeX when it is installed."""
    import matplotlib as mpl
    if os.path.isdir('/Library/TeX/texbin') and '/Library/TeX/texbin' not in os.environ.get('PATH', ''):
        os.environ['PATH'] += os.pathsep + '/Library/TeX/texbin'
    if usetex is None:
        usetex = shutil.which('latex') is not None
    _STATE['palette'] = palette
    mpl.rcParams.update({
        'text.usetex': usetex, 'font.family': 'serif', 'font.serif': ['Computer Modern Roman'],
        'mathtext.fontset': 'cm', 'text.latex.preamble': r'\usepackage{amsmath}\usepackage{bm}',
        'axes.prop_cycle': mpl.cycler(color=PALETTES[palette]),
        'font.size': size, 'axes.labelsize': size, 'legend.fontsize': size - 1,
        'xtick.labelsize': size - 1, 'ytick.labelsize': size - 1,
        'axes.linewidth': 0.6, 'lines.linewidth': 1.3, 'figure.dpi': 120,
        'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02, 'legend.frameon': False,
        'axes.facecolor': 'white', 'figure.facecolor': 'white'})
    return usetex


def color(i):
    p = PALETTES[_STATE['palette']]
    return p[i % len(p)]


def ref():
    return REF[_STATE['palette']]
