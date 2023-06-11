from .core import PageProcessor
import hwr.segmenter.lines.transforms as lin_seg_tx
from functools import partial

from . import implem

from hwr.segmenter import conn_comps


def page_processor_factory():
    return PageProcessor(
        # first page load transform
        page_load_tx=lin_seg_tx.ComposeTx([
            lin_seg_tx.SimpleResizer(),
            lin_seg_tx.DeskewAndPadTx2(),
            lin_seg_tx.ProjectionCutter(pad=50),
            lambda x: ~x,
        ]),

        # transform for the contour finder
        line_contours_find_tx=lin_seg_tx.ComposeTx([
            partial(implem.line_blur, h_factor=2, v_factor=100),
            lin_seg_tx.get_binary,
        ]),

        line_contours_postproc=None,  # here we should drop contours that are too small
        line_contours_sorter=implem.sort_contours_for_lines_right_left,

        # connected comps analysis transform
        conn_comp_tx=lin_seg_tx.ComposeTx([
            lin_seg_tx.get_binary
        ]),
        line_relative_conn_comp_sorter=conn_comps.sort_comps_for_lines_right_left,

        # this module can modify the structure of the the lines before
        # DEFAULT: no recons, output the raw line strips from the PathFinding step
        # line_recons=simple_reconstruct_lines
        line_recons=implem.comp_split_reconstruct_lines
    )
