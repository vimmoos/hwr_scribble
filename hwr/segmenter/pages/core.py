import logging
from dataclasses import dataclass, field
from typing import Optional

from benedict import OrderedBeneDict
from scipy.spatial import ConvexHull

from hwr.plot import utils as vizutils
from hwr.segmenter import conn_comps
from hwr.segmenter.types import *
from hwr.segmenter.words.core import segment_words as default_segment_words
from .implem import (
    open_cv_contours_finder,
    no_reconstruct_lines,
    contours_cut,
    sort_contours_for_lines_right_left,
)
from ..lines.types import ImgTx

import matplotlib.pyplot as plt
import numpy as np
import distinctipy


@dataclass
class PageProcessor:
    INSTANCE_COUNT = 0

    # transform to apply on image
    page_load_tx: Optional[ImgTx] = None

    # transform to apply before find contours
    line_contours_find_tx: Optional[ImgTx] = None
    line_contours_finder: ContoursFinder = open_cv_contours_finder
    # post process contours
    line_contours_postproc: Optional[Cv2ContoursTx] = None
    # sort the contours
    line_contours_sorter: Optional[Cv2ContoursTx] = sort_contours_for_lines_right_left

    # transform to apply to image before analyzing conn-comps
    # this shoudld ensure binarization
    conn_comp_tx: Optional[ImgTx] = None

    line_relative_conn_comp_sorter: Optional[ComponentsSorter] = None

    # -- parameter numbers ---
    conn_comp_line_intersect_area_threshold: Optional[float] = 0.05

    # -- intermediated outputs manager --
    s: OrderedBeneDict = field(default_factory=OrderedBeneDict, init=False)

    # --
    line_recons: LineRecons = no_reconstruct_lines

    # --
    word_segmentator: WordSegmentator = default_segment_words

    def __post_init__(self):
        self.instnace_number = type(self).INSTANCE_COUNT
        type(self).INSTANCE_COUNT += 1
        self.log = logging.getLogger(f"PageProcessor-{self.instnace_number}")

    def _ifdef(self, tx, X):
        return tx(X) if tx is not None else X

    def __call__(self, input):
        self.s.clear()
        # raw input
        self.s.input = input
        self.log.info("input shape: %s", input.shape)

        # transformed input
        self.s.X = self._ifdef(self.page_load_tx, self.s.input)
        self.log.info("X shape: %s", self.s.X.shape)

        # transform X for contour finding
        self.s.cont_input = self._ifdef(self.line_contours_find_tx, self.s.X)
        self.log.info("cont_input shape: %s", self.s.cont_input.shape)

        # find line contours
        self.s._contours = self.line_contours_finder(self.s.cont_input)
        self.log.info("Identified %s contours", len(self.s._contours))

        # refine/postrpocess contours
        self.s.contours = self._ifdef(self.line_contours_postproc, self.s._contours)
        self.log.info("Contours after postprocessor: %s", len(self.s.contours))

        # sort contours
        self.s.sorted_contours = self._ifdef(self.line_contours_sorter, self.s.contours)

        self.s.dropped_cont_indexes = []
        self.s.raw_line_strips = contours_cut(~self.s.X,
                                              self.s.sorted_contours,
                                              dropped_indexs_store=self.s.dropped_cont_indexes)

        self.s.dropped_cont_indexes = set(self.s.dropped_cont_indexes)
        self.log.info("Produced %s raw line strips (dropped %s)", len(self.s.raw_line_strips),
                      len(self.s.dropped_cont_indexes))

        # CONN COMP
        self.s.conn_comp_input = self._ifdef(self.conn_comp_tx, self.s.X)
        self.log.info("comp_input shape: %s", self.s.conn_comp_input.shape)

        self.s.conn_comp_descs, self.s.conn_comp_meta = conn_comps.label_regions(self.s.conn_comp_input)
        self.log.info("Identified %s connected components", len(self.s.conn_comp_descs))

        self.s.contour_index_to_components = OrderedDict()

        for i, cont in enumerate(self.s.sorted_contours):
            intersecting_components = conn_comps.get_components_in_contour(
                self.s.conn_comp_descs,
                cont,
                area_thresh_perc=self.conn_comp_line_intersect_area_threshold)
            sorted_comps = self._ifdef(self.line_relative_conn_comp_sorter, intersecting_components)
            self.log.info("Contour #%s contains %s conn-comps", i, len(intersecting_components))

            if i not in self.s.dropped_cont_indexes:
                self.s.contour_index_to_components[i] = [comp["label"] for comp in sorted_comps]

        self.s.lino_img_index = self.line_recons(self.s.X,
                                                 self.s.raw_line_strips,
                                                 self.s.conn_comp_descs,
                                                 self.s.contour_index_to_components)

        self.s.lino_word_imgs = OrderedDict()
        self.s.lino_unsegm_word_imgs = OrderedDict()

        for lino, img in self.s.lino_img_index.items():
            try:
                word_imgs = self.word_segmentator(img)
                self.s.lino_word_imgs[lino] = word_imgs
            except Exception as e:
                self.s.lino_unsegm_word_imgs[lino] = img
                self.log.warning("Caught error while segmenting line %s: %s", lino, e)

    # -----------------------------------------
    # -----------------------------------------
    # ---------- Diagnosis helpers ------------

    def show_contours(self):
        fig, ax = plt.subplots()
        ax.imshow(vizutils.draw_contours_bw(self.s.X, self.s.sorted_contours))
        return fig

    def show_raw_strips(self):
        return vizutils.show_lines(self.s.raw_line_strips.values())

    def show_raw_segm_page(self):
        fig, ax = plt.subplots()
        ax.imshow(vizutils.make_word_display_line(self.s.raw_line_strips.values()))
        return fig

    def show_raw_segm_padded(self):
        padded_strips = vizutils.pad_to_uniform_strips(self.s.raw_line_strips.values(), axis="width")
        fig, (ax1, ax2) = plt.subplots(ncols=2)
        ax1.imshow(self.s.X, cmap="Greys_r")
        ax2.imshow(vizutils.make_word_display_line(padded_strips))
        return fig

    def show_page_conn_comps(self, show_text_label=False):
        return conn_comps.label_regions_view2(self.s.conn_comp_descs, self.s.conn_comp_meta,
                                              show_text_label=show_text_label)

    def show_debug_comps(self, figsize=(10, 10)):
        # def debug_comps(X, components, figsize=(10,10)):
        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(self.s.X, cmap="Greys_r")
        colors = distinctipy.get_colors(len(self.s.contour_index_to_components))
        for ngroup, (cgroup, component_indexes) in enumerate(self.s.contour_index_to_components.items()):
            group_color = colors[ngroup]
            group_centers = []
            for comp_idx in component_indexes:
                d = self.s.conn_comp_descs[comp_idx]
                group_centers.append(tuple(reversed(d["props"].centroid)))
                # print(d)
                xy = (d["w_bounds"][0] - 1, d["h_bounds"][0] - 1)
                width = (d["w_bounds"][1] - d["w_bounds"][0]) + 1
                height = (d["h_bounds"][1] - d["h_bounds"][0]) + 1
                # print(xy, width, height)
                ax.add_patch(plt.Rectangle(xy, width=width, height=height,
                                           edgecolor=group_color,
                                           alpha=0.5,
                                           linewidth=2,
                                           facecolor="none"))

                try:
                    hull = ConvexHull(d["poly"]).points
                    ax.add_patch(plt.Polygon(hull,
                                             edgecolor=group_color,
                                             alpha=0.5,
                                             linewidth=2,
                                             facecolor=group_color))
                except Exception as e:
                    logging.warning("Could not draw hull of %s: %s", comp_idx, e)

            group_centers = np.array(group_centers)
            if len(group_centers.shape) > 1 and group_centers.shape[1] > 1:
                ax.plot(group_centers[:, 0], group_centers[:, 1], "-x", color=group_color)

        fig.tight_layout()
        return fig

    def show_recons_lines(self, figsize=(10, 10)):
        return vizutils.imshow_rows(self.s.lino_img_index.values(), figsize=figsize);

    def show_recons_segm_padded(self):
        padded_strips = vizutils.pad_to_uniform_strips(self.s.lino_img_index.values(), axis="width")
        fig, (ax1, ax2) = plt.subplots(ncols=2)
        ax1.imshow(self.s.X, cmap="Greys_r")
        ax2.imshow(vizutils.make_word_display_line(padded_strips))
        return fig

    def show_words_per_line(self, figsize=(10, 10)):
        fig, axs = plt.subplots(nrows=len(self.s.lino_word_imgs), ncols=2, figsize=figsize)

        for i, (lino, word_imgs) in enumerate(self.s.lino_word_imgs.items()):
            axs[i, 0].imshow(self.s.lino_img_index[lino])
            axs[i, 1].imshow(vizutils.make_hstack_image(reversed(word_imgs)))
            axs[i, 0].set_xticks([])
            axs[i, 1].set_xticks([])
            axs[i, 0].set_title(f"Line #{i} contour_id={lino}")

        fig.tight_layout()
        return fig
