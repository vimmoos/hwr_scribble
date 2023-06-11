import cv2
from hwr.segmenter.types import *
from hwr.segmenter.lines import transforms as lin_seg_tx
import numpy as np
from functools import reduce
import operator
import logging


def open_cv_contours_finder(X: NDArray) -> List[Cv2Contours]:
    return cv2.findContours(X.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]


def no_reconstruct_lines(_X: NDArray,
                         raw_strips: OrderedDict[int, NDArray],
                         _comp_desc: Dict[int, ComponentDescriptor],
                         _comp_index: OrderedDict[input, List[int]]
                         ) -> OrderedDict[int, NDArray]:
    return raw_strips


def simple_reconstruct_lines(_X: NDArray,
                             _raw_strips: OrderedDict[int, NDArray],
                             comp_desc: Dict[int, ComponentDescriptor],
                             comp_index: OrderedDict[input, List[int]]
                             ) -> OrderedDict[int, NDArray]:
    # _X = ppr.s.X
    # comp_desc = ppr.s.conn_comp_descs
    # comp_index = ppr.s.contour_index_to_components

    line_keys = list(comp_index.keys())
    out_structure = OrderedDict()
    for curr_line in line_keys:
        line_rec = np.zeros_like(_X)
        for comp in comp_index[curr_line]:
            line_rec += _X * comp_desc[comp]["comp_map"]
        out_structure[curr_line] = lin_seg_tx.ProjectionCutter(complement=0, margin=0, pad=0)(line_rec)
    return out_structure


def comp_split_reconstruct_lines(_X: NDArray,
                                 _raw_strips: OrderedDict[int, NDArray],
                                 comp_desc: Dict[int, ComponentDescriptor],
                                 comp_index: OrderedDict[input, List[int]],
                                 cut_a_b: Tuple[float, float] = (1 / 3, 0.4)
                                 ) -> OrderedDict[int, NDArray]:
    # _X = ppr.s.X
    # comp_desc = ppr.s.conn_comp_descs
    # comp_index = ppr.s.contour_index_to_components

    line_keys = list(comp_index.keys())
    out_structure = OrderedDict()
    for i, curr_line in enumerate(line_keys):
        out_structure[curr_line] = []
        prev_line = None if i == 0 else line_keys[i - 1]
        next_line = None if i == len(line_keys) - 1 else line_keys[i + 1]
        # print(prev_line, '[',curr_line,']', next_line)

        for comp in comp_index[curr_line]:
            mask = comp_desc[comp]["comp_map"].copy()
            _, y_center = comp_desc[comp]["xy_center"]
            # print(f"\t{i}--{comp}")

            h_min, w_min, h_max, w_max = comp_desc[comp]["props"].bbox
            cut_above = int(h_min + (h_max - h_min) * cut_a_b[0])
            cut_below = int(h_min + (h_max - h_min) * cut_a_b[1])
            if prev_line is not None and next_line is not None and comp in comp_index[prev_line] and comp in comp_index[
                next_line]:
                mask[:cut_above, :] = False
                mask[cut_below:, :] = False
                # print("found in both")
            elif prev_line is not None and comp in comp_index[prev_line]:
                # print("found in above")
                cut_y = int(y_center)
                # print(cut_y)
                mask[:cut_above, :] = False
            elif next_line is not None and comp in comp_index[next_line]:
                # print("found in below")
                cut_y = int(y_center)
                # print(cut_y)
                mask[cut_below:, :] = False

            out_structure[curr_line].append(_X * mask)

        # reduce the partial images
        if len(out_structure[curr_line]):
            out_structure[curr_line] = lin_seg_tx.ProjectionCutter(complement=0, margin=0, pad=0)(
                reduce(operator.add, out_structure[curr_line])
            )
        else:
            logging.warning("Found line structre with no components, dropping... #%s cont-index=%s", i, curr_line)
            del out_structure[curr_line]
    # reduce the partial images
    # for k in out_structure:
    #     out_structure[k] = sum(out_structure[k])

    return out_structure


def key_top_down_right_left(cont):
    "Give the sorting key for a Cv2Contour"
    cont = cont.reshape(-1, 2)
    vert = cont.min(axis=0)
    horiz = cont.max(axis=0)
    # print(cont, vert, horiz)
    _, min_y = vert
    max_x, _ = horiz
    return min_y, -max_x


def sort_contours_for_lines_right_left(conts):
    "Sort Cv2Contours from top to bottom breaking ties right to left"
    return sorted(conts, key=key_top_down_right_left)


def contours_cut(img, contours, dropped_indexs_store=None):
    cuts = OrderedDict()
    # TODO DROP the Projection cutter here and rather use just non-zero approach on the mask
    cutter = lin_seg_tx.ProjectionCutter(complement=0, margin=0, pad=0)
    for idx, cont in enumerate(contours):
        mask = np.zeros_like(img)
        cv2.drawContours(mask, contours, idx, 255, -1)
        out = np.zeros_like(img)
        out[mask == 255] = ~img[mask == 255]
        cutout = cutter(out)
        if img.shape != cutout.shape:
            cuts[idx] = cutout
        else:
            logging.info("Discarding abnormal cutout, contour idx=%s", idx)
            if dropped_indexs_store is not None:
                dropped_indexs_store.append(idx)

    return cuts


def line_blur(line, h_factor, v_factor):
    h_size = line.shape[1] // h_factor
    v_size = line.shape[0] // v_factor
    return cv2.blur(line, ksize=(h_size, v_size))
