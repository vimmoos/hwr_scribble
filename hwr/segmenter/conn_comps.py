from .types import *
import numpy as np
import matplotlib.pyplot as plt
from shapely import Polygon
import logging


def label_regions(X) -> Tuple[Dict[int, ComponentDescriptor], ComponentsMeta]:
    blob_map = measure.label(X, background=0)
    blob_lables = set(np.unique(blob_map))
    props_by_id = {p.label: p for p in measure.regionprops(blob_map)}

    tot_p = X.shape[0] * X.shape[1]

    desc = {}
    for comp in blob_lables - {0}:
        props = props_by_id[comp]
        polygon = props.coords[:, ::-1]
        # filter the image to only values equal to the component label
        comp_map = blob_map == comp  # the mask
        h_vals, w_vals = np.where(comp_map == True)

        h_min, h_max = h_bounds = min(h_vals), max(h_vals)
        w_min, w_max = w_bounds = min(w_vals), max(w_vals)

        # slicer = (slice(*h_bounds), slice(*w_bounds))
        # print(slicer, props.slice)

        cut = (X * comp_map)[props.slice]
        strip_cut = (X * comp_map)[props.slice[0], :]

        xy_center = (w_min + (w_max - w_min) / 2, h_min + (h_max - h_min) / 2)

        desc[comp] = ComponentDescriptor(label=comp,

                                         pix_count=props.num_pixels,
                                         pix_perc=props.num_pixels / tot_p,

                                         h_bounds=h_bounds,
                                         w_bounds=w_bounds,

                                         area=(h_max - h_min) * (w_max - w_min),
                                         xy_center=xy_center,

                                         comp_map=comp_map,
                                         cut=cut,
                                         strip_cut=strip_cut,
                                         poly=polygon,
                                         props=props)

    return desc, ComponentsMeta(X=X, blob_map=blob_map, blob_lables=blob_lables)


def label_regions_view2(desc, meta, figsize=(10, 10), show_text_label=False):
    components = meta["blob_lables"]
    fig, ax = plt.subplots(figsize=figsize)

    ax.imshow(meta["blob_map"], cmap="nipy_spectral")

    for i, c in enumerate(components):
        if c > 0:
            d = desc[c]
            # print(d)
            xy = (d["w_bounds"][0] - 1, d["h_bounds"][0] - 1)
            width = (d["w_bounds"][1] - d["w_bounds"][0]) + 1
            height = (d["h_bounds"][1] - d["h_bounds"][0]) + 1
            # print(xy, width, height)
            ax.add_patch(plt.Rectangle(xy, width=width, height=height,
                                       edgecolor="white",
                                       facecolor="none"))
            if show_text_label:
                ax.text(xy[0], xy[1], f"Comp-{c}", fontdict={"color": "white"})

    fig.tight_layout()
    return fig


def cont_box(cont):
    cont_path = cont.reshape(-1, 2)
    max_x, max_y = np.max(cont_path, axis=0)
    min_x, min_y = np.min(cont_path, axis=0)
    return (min_y, max_y), (min_x, max_x)


def get_components_in_contour(comp_desc, contour, area_thresh_perc=None):
    cont_path = contour.reshape(-1, 2)
    if len(cont_path) < 3:
        return []

    cont_poly = Polygon(cont_path).convex_hull
    matches = []

    for desc in comp_desc.values():
        if len(desc["poly"]) > 2:
            comp_poly = Polygon(desc["poly"]).convex_hull
            try:
                inters = cont_poly.intersection(comp_poly)
                if inters is None or inters.area == 0 or inters.is_empty:
                    continue
                # if there is some intersection within the line contour hull

                # and the intersection is larger than the required percentage of the comp's area
                if area_thresh_perc is not None and (inters.area / comp_poly.area) < area_thresh_perc:
                    logging.info(
                        "Component %s interescts only marginally with contour, dropping... intersect area perc=%s",
                        desc["label"], (inters.area / comp_poly.area)
                    )
                    continue
                # then record the match
                matches.append(desc)
            except Exception as e:
                logging.warn("Shapely error for polygon of component: %s, %s", desc["label"], e)
                pass

    return matches


def key_comp_right_left(comp_desc):
    horiz = comp_desc["w_bounds"]
    _, max_x = horiz
    return max_x


def sort_comps_for_lines_right_left(comp_descs):
    return sorted(comp_descs, key=key_comp_right_left)
