class ErrorUnsgementableWord(Exception):
    pass


# used once
def img_batch_to_tensor_batch(Xs):
    return th.Tensor(np.array(Xs)).unsqueeze(1)


# unused
def slices_to_bound(slices):
    start = min([s.start for s in slices])
    end = max([s.stop for s in slices])
    return start, end


#
def merge_slices(slices):
    return slice(*slices_to_bound(slices))
    # start, end = slices_to_bound(slices)
    # return slice(start, end)


@dataclass
class WindowOverSegment:
    pixel_lookahed: int = 3

    s: OrderedBeneDict = field(default_factory=OrderedBeneDict)

    WORD_DISCARD_BIN: Optional[List["WindowOverSegment"]] = None

    def __call__(self, X):
        try:
            self.s.input = X
            self.s.vpp = np.sum(X, axis=0)
            self.s.vpp_unit = shift_in_unit(self.s.vpp)
            _, min_peaks = peakdetect(
                self.s.vpp_unit, lookahead=self.pixel_lookahed
            )
            self.s.min_peaks_xy = np.array(min_peaks)
            self.s.min_peaks_x = self.s.min_peaks_xy[:, 0]

            self.s.bounds = [0] + list(self.s.min_peaks_x) + [X.shape[1]]
            self.s.x_slices = [
                slice(int(mmin), int(mmax))
                for mmin, mmax in list(
                    reversed(list(zip(self.s.bounds, self.s.bounds[1:])))
                )
            ]

            self.s.img_slices = [X[:, sl] for sl in self.s.x_slices]

        except Exception as e:
            # raise ErrorUnsgementableWord(e, self)
            self.WORD_DISCARD_BIN.append(self)
            self.s.img_slices = []
            self.s.x_slices = []

        return self.s.img_slices.copy()

    def get_current_slices(self):
        return self.s.x_slices.copy()

    def show_debug(self):
        fig, (ax1, ax2) = plt.subplots(2, 1)
        tit = "Window oversegment debug"
        ax1.plot(self.s.input.shape[0] * (1 - self.s.vpp_unit))

        ax1.imshow(self.s.input)
        if len(self.s.min_peaks_xy):
            ax1.scatter(
                self.s.min_peaks_xy[:, 0],
                self.s.input.shape[0] * (1 - self.s.min_peaks_xy[:, 1]),
                marker="x",
                color="red",
            )
            ax2.imshow(
                vizutils.make_hstack_image(
                    reversed(self.s.img_slices), sep_width=1
                )
            )
        else:
            tit += " (no min-peaks)"
            ax2.set_axis_off()
        fig.suptitle(tit)
        return fig


def get_chars2(
    X,
    model,
    window_tx: Callable[[NDArray], NDArray],
    MIN_WIN=10,
    MAX_WIN=40,
):
    wos = WindowOverSegment()
    # pass the image to to obtain a list of images
    all_slices = list(wos(X))
    all_slices_meta = wos.get_current_slices()

    print(f"Starting with {len(all_slices)} slices")
    # now we must pass frames to classifer untill we exaust them
    # each time we pass the first N windows to

    selections = []  # this is the returned object

    # create coupled dequeues
    img_slices = deque(all_slices.copy())
    img_slices_meta = deque(all_slices_meta.copy())

    it = -1
    # meta_slices = all_slices_meta.copy()
    while img_slices:
        it += 1
        print(f"------- Iteration {it} -------")
        batch = []
        batch_meta = []

        # take the first in queue, this is always consumes from the queue so this eventually terminates
        curr = img_slices.popleft()
        curr_meta = img_slices_meta.popleft()

        parts = deque([])  # accumulate the other used parts
        # the first one will be the last to appear, so they are in the
        # order in which they should be put back in (via append left)
        # in the main queue (img_slices) if they end up not being used
        # which means we can 'append-left' back if not used in this order
        parts_meta = deque([])

        # untill we reach above MIN_WIN then keep stacking
        while curr.shape[1] < MIN_WIN and img_slices:
            print(
                f"Window w={curr.shape[1]} below MIN_WIN={MIN_WIN}, enlarging..."
            )

            _next = img_slices.popleft()
            _next_meta = img_slices_meta.popleft()

            parts.appendleft(_next)
            parts_meta.appendleft(_next_meta)

            curr = np.hstack((_next, curr))
            curr_meta = merge_slices((_next_meta, curr_meta))

        # if the img_slices finished before reaching MIN_WIN
        # then curr contains what was there, but we don't produce the frame at all
        if curr.shape[1] < MIN_WIN:
            print("Last window below MIN_WIN")
            break

        print("Created first batch frame, consumed slices:", len(parts))
        batch.append(curr)
        batch_meta.append(curr_meta)

        # note that the first frame is allowed to escape MAX_WIN, but will not produce attionals

        print("Generating additional fram up to max width")

        # while we are belew max and can take other slices
        while curr.shape[1] <= MAX_WIN and img_slices:
            _next = img_slices.popleft()  # take the next slice
            _next_meta = img_slices_meta.popleft()

            new_curr = np.hstack((_next, curr))
            new_curr_meta = merge_slices((_next_meta, curr_meta))

            # if with this we go out of bounds then put it back
            if new_curr.shape[1] > MAX_WIN:
                img_slices.appendleft(_next)  # put back
                img_slices_meta.appendleft(_next_meta)
                break  # we are done

            parts.appendleft(_next)  # register the part
            parts_meta.appendleft(_next_meta)

            print(f"Creating {len(batch)}-th batch frame")

            batch.append(new_curr)
            batch_meta.append(new_curr_meta)

            curr = new_curr
            curr_meta = new_curr_meta

        # ok now we have a batch with at least 1 item, possibly more,
        # where items have min width MIN_WIN and max MAX_WIN

        print(f"Preparing batch of size={len(batch)} valid slices")
        converted_frames = [window_tx(x) for x in batch]
        tensor_batch = img_batch_to_tensor_batch(
            [frame / 255 for frame in converted_frames]
        )
        print("Created tensor batch: ", tensor_batch.shape)

        print("Running classifier...")
        with uq_utils.confidences(model, n=100, only_confidence=False) as f:
            rvotes, _, imgs = f(tensor_batch)

        # extract the predicted classes
        cls = [Counter(x).most_common(1)[0][0] for x in rvotes]
        # extract the relative support for the selected classes
        rvalues = np.array([max(rvote.values()) for rvote in rvotes])

        # argmax on the confidence, this tells us what slice index to chose
        # 0 being use just the first frame, and e.g. 1 use the first 2 and so on
        chosen_slice_index = np.argmax(np.array(rvalues))
        # chosen_slice = slicez[chosen_slice_index]
        chosen_conf = rvalues[chosen_slice_index]
        chosen_label = cls[chosen_slice_index]
        chosen_bounds = batch_meta[chosen_slice_index]

        # img_bounds = slices_to_bound(meta_slices[slicez[chosen_slice_index]])

        print(
            f"Chosen slice: #{chosen_slice_index} conf = {chosen_conf} label = {chosen_label}"
        )
        selections.append(
            BeneDict(
                {
                    "char": batch[chosen_slice_index],
                    # "tensor": tensor_batch[chosen_slice_index],
                    "frame": converted_frames[chosen_slice_index],
                    "class": chosen_label,
                    "conf": chosen_conf,
                    "bounds": chosen_bounds,
                    "recons": imgs[chosen_slice_index],
                    # what was this chosen among?
                    "meta": {
                        "batch": batch,
                        "frames": converted_frames,
                        "clss": cls,
                        "rvalues": rvalues,
                        "avg_recons": np.mean(imgs, axis=0),
                    },
                }
            )
        )

        # put those that where not chosen
        _put_back_idx = -chosen_slice_index if chosen_slice_index else None
        put_back = list(parts)[:_put_back_idx]
        put_back_meta = list(parts_meta)[:_put_back_idx]

        for part, part_meta in zip(put_back, put_back_meta):
            img_slices.appendleft(part)
            img_slices_meta.appendleft(part_meta)

        print(
            f"Put back {len(put_back)} slices, Remaining slices:",
            len(img_slices),
        )

    return selections


@dataclass
class WordProcessor:
    core: Callable[[NDArray], Tuple[List[int], Any]]

    def __call__(self, w: Word):
        w.labels, w.meta = self.core(w.img)
        return w


def process_words_stream(
    wp: WordProcessor, stream: Iterable[Word]
) -> Generator[Word, None, None]:
    for w in stream:
        yield wp(w)


@dataclass
class WordProcCore0:
    model: Any
    window_tx: Callable[[NDArray], NDArray]
    meta_keys = ["conf", "meta.avg_recons", "bounds"]

    def __call__(self, X: NDArray) -> Tuple[List[int], Any]:
        ret = get_chars2(X, self.model, self.window_tx)
        pred_y = [r["class"] for r in ret]
        # pred_names = [CLASS_NAMES[y] for y in pred_y]
        return pred_y, [sel_keys(r, self.meta_keys) for r in ret]
