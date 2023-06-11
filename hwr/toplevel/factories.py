from hwr.toplevel import implem
from hwr.toplevel.types import (
    RunContext,
    IWordLoader,
    IWordStreamProcessor,
    IWordsPager,
    IPagesWriter
)
from hwr import recognizer
from hwr.network import load_model, Autoencoder
from hwr.data_proc.char_proc import real_txs
import torch

from hwr import network


def simple_word_loader(_: RunContext) -> IWordLoader:
    return recognizer.WordLoader.from_path


def word_proc_0(_: RunContext) -> IWordStreamProcessor:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(Autoencoder)
    model.to(device)
    return implem.SimpleWordStreamProc(core=implem.WordProcCore0(
        model=model,
        window_tx=real_txs,
        torch_device=device
    ))


def word_proc_1(_: RunContext) -> IWordStreamProcessor:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(Autoencoder, network.model_aug)
    model.to(device)
    return implem.SimpleWordStreamProc(core=implem.WordProcCore0(
        model=model,
        window_tx=real_txs,
        torch_device=device
    ))


def simple_word_pager(_: RunContext) -> IWordsPager:
    return implem.word_stream_to_pages


def simple_unicode_writer(ctx: RunContext) -> IPagesWriter:
    return implem.UnicodePageWriter(
        log=ctx.log,
        rm_dest=ctx.clean_start
    )
