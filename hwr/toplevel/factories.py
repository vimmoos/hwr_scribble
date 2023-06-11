from hwr.toplevel import implem

from hwr.toplevel.types import RunContext, IWordLoader, IWordsPager, IWordStreamProcessor

from hwr import recognizer


def simple_word_pager(_: RunContext) -> IWordsPager:
    return implem.word_stream_to_pages


def simple_word_loader(_: RunContext) -> IWordLoader:
    return recognizer.WordLoader.from_path


from hwr.network import load_model, Autoencoder

import torch

# def word_proc_0(_: RunContext) -> IWordStreamProcessor:
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     model = load_model(Autoencoder)
#     model.to(device)
#     return implem.SimpleWordStreamProc(core=)
