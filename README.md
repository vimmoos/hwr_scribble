# Dead Sea Scroll Project

## Installation

### Virtual Environment

Create and source a virtualenv enviroment.

I usally do it as follows, but any virtualenv strategy works:

1. `$ virtualenv --python=/usr/bin/python3.10 ~/.pyenvs/hwr`

2. `source ~/.pyenvs/hwr/bin/activate`

### Dependencies

Install project dependecies via pip from the project root

1. `pip install -r reqs.txt`

## Running the full pipeline

There are several pipeline configuration available but the one we selected as best
based on the edit distance to the provided test set transcription can be run
via the simple entrypoint in `./main.py`.

The default usage is as required: just pass the path to the input images folder as
positional argument of the cli invocation. The progream will produce results in "./results"

### The help

``` 
❯ python ./main.py --help
                                                                             
 Usage: main.py [OPTIONS] [INPUT]                                            
                                                                             
 :param input: :param output: :param diag_plots: :return:                    
                                                                             
╭─ Arguments ───────────────────────────────────────────────────────────────╮
│   input      [INPUT]  Path to the input images [default: None]            │
╰───────────────────────────────────────────────────────────────────────────╯
╭─ Options ─────────────────────────────────────────────────────────────────╮
│ --output                           DIRECTORY  Path where to produce the   │
│                                               output transcriptions       │
│                                               [default: results]          │
│ --diag-plots    --no-diag-plots               [default: no-diag-plots]    │
│ --help                                        Show this message and exit. │
╰───────────────────────────────────────────────────────────────────────────╯
```
