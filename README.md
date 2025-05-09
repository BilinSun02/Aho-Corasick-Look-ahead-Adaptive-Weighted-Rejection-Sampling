# Getting started
Download or clone this code base, `cd` to it, and run `python -m pip install -e .` from there.

Then you can play with `examples`. These examples are hopefully self-explanatory enough to serve
to explain how you should use the `AhoCorasickSampler` class.

# Caveat
This sampler does not support batch inference with batch size greater than 1. It also does not
support training through it. For model evaluation and RL algorithms that only need "good samples",
however, this is probably enough.
