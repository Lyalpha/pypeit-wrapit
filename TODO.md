# General

* Add `uv`-based installation instructions.
* Containerize.


# LT_SPRAT

* Currently only allows use of archival sensisitivty function, or to pass
    your own sensitivity function file. Needs functionality to detect standard in input
    files and derive sensitivity function on the fly.
* Be clearer that a run should contain the necessary science and arc files for a single target
    in a single observation set-up.
* Add a overview handler to take a large directory/list of SPRAT files and organise into
    individual runs per target/setup, calling the existing function as neeeded.
* Log to file as well as console.
* Log PypeIt outputs to file.
* Review weighted mean function, use native numpy function.
* Add tests!


# NOT_ALFOSC

Not implemented yet.