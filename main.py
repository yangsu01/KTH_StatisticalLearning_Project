"""Single entry point: reproduces both results of the report with one command.

    python main.py

writes the three figures to `figures/` and prints the two comparison tables:

    figures/figure2_reconstructions.png   the paper's Figure 2, ISTA and FISTA at 100 and 200
                                          iterations (report Figure 1)
    figures/objective.png                 objective against iteration at the paper's step
                                          (report Figure 2)
    figures/variation.png                 the step-size variation (report Figure 3)
    stdout                                the reproduction table against the published values,
                                          and the variation table

Everything is computed in `replication.py`; this file only calls it. Run from the repository
root, since the data path (`data/cameraman.tif`) and the output directory (`figures/`) are
relative.
"""

from replication import run

if __name__ == "__main__":
    run()
