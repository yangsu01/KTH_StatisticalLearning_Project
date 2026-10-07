"""Reproduction of Beck & Teboulle (2009), ISTA and FISTA on the cameraman deblurring example.

Every value taken from the paper is cited where it is used; every choice the paper leaves open is stated where it is taken and collected in ASSUMPTIONS.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.linalg import LinearOperator, eigsh

from image_processing import ImageProcessor

# paper's experimental settings
KERNEL_SIZE = 9
BLUR_SIGMA = 4.0
NOISE_SIGMA = 1e-3
LAMBDA_CAMERAMAN = 2e-5
ITERS = 1000
PANEL_COUNTS = (100, 200)
HAAR_LEVELS = 3

# the step-size variation: our experiment
STEP_FACTORS = (
    0.5,
    0.75,
    1.0,  # t = 1/L(f), the step every theorem in the paper assumes
    1.5,
    2.5,  # past the classical bound t = 2/L(f), so both methods must break
)
STEP_ITERS = 100  # iterations per curve: enough to see which curves rise

# the noise seed (the paper publishes no realisation)
SEED = 1

IMAGE_PATH = Path("data/cameraman.tif")
FIGURES_DIR = Path("figures")

# The cameraman objectives the paper publishes: iterations 100 and 200 in the caption of Figure 2, iteration 1000 in the text of section 5.1.
PUBLISHED = {
    ("ISTA", 100): 5.44e-1,
    ("ISTA", 200): 3.60e-1,
    ("ISTA", 1000): 2.45e-1,
    ("FISTA", 100): 2.40e-1,
    ("FISTA", 200): 2.28e-1,
    ("FISTA", 1000): 2.23e-1,
}

# What the paper leaves unspecified and what we chose

ASSUMPTIONS = [
    "noise realisation: section 5.1 gives the standard deviation 1e-3 but not the seed; we fix SEED = 1",
    "initial point: 'the initial image was the blurred image' (section 5.1)",
    "Haar conventions: the paper fixes three stages and that W is the inverse transform (section 5.1)",
    "L(f): section 5.1 gives L(f) = 2 lambda_max(A^T A) but not how to obtain lambda_max; we estimate it matrix-free by Lanczos on the module's operators, which lands on 1 to about 1e-12",
    "stopping rule: no tolerance is given, only iteration counts; the runs use the counts the paper reports",
    "variation: the step-size range, outside the paper claim",
    "comparison: the paper publishes cameraman objectives only at iterations 100, 200 and 1000, so those are the only values compared",
]


class Problem(NamedTuple):
    processor: ImageProcessor
    b: np.ndarray
    x0: np.ndarray
    lam: float
    L: float


class Result(NamedTuple):
    x: np.ndarray
    Fs: np.ndarray  # Fs[k] is the objective after k iterations from x0
    seconds: float
    iters: int


class Variation(NamedTuple):
    factors: tuple[float, ...]  # step sizes as multiples of the paper's step 1/L(f)
    curves: dict[tuple[str, float], np.ndarray]  # (method, factor) -> F_k
    t_paper: float
    t_classical: float


def load_cameraman(path: Path = IMAGE_PATH):
    """Load the 256x256 cameraman image scaled to [0, 1] (section 5.1).
    Returns the processor and the image as a 2-D array.
    """
    path = Path(path)
    processor = ImageProcessor(str(path))
    image = processor.load_image().reshape(processor.original_shape, order="F")
    return processor, image


# the operators


def W(x: np.ndarray, processor: ImageProcessor) -> np.ndarray:
    """W, the paper's inverse three-stage Haar transform: coefficients -> 2-D image, the module's
    ImageProcessor.apply_W at the paper's three stages (section 5.1)."""
    return processor.apply_W(x, levels=HAAR_LEVELS)


def WT(u: np.ndarray, processor: ImageProcessor) -> np.ndarray:
    """W^T: 2-D image -> flat coefficient vector, the module's ImageProcessor.apply_W_T."""
    return processor.apply_W_T(u, levels=HAAR_LEVELS).ravel(order="F")


def A(x: np.ndarray, processor: ImageProcessor) -> np.ndarray:
    """A = R W: coefficients -> blurred image, the module's ImageProcessor.apply_A at
    the paper's settings (section 5.1)."""
    return processor.apply_A(x, KERNEL_SIZE, BLUR_SIGMA, HAAR_LEVELS)


def AT(z: np.ndarray, processor: ImageProcessor) -> np.ndarray:
    """A^T = W^T R^T, the module's ImageProcessor.apply_A_T."""
    return processor.apply_A_T(z, KERNEL_SIZE, BLUR_SIGMA, HAAR_LEVELS)


# the problem


def f(x: np.ndarray, b: np.ndarray, processor: ImageProcessor) -> float:
    """The smooth term, f(x) = ||A x - b||^2."""
    return np.linalg.norm(A(x, processor) - b) ** 2


def g(x: np.ndarray, lam: float) -> float:
    """The nonsmooth term, g(x) = lam ||x||_1."""
    return lam * np.abs(x).sum()


def F(x: np.ndarray, b: np.ndarray, lam: float, processor: ImageProcessor) -> float:
    """The objective eq. (1.3)): F(x) = f(x) + g(x)."""
    return f(x, b, processor) + g(x, lam)


def grad_f(x: np.ndarray, b: np.ndarray, processor: ImageProcessor) -> np.ndarray:
    """grad f(x) = 2 A^T (A x - b) (section 2.2, Example 2.2)."""
    return 2 * AT(A(x, processor) - b, processor)


def observe(
    image: np.ndarray,
    processor: ImageProcessor,
    sigma: float = NOISE_SIGMA,
    seed: int = SEED,
) -> np.ndarray:
    """The observation b = A x + w, with x the image's wavelet coefficients.

    The noise is zero-mean white Gaussian with standard deviation 1e-3 (Beck & Teboulle,
    section 5.1); the realisation is ours, fixed by `seed`. ImageProcessor.add_noise draws from
    NumPy's global generator, so the seed is set here rather than passed to the module.
    """
    np.random.seed(seed)
    return processor.add_noise(A(WT(image, processor), processor), sigma)


def initial_point(b: np.ndarray, n: int, processor: ImageProcessor) -> np.ndarray:
    """ "The initial image was the blurred image" (section 5.1), read in
    coefficient space: x0 = W^T b, the coefficients of the blurred image."""
    return WT(b.reshape(n, n, order="F"), processor)


def lipschitz(n: int, processor: ImageProcessor) -> float:
    """L(f) = 2 lambda_max(A^T A) (section 2.2, Example 2.2).
    The largest eigenvalue is estimated matrix-free: Lanczos only asks A and A^T for matvecs
    """
    AtA = LinearOperator(
        (n * n, n * n), matvec=lambda v: AT(A(v, processor), processor), dtype=float
    )
    return 2 * eigsh(AtA, k=1, which="LA", return_eigenvectors=False)[0]


def setup(n: int = 256, lam: float = LAMBDA_CAMERAMAN, seed: int = SEED) -> Problem:
    """Build the whole problem: the image, the observation, the initial point and the Lipschitz
    constant. Use the returned bundle rather than calling it again.
    """
    processor, image = load_cameraman()
    b = observe(image, processor, seed=seed)
    return Problem(
        processor,
        b,
        initial_point(b, n, processor),
        lam,
        lipschitz(n, processor),
    )


# the algorithms


def T_alpha(v: np.ndarray, alpha: float) -> np.ndarray:
    """The shrinkage operator (eq. (1.5)):
    T_alpha(v)_i = (|v_i| - alpha)_+ sign(v_i)."""
    return np.sign(v) * np.maximum(np.abs(v) - alpha, 0)


def p_L(
    y: np.ndarray,
    b: np.ndarray,
    lam: float,
    L: float,
    processor: ImageProcessor,
) -> np.ndarray:
    """The proximal gradient step (eq (2.5)-(2.6)):
    p_L(y) = T_{lam/L}(y - grad f(y)/L)."""
    return T_alpha(y - grad_f(y, b, processor) / L, lam / L)


def ista(x0, b, lam, iters, L, processor) -> Result:
    """ISTA with a constant step, x_k = p_L(x_{k-1}) (eq. (3.1)).

    Stops early only if the iteration overflows to inf/nan, which happens for the variation's
    too-large steps.
    """
    start = time.perf_counter()
    x = x0
    Fs = [F(x0, b, lam, processor)]
    for _ in range(iters):
        x = p_L(x, b, lam, L, processor)
        Fs.append(F(x, b, lam, processor))
        if not np.isfinite(Fs[-1]):
            break
    return Result(x, np.array(Fs), time.perf_counter() - start, iters)


def fista(x0, b, lam, iters, L, processor) -> Result:
    """FISTA with a constant step (eq (4.1)-(4.3)):
    x_k = p_L(y_k), t_{k+1} = (1 + sqrt(1 + 4 t_k^2))/2,
    y_{k+1} = x_k + ((t_k - 1)/t_{k+1})(x_k - x_{k-1}), started at y_1 = x_0, t_1 = 1.

    Stops early on overflow, as in `ista`.
    """
    start = time.perf_counter()
    x_prev = x = x0
    t = 1.0
    Fs = [F(x0, b, lam, processor)]
    for _ in range(iters):
        t_next = (1 + np.sqrt(1 + 4 * t**2)) / 2
        y = x + ((t - 1) / t_next) * (x - x_prev)
        x_prev, x, t = x, p_L(y, b, lam, L, processor), t_next
        Fs.append(F(x, b, lam, processor))
        if not np.isfinite(Fs[-1]):
            break
    return Result(x, np.array(Fs), time.perf_counter() - start, iters)


# the paper's cameraman experiment (section 5.1)


def reproduction(problem: Problem, iters: int = ITERS):
    """ISTA and FISTA on the cameraman image for `iters` iterations, both at the paper's constant
    step 1/L(f)."""
    args = (problem.x0, problem.b, problem.lam, iters, problem.L, problem.processor)
    return ista(*args), fista(*args)


def figure_reconstruction(problem: Problem, counts=PANEL_COUNTS):
    """The paper's Figure 2: ISTA and FISTA reconstructions after 100 and 200 iterations."""
    panels = [(method, k) for k in counts for method in ("ISTA", "FISTA")]
    fig, axes = plt.subplots(2, 2, figsize=(6.6, 6.6))
    for ax, (method, k) in zip(axes.ravel(), panels):
        run = ista if method == "ISTA" else fista
        result = run(
            problem.x0,
            problem.b,
            problem.lam,
            k,
            problem.L,
            problem.processor,
        )
        ax.imshow(W(result.x, problem.processor), cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"{method}, {k} iterations, F = {result.Fs[-1]:.3f}", fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    return fig


def figure_objective(results, lam: float):
    """The objective against iteration on a logarithmic vertical axis, with the iteration counts
    the paper reports marked."""
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    for result, label in zip(results, ("ISTA", "FISTA")):
        ax.semilogy(result.Fs, label=label)
    for k in (100, 200, 1000):
        ax.axvline(k, color="k", ls=":", lw=0.8)
        ax.text(
            k,
            1.005,
            f"{k}",
            transform=ax.get_xaxis_transform(),
            fontsize=7,
            ha="right",
            va="bottom",
            rotation=90,
        )
    ax.set_xlim(0, len(results[0].Fs) - 1)
    ax.set_xlabel("iteration k")
    ax.set_ylabel("F(x_k)")
    ax.set_title(f"cameraman, lam = {lam:g}")
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


# the variation: the step size pushed past the paper's


def variation(
    problem: Problem, factors=STEP_FACTORS, iters: int = STEP_ITERS
) -> Variation:
    """Both methods at several constant steps, each a multiple of the paper's 1/L(f).

    The paper proves its rates for t = 1/L(f) (section 2.1) and quotes the classical gradient condition t < 2/L(f), so 1.0 is the paper's step and 2.0 is the classical bound.
    """
    t_paper = 1 / problem.L  # the step every theorem in the paper assumes
    t_classical = 2 / problem.L  # the classical convergence bound of section 2.1

    def run(method, factor) -> np.ndarray:
        """`method` at step t = factor / L(f): the step enters p_L as L = 1/t."""
        return method(
            problem.x0,
            problem.b,
            problem.lam,
            iters,
            problem.L / factor,
            problem.processor,
        ).Fs

    # the overflows of the too-large steps are expected; keep them quiet
    with np.errstate(over="ignore", invalid="ignore"):
        curves = {
            (name, factor): run(method, factor)
            for name, method in (("ISTA", ista), ("FISTA", fista))
            for factor in factors
        }

    return Variation(factors, curves, t_paper, t_classical)


def figure_variation(var: Variation):
    """The variation: F_k against the iteration at each step size, one panel per method."""
    top = 10e3
    clip = lambda v: np.clip(np.where(np.isfinite(v), v, top), 1e-12, top)

    fig, axes = plt.subplots(
        1, 2, figsize=(9.2, 3.8), layout="constrained", sharey=True
    )
    handles = []
    for ax, method in zip(axes, ("ISTA", "FISTA")):
        for factor in var.factors:
            label = f"t = {factor:g}/L(f)"
            if factor == 1.0:
                label += ", the paper's step"
            (line,) = ax.semilogy(clip(var.curves[(method, factor)]), label=label)
            handles.append(line)
        ax.set_ylim(1e-1, top / 10)
        ax.set_xlabel("iteration k")
        ax.set_title(method)
    axes[0].set_ylabel("F(x_k)")
    fig.legend(
        handles[: len(var.factors)],
        [line.get_label() for line in handles[: len(var.factors)]],
        loc="outside lower center",
        ncol=len(var.factors),
        fontsize=8,
        frameon=False,
    )
    fig.suptitle(
        "constant step t = c/L(f); c = 1 is the paper's step, c = 2 the classical bound.",
        fontsize=9,
    )
    return fig


# outputs


def save_figure(fig, name: str, outdir: Path = FIGURES_DIR) -> Path:
    """Write one figure to figures/<name>.png."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return path


def print_comparison(ist: Result, fis: Result, var: Variation) -> None:
    """The comparison table: the reproduced objectives against the ones the paper publishes."""

    print("\nreproduction, cameraman 256x256")
    print(f"{'':6s}{'k':>6s}{'ours':>13s}{'paper':>13s}")
    for method, result in (("ISTA", ist), ("FISTA", fis)):
        for k in (100, 200, 1000):
            print(
                f"{method:6s}{k:6d}{result.Fs[k]:13.3e}{PUBLISHED[(method, k)]:13.3e}"
            )

    print(
        "\nvariation, constant step t = c/L(f)  (c = 1 paper's step, c = 2 classical bound)"
    )
    print(f"{'':6s}{'c':>5s}{'t':>9s}{'iters':>7s}{'F_end':>12s}{'F_end/F_0':>11s}")
    for method in ("ISTA", "FISTA"):
        for factor in var.factors:
            Fs = var.curves[(method, factor)]
            finite = Fs[np.isfinite(Fs)]
            print(
                f"{method:6s}{factor:5g}{factor * var.t_paper:9.4f}{len(finite) - 1:7d}"
                f"{finite[-1]:12.3e}{finite[-1] / Fs[0]:11.3g}"
            )
    print(
        "iters = iterations that stayed finite; a ratio above 1 means the curve rose\n"
    )


def run() -> None:
    """Reproduce the paper's cameraman example and run the variation: write the three figures to figures/ and print the comparison table."""

    problem = setup()
    ist, fis = reproduction(problem)
    var = variation(problem)

    for fig, name in (
        (figure_reconstruction(problem), "figure2_reconstructions"),
        (figure_objective((ist, fis), problem.lam), "objective"),
        (figure_variation(var), "variation"),
    ):
        save_figure(fig, name)
    print_comparison(ist, fis, var)
