#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import math
import time
from typing import Callable, NamedTuple, Tuple

import numpy as np
import gpytorch
import torch
from botorch.acquisition.multi_objective.monte_carlo import (
    qExpectedHypervolumeImprovement,
)
from botorch.models.deterministic import GenericDeterministicModel
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.utils.gp_sampling import get_gp_samples
from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.utils.multi_objective.box_decompositions.box_decomposition import (
    BoxDecomposition,
)
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    NondominatedPartitioning,
    FastNondominatedPartitioning,
)
from botorch.utils.sampling import sample_simplex
from botorch.utils.transforms import normalize, unnormalize
from .state import TRBOState
from .utils import (
    decay_function,
    get_indices_in_hypercube,
    get_indices_in_hypersphere,
    sample_tr_discrete_points,
    sample_tr_discrete_points_subset_d,
)
from torch import Tensor
from torch.quasirandom import SobolEngine
from logging import info, debug, warning

def convert_bounds_to_inequality_constraints(bounds):
    r"""Convert bounds into inequality constraints of the form Ax <= b.

    Args:
        bounds: A `2 x d`-dim tensor of bounds

    Returns:
        A two-element tuple containing
            - A: A `2d x d`-dim tensor of coefficients
            - b: A `2d x 1`-dim tensor containing the right hand side
    """
    d = bounds.shape[-1]
    eye = torch.eye(d, dtype=bounds.dtype, device=bounds.device)
    lower, upper = bounds
    lower_finite, upper_finite = bounds.isfinite()
    A = torch.cat([-eye[lower_finite], eye[upper_finite]], dim=0)
    b = torch.cat([-lower[lower_finite], upper[upper_finite]], dim=0).unsqueeze(-1)
    return A, b

def from_unit_cube(x, lower_bounds, upper_bounds):
    """
    Project points that were scaled to unit cube back to full space.

    Args:
        x: the points
        lower_bounds: the lower bounds of the full space
        upper_bounds: the upper bounds of the full space

    Returns: scaled points

    """
    
    xx = x * (upper_bounds - lower_bounds) + lower_bounds
    return xx


def from_1_around_origin(x, lower_bounds, upper_bounds):
    """
    Project points that were scaled to one-around-origin cube back to full space.

    Args:
        x: the points
        lower_bounds: the lower bounds of the full space
        upper_bounds: the upper bounds of the full space

    Returns: scaled points

    """
    xx = (x + 1) / 2
    return from_unit_cube(xx, lower_bounds, upper_bounds)

def to_unit_cube(x, lower_bounds, upper_bounds):
    """
    Project to [0, 1]^d from hypercube with bounds lb and ub

    Args:
        x: the points to scale
        lower_bounds: the lower bounds in the unscaled space
        upper_bounds: the upper bounds un the unscaled space

    Returns: scaled points

    """
    
    xx = (x - lower_bounds) / (upper_bounds - lower_bounds)
    return xx

def to_1_around_origin(x, lower_bounds, upper_bounds):
    """
    Project to [-1, 1]^d from hypercube with bounds lb and ub

    Args:
        x: the points to scale
        lower_bounds: the lower bounds in the unscaled space
        upper_bounds: the upper bounds un the unscaled space

    Returns: the scaled points.

    """
    x = to_unit_cube(x, lower_bounds, upper_bounds)
    xx = x * 2 - 1
    return xx

class CandidateSelectionOutput(NamedTuple):
    X_cand: Tensor
    tr_indices: Tensor


def get_partitioning(
    trbo_state: TRBOState, ref_point: Tensor, Y: Tensor
) -> BoxDecomposition:
    """Helper method for constructing a box decomposition"""
    if trbo_state.tr_hparams.use_approximate_hv_computations:
        alpha = (
            trbo_state.tr_hparams.approximate_hv_alpha
            if trbo_state.tr_hparams.approximate_hv_alpha is not None
            else get_default_partitioning_alpha(trbo_state.num_objectives)
        )
        partitioning = NondominatedPartitioning(ref_point=ref_point, Y=Y, alpha=alpha)
    else:
        partitioning = FastNondominatedPartitioning(ref_point=ref_point, Y=Y)
    return partitioning


def _make_unstandardizer(Y_mean: Tensor, Y_std: Tensor) -> Callable[[Tensor], Tensor]:
    def unstandardizer(Y: Tensor) -> Tensor:
        return Y * Y_std + Y_mean

    return unstandardizer


def preds_and_feas(
    trbo_state: TRBOState, tr_idx: int, X: Tensor
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Compute model predictions and constraint violations."""
    tkwargs = {"device": trbo_state.bounds.device, "dtype": trbo_state.bounds.dtype}
    tr = trbo_state.trust_regions[tr_idx]
    objective = tr.objective
    model = trbo_state.models[tr_idx]
    preds, dists = model.get_predictions_and_distances(X)
    # apply objective
    f_obj = objective(preds).clone()

    if trbo_state.constraints is not None:
        constraint_value = torch.stack(
            [c(preds) for c in trbo_state.constraints], dim=-1
        )
        feas = (constraint_value <= 0.0).all(dim=-1)
        violation = torch.clamp(constraint_value, 0.0).sum(dim=-1)
    else:
        feas = torch.ones(len(f_obj), device=tkwargs["device"], dtype=torch.bool)
        violation = torch.zeros(len(f_obj), **tkwargs)
    return f_obj, feas, violation, dists


def unit_rescale(x: Tensor) -> Tensor:
    """Helper function for normalizing a 1D input to [0, 1]."""
    if not x.dim() == 1:
        raise RuntimeError(f"Expected a 1D input, got shape: {list(x.shape)}")
    if x.min() == x.max():
        return 0.5 * torch.ones(x.shape, dtype=x.dtype, device=x.device)
    return (x - x.min()) / (x.max() - x.min())


def TS_select_batch_MORBO(trbo_state: TRBOState) -> CandidateSelectionOutput:
    r"""Generate a batch using Thompson sampling as presented in the MORBO.

    Select points using Thompson sampling. When using hypervolume we do greedy selection
    across all trust regions. When using random scalarizations we select a trust region at
    random and then generate a candidate using that trust region as comparing scalarizations
    from different trust regions doesn't work well.

    If there is no feasible candidate we choose the candidate that minimizes the total constraint
    violation. If hypervolume improvement is used but no candidate has non-zero hypervolume improvement
    then we pick the candidate according to a random scalarization.
    """
    tkwargs = {"device": trbo_state.bounds.device, "dtype": trbo_state.bounds.dtype}
    dim = trbo_state.dim
    batch_size = trbo_state.tr_hparams.batch_size
    n_trs = len(trbo_state.trust_regions)
    X_next = torch.empty(0, dim, **tkwargs)
    use_rffs = trbo_state.tr_hparams.use_simple_rff

    # We currently just pick a random trust region to start with and then loop over
    # the trust regions consecutively.
    tr_indices_selected = torch.zeros(
        batch_size, device=tkwargs["device"], dtype=torch.long
    )
    time_sampling, time_hvi = 0, 0
    for i in range(batch_size):
        debug(f'Sampling batch {i}')
        if trbo_state.tr_hparams.hypervolume:  # Greedy selection
            tr_indices = torch.arange(n_trs, device=tkwargs["device"])
        else:  # Greedy selection doesn't work well for random scalarizations
            tr_indices = torch.randint(
                n_trs, (1,), dtype=torch.long, device=tkwargs["device"]
            )

        # There are three different selection rules depending on constraints etc.
        # (1) Minimize violaton (2) Breaking ties (3) HVI
        best_cand = None
        best_acqval = float("-inf")
        best_tr_idx = -1
        best_selection_rule = -1

        # Loop over all trust regions we are considering
        for tr_idx in tr_indices:
            debug(f'Sampling tr {tr_idx}')
            tr = trbo_state.trust_regions[tr_idx]
            if tr.tr_hparams.hypervolume:
                best_X = normalize(tr.best_X, bounds=tr.bounds)
                # only use pareto points inside TR for generating candidates
                if tr.tr_hparams.spherical:
                    indices = get_indices_in_hypersphere(
                        tr.X_center_normalized, best_X, R=tr.radius
                    )
                else:
                    indices = get_indices_in_hypercube(
                        tr.X_center_normalized, best_X, length=tr.length
                    )
                best_X = best_X[indices]
                
                if tr.tr_hparams.max_centers is not None:
                    max_centers = tr.tr_hparams.max_centers
                    if best_X.shape[0] > max_centers:
                        limit_indices = torch.randperm(best_X.shape[0])[:max_centers]
                        best_X = best_X[limit_indices]
            else:
                best_X = tr.X_center_normalized

            if tr.sampler is None:
                # Perturbation probability
                prob_perturb = min(20.0 / dim, 1.0) * decay_function(
                    n=max(trbo_state.tr_hparams.n_initial_points, trbo_state.n_evals),
                    n0=trbo_state.tr_hparams.n_initial_points,
                    n_max=max(trbo_state.max_evals, trbo_state.n_evals),
                    alpha=0.5,
                )

            # Pending points that are in this hypercube
            if tr.tr_hparams.spherical:
                inds_next_in_tr = get_indices_in_hypersphere(
                    X_center=tr.X_center_normalized, X=X_next, R=2*tr.radius # use 2R since perturbation centers may live on edge of TR
                )
            else:
                inds_next_in_tr = get_indices_in_hypercube(
                    X_center=tr.X_center_normalized, X=X_next, length=tr.length
                )

            if tr.sampler is None:
                # Sample from all Pareto optimal points in the TR
                X_cand = sample_tr_discrete_points_subset_d(
                    best_X=best_X,
                    normalized_tr_bounds=tr.get_bounds(),
                    n_discrete_points=trbo_state.tr_hparams.raw_samples,
                    length=tr.length,
                    qmc=trbo_state.tr_hparams.qmc,
                    trunc_normal_perturb=trbo_state.tr_hparams.trunc_normal_perturb,
                    prob_perturb=prob_perturb,
                )
            else:
                # CTS
                # for now assume bounds are the only constraints
                normalized_tr_bounds = torch.cat([torch.zeros(1,dim), torch.ones(1,dim)], dim=0).to(best_X) # use global bounds and rely on Radius to enforce locality
                A, b = convert_bounds_to_inequality_constraints(bounds=normalized_tr_bounds)
                X_cand = tr.sampler.forward_STuRBO(
                    best_X=best_X,
                    A=A,
                    b=b,
                    Rmax=tr.radius,
                    sigma=tr.length/0.8 * torch.tensor(trbo_state.tr_hparams.sigma_init).to(best_X),
                    n_discrete_points=trbo_state.tr_hparams.raw_samples,
                )

            # Unnormalize initial conditions to the original hypercube for prediction
            X_cand_unnormalized = unnormalize(X_cand, bounds=tr.bounds)

            objective = trbo_state.trust_regions[tr_idx].objective
            model = trbo_state.models[tr_idx]

            # TODO: Make num_rff_features a hyperparameter of TuRBO
            if use_rffs:
                models = [model] if not isinstance(model, ModelListGP) else model.models
                sample_model = get_gp_samples(
                    model=model,
                    num_outputs=len(models),
                    n_samples=1,
                    num_rff_features=1024,
                )

            # Get the pending points inside the TR and stack them to the candidates
            if len(inds_next_in_tr) > 0:
                X_next_unnormalized = unnormalize(
                    X_next[inds_next_in_tr], bounds=tr.bounds
                )  # Unnormalize pending points for prediction
                X_cand_unnormalized = torch.cat(
                    (X_cand_unnormalized, X_next_unnormalized)
                )

            start = time.time()
            # TODO: Remove the `max_eager_kernel_size` setting when
            # https://github.com/cornellius-gp/gpytorch/issues/1853 has been fixed.
            with torch.no_grad(), gpytorch.settings.fast_computations(
                log_prob=False,
                covar_root_decomposition=False,
                solves=False,
            ), gpytorch.settings.max_eager_kernel_size(float("inf")):
                if use_rffs:
                    Y_sample = (
                        sample_model(X_cand_unnormalized).to(**tkwargs).squeeze(0)
                    )
                else:
                    Y_sample = (
                        model.posterior(X_cand_unnormalized)
                        .sample(torch.Size([1]))
                        .squeeze(0)
                    )
            end = time.time()
            time_sampling += end - start

            # apply objective
            f_obj = objective(Y_sample).clone()

            if trbo_state.constraints is not None:
                constraint_value = torch.stack(
                    [c(Y_sample) for c in trbo_state.constraints], dim=-1
                )
                feas = (constraint_value <= 0.0).all(dim=-1)
                violation = torch.clamp(constraint_value, 0.0).sum(dim=-1)
            else:
                feas = torch.ones(
                    len(f_obj), device=tkwargs["device"], dtype=torch.bool
                )
                violation = torch.zeros(len(f_obj), **tkwargs)

            # Remove the pending points and make sure we don't pick them
            if len(inds_next_in_tr) > 0:
                f_obj_next_in_tr = f_obj[-len(inds_next_in_tr) :].clone()
                feas_in_tr = feas[-len(inds_next_in_tr) :].clone()
                # To make sure these are never picked we set the violation to something large
                feas[-len(inds_next_in_tr) :] = False
                violation[-len(inds_next_in_tr) :] = float("inf")

            start = time.time()
            if not any(feas):  # Ignore the objectives if all are infeasible
                selection_rule = 1
                value_score = -1 * violation
                warning(f"{i}) No feasible point, minimizing violation")
            else:
                value_score = float("-inf") * torch.ones(len(f_obj), **tkwargs)
                if trbo_state.tr_hparams.hypervolume:
                    ref_point = trbo_state.ref_point.clone()
                    # This indexes so we have to clone here
                    pareto_Y_better_than_ref = objective(
                        trbo_state.pareto_Y_better_than_ref
                    ).clone()

                    # Include pending points inside this TR when computing the HVI
                    if len(inds_next_in_tr) > 0:
                        f_obj_next_in_tr_better_than_ref = f_obj_next_in_tr[
                            feas_in_tr & (f_obj_next_in_tr > ref_point).all(dim=-1)
                        ]  # Feasible predicted values better than the reference point
                        if len(f_obj_next_in_tr_better_than_ref) > 0:
                            pareto_Y_better_than_ref = torch.cat(
                                (
                                    pareto_Y_better_than_ref,
                                    f_obj_next_in_tr_better_than_ref,
                                ),
                                dim=0,
                            )
                            pareto_Y_better_than_ref = pareto_Y_better_than_ref[
                                is_non_dominated(pareto_Y_better_than_ref)
                            ]

                    # Set points that are either infeasible or not better than the
                    # reference point to have value score zero. If there are no
                    # such points or if no candidate point ends up on the Pareto
                    # frontier we use a random scalarization to break ties.
                    better_than_ref = feas & (f_obj > ref_point).all(dim=-1)
                    if any(better_than_ref):
                        f_obj_better_than_ref = f_obj[better_than_ref]  # m x o
                        # compute box decomposition
                        partitioning = get_partitioning(
                            trbo_state=trbo_state,
                            ref_point=ref_point,
                            Y=pareto_Y_better_than_ref,
                        )
                        # create a deterministic model that returns TS samples that we have
                        # already drawn (with an added dim for q=1). This lets us
                        # batch-evaluate the HVI using samples from the joint posterior
                        # over the discrete set.
                        def get_batched_objective_samples(X):
                            # return a raw_samples x 1 x m-dim tensor of feasible objectives
                            return f_obj_better_than_ref.unsqueeze(1)

                        sampled_model = GenericDeterministicModel(
                            f=get_batched_objective_samples,
                            num_outputs=f_obj_better_than_ref.shape[-1],
                        )
                        acqf = qExpectedHypervolumeImprovement(
                            model=sampled_model,
                            ref_point=ref_point,
                            partitioning=partitioning,
                            # sampler=SobolQMCNormalSampler(
                            #     num_samples=1
                            # ),  # dummy sampler
                        )
                        with torch.no_grad():
                            # add a q-batch dimension to compute HVI for each
                            # discrete point alone
                            hvi = acqf(  # dummy input
                                X_cand_unnormalized[better_than_ref].unsqueeze(1)
                            ).to(device=tkwargs["device"])
                        pareto_mask = hvi > 0
                    if any(better_than_ref) and any(pareto_mask):
                        # Hypervolume improvement
                        selection_rule = 3
                        value_score[better_than_ref] = hvi
                    else:
                        selection_rule = 2
                        warning(f"{i}) Breaking ties using a random scalarization")
                        weights = sample_simplex(
                            d=trbo_state.num_objectives, n=1, **tkwargs
                        )
                        value_score[feas] = (f_obj[feas] @ weights.t()).squeeze(-1)
                else:  # Random scalarization
                    selection_rule = 2
                    value_score[feas] = f_obj[feas]
            end = time.time()
            time_hvi += end - start

            # Pick the best point
            ind_best = value_score.argmax()
            x_best = X_cand[ind_best, :].unsqueeze(0)

            if selection_rule > best_selection_rule or (
                selection_rule == best_selection_rule
                and value_score.max() > best_acqval
            ):
                best_selection_rule = selection_rule
                best_acqval = value_score.max()
                best_tr_idx = tr_idx
                best_cand = x_best.clone()

        # Save the best candidate
        tr_indices_selected[i] = best_tr_idx
        X_next = torch.cat((X_next, best_cand), dim=0)

    # Unnormalize from [0, 1] to original problem space
    # NOTE: tr.bounds is the same for all TRs, so we can use any of them
    X_next = unnormalize(X=X_next, bounds=tr.bounds)

    debug(f"Time spent on sampling: {time_sampling:.1f} seconds")
    debug(f"Time spent on HVI computations: {time_hvi:.1f} seconds")
    tr_counts = [(tr_indices_selected == i).sum().cpu().item() for i in range(n_trs)]
    debug(f"Number of points selected from each TR: {tr_counts}")
    return CandidateSelectionOutput(X_cand=X_next, tr_indices=tr_indices_selected)
