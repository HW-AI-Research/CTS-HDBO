###############################################################################
# Copyright (c) 2019 Uber Technologies, Inc.                                  #
#                                                                             #
# Licensed under the Uber Non-Commercial License (the "License");             #
# you may not use this file except in compliance with the License.            #
# You may obtain a copy of the License at the root directory of this project. #
#                                                                             #
# See the License for the specific language governing permissions and         #
# limitations under the License.                                              #
###############################################################################

import math
import sys
from copy import deepcopy

import gpytorch
import numpy as np
import torch
from torch.quasirandom import SobolEngine

from .gp import train_gp
from .utils import from_unit_cube, latin_hypercube, to_unit_cube
from cts.baxus.util.acquisition_functions import ExpectedImprovement
from cts.baxus.util.behaviors.gp_configuration import GPBehaviour
from cts.baxus.gp import train_bock_gp
from gpytorch import settings

import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib import rc
import matplotlib.font_manager as font_manager
import pandas as pd
import time
from logging import info, debug, warning

class Turbo1:
    """The TuRBO-1 algorithm.

    Parameters
    ----------
    f : function handle
    lb : Lower variable bounds, numpy.array, shape (d,).
    ub : Upper variable bounds, numpy.array, shape (d,).
    n_init : Number of initial points (2*dim is recommended), int.
    max_evals : Total evaluation budget, int.
    batch_size : Number of points in each batch, int.
    verbose : If you want to print information about the optimization progress, bool.
    use_ard : If you want to use ARD for the GP kernel.
    max_cholesky_size : Largest number of training points where we use Cholesky, int
    n_training_steps : Number of training steps for learning the GP hypers, int
    min_cuda : We use float64 on the CPU if we have this or fewer datapoints
    device : Device to use for GP fitting ("cpu" or "cuda")
    dtype : Dtype to use for GP fitting ("float32" or "float64")

    Example usage:
        turbo1 = Turbo1(f=f, lb=lb, ub=ub, n_init=n_init, max_evals=max_evals)
        turbo1.optimize()  # Run optimization
        X, fX = turbo1.X, turbo1.fX  # Evaluated points
    """

    def __init__(
        self,
        f,
        lb,
        ub,
        n_init,
        max_evals,
        batch_size=1,
        verbose=True,
        failtol=None,
        length_init=0.8,
        scale_TR_sides=True,
        RAASP=True, # random axis-aligned subspace perturbations
        BOCK=False,
        use_ard=True,
        max_cholesky_size=2000,
        n_training_steps=50,
        min_cuda=1024,
        device="cpu",
        dtype="float64",
        logdir=None,
    ):

        self.logdir = logdir
        self.log_active_subspace = False
        self.log_uncertainty = False
        if self.log_uncertainty:
            self.uncertainty_log = {
                'pert_mag': [],
                'ss_pert_mag': [],
                'uncertainty': []
            }
        # Very basic input checks
        assert lb.ndim == 1 and ub.ndim == 1
        assert len(lb) == len(ub)
        assert np.all(ub > lb)
        assert max_evals > 0 and isinstance(max_evals, int)
        assert n_init > 0 and isinstance(n_init, int)
        assert batch_size > 0 and isinstance(batch_size, int)
        assert isinstance(verbose, bool) and isinstance(use_ard, bool)
        assert max_cholesky_size >= 0 and isinstance(batch_size, int)
        assert n_training_steps >= 30 and isinstance(n_training_steps, int)
        assert max_evals > n_init and max_evals > batch_size
        # assert device == "cpu" or device == "cuda"
        assert dtype == "float32" or dtype == "float64"
        if device == "cuda":
            assert torch.cuda.is_available(), "can't use cuda if it's not available"

        # Save function information
        self.f = f
        self.dim = len(lb)
        self.lb = lb
        self.ub = ub

        # Settings
        self.n_init = n_init
        self.max_evals = max_evals
        self.batch_size = batch_size
        self.verbose = verbose
        self.use_ard = use_ard
        self.max_cholesky_size = max_cholesky_size
        self.n_training_steps = n_training_steps

        # Hyperparameters
        self.mean = np.zeros((0, 1))
        self.signal_var = np.zeros((0, 1))
        self.noise_var = np.zeros((0, 1))
        self.lengthscales = np.zeros((0, self.dim)) if self.use_ard else np.zeros((0, 1))
        self.RAASP = RAASP # random axis-aligned subspace perturbations
        self.BOCK = BOCK
        if self.BOCK:
            self.use_EI = True
        else:
            self.use_EI = False
        

        # Tolerances and counters
        self.n_cand = min(100 * self.dim, 5000)
        # self.n_cand = min(min(100 * self.dim, 450_000 // self.dim), 5000)

        self.scale_TR_sides = scale_TR_sides
        self.succtol = 3
        self.n_evals = 0

        # Trust region sizes
        self.length_min = 0.5 ** 7
        # self.length_max = 1.6
        self.length_init = length_init
        self.length_max = max(self.length_init, 1.6)

        if failtol is None:
            self.failtol = np.ceil(np.max([4.0 / batch_size, self.dim / batch_size]))
            # k = np.ceil(-np.log2(self.length_min / self.length_init))
            # self.failtol = np.round(0.5 * (self.max_evals - self.n_init) / k) 
            # self.failtol = np.round((self.max_evals - self.n_init) / k) 
        else:
            self.failtol = failtol

        # Save the full history
        self.X = np.zeros((0, self.dim))
        self.fX = np.zeros((0, 1))

        # Device and dtype for GPyTorch
        self.min_cuda = min_cuda
        self.dtype = torch.float32 if dtype == "float32" else torch.float64
        self.device = device
        if self.verbose:
            info("Using dtype = %s, Using device = %s" % (self.dtype, self.device))
            sys.stdout.flush()

        # Initialize parameters
        self._restart()

    def _restart(self):
        self._X = []
        self._fX = []
        self.failcount = 0
        self.succcount = 0
        self.length = self.length_init

    def _adjust_length(self, fX_next):
        if np.min(fX_next) < np.min(self._fX) - 1e-3 * math.fabs(np.min(self._fX)):
            self.succcount += 1
            self.failcount = 0
        else:
            self.succcount = 0
            self.failcount += 1

        if self.succcount == self.succtol:  # Expand trust region
            self.length = min([2.0 * self.length, self.length_max])
            self.succcount = 0
        elif self.failcount == self.failtol:  # Shrink trust region
            self.length /= 2.0
            self.failcount = 0

    def _create_candidates(self, X, fX, length, n_training_steps, hypers):
        """Generate candidates assuming X has been scaled to [0,1]^d."""
        # Pick the center as the point with the smallest function values
        # NOTE: This may not be robust to noise, in which case the posterior mean of the GP can be used instead
        assert X.min() >= 0.0 and X.max() <= 1.0
        if self.use_EI:
            fX = fX.copy() * (-1)
        # Standardize function values.
        mu, sigma = np.median(fX), fX.std()
        sigma = 1.0 if sigma < 1e-6 else sigma

        fX = (deepcopy(fX) - mu) / sigma

        # Figure out what device we are running on
        if len(X) < self.min_cuda:
            device, dtype = torch.device("cpu"), torch.float64
        else:
            device, dtype = self.device, self.dtype

        # We use CG + Lanczos for training if we have enough data
        with gpytorch.settings.max_cholesky_size(self.max_cholesky_size):
            X_torch = torch.tensor(X).to(device=device, dtype=dtype)
            y_torch = torch.tensor(fX).to(device=device, dtype=dtype)
            if self.BOCK:
                # if self.use_EI:
                #     # EI expects maximization
                #     y_torch *= -1
                with settings.cholesky_max_tries(30):
                    gp, _ = train_bock_gp(
                        train_x=X_torch,
                        train_y=y_torch,
                        use_ard=self.use_ard,
                        gp_behaviour=GPBehaviour(),
                        hypers=hypers,
                    )                
            else:
                gp = train_gp(
                    train_x=X_torch, train_y=y_torch, use_ard=self.use_ard, num_steps=n_training_steps, hypers=hypers
                )

            # Save state dict
            hypers = gp.state_dict()

        # Create the trust region boundaries
        x_center = X[fX.argmin().item(), :][None, :]
        if self.scale_TR_sides:
            weights = gp.covar_module.base_kernel.lengthscale.cpu().detach().numpy().ravel()
            weights = weights / weights.mean()  # This will make the next line more stable
            weights = weights / np.prod(np.power(weights, 1.0 / len(weights)))  # We now have weights.prod() = 1
        else:
            weights = 1.0
        lb = np.clip(x_center - weights * length / 2.0, 0.0, 1.0)
        ub = np.clip(x_center + weights * length / 2.0, 0.0, 1.0)

        # Draw a Sobolev sequence in [lb, ub]
        seed = np.random.randint(int(1e6))
        sobol = SobolEngine(self.dim, scramble=True, seed=seed)
        pert = sobol.draw(self.n_cand).to(dtype=dtype, device=device).cpu().detach().numpy()
        pert = lb + (ub - lb) * pert

        # Create a perturbation mask
        if self.RAASP: # random axis-aligned subspace perturbations
            prob_perturb = min(20.0 / self.dim, 1.0)
        else:
            prob_perturb = 1.0
        mask = np.random.rand(self.n_cand, self.dim) <= prob_perturb
        ind = np.where(np.sum(mask, axis=1) == 0)[0]
        mask[ind, np.random.randint(0, self.dim - 1, size=len(ind))] = 1

        # Create candidate points
        X_cand = x_center.copy() * np.ones((self.n_cand, self.dim))
        X_cand[mask] = pert[mask]

        # Figure out what device we are running on
        if len(X_cand) < self.min_cuda:
            device, dtype = torch.device("cpu"), torch.float64
        else:
            device, dtype = self.device, self.dtype

        # We may have to move the GP to a new device
        gp = gp.to(dtype=dtype, device=device)

        if self.use_EI:
            # this EI assumes maximization
            EI = ExpectedImprovement(gp, best_f=fX.max(), lb=lb, ub=ub)
            start = time.time()
            with settings.cholesky_max_tries(30):
                X_cand, y_cand = EI.optimize()
            end = time.time()
            info(
                f"Optimizing EI took {end - start:.2f}s with {len(self._X)} datapoints.")
            del EI
            # y_cand = torch.unsqueeze(y_cand, 1)

            # De-standardize the sampled values and negate
            y_cand = (mu + sigma * y_cand) * (-1)
            
            y_cand = np.expand_dims([y_cand],0)
            
        else:
            # We use Lanczos for sampling if we have enough data
            with torch.no_grad(), gpytorch.settings.max_cholesky_size(self.max_cholesky_size):
                X_cand_torch = torch.tensor(X_cand).to(device=device, dtype=dtype)
                y_cand = gp.likelihood(gp(X_cand_torch)).sample(torch.Size([self.batch_size])).t().cpu().detach().numpy()

            # Remove the torch variables
            if self.log_uncertainty:
                del X_torch, y_torch, X_cand_torch
            else:
                del X_torch, y_torch, X_cand_torch, gp
            # De-standardize the sampled values
            y_cand = mu + sigma * y_cand

        if self.log_uncertainty:
            return X_cand, y_cand, hypers, gp
        else:
            return X_cand, y_cand, hypers

    def _select_candidates(self, X_cand, y_cand):
        """Select candidates."""
        X_next = np.ones((self.batch_size, self.dim))
        for i in range(self.batch_size):
            # Pick the best point and make sure we never pick it again
            indbest = np.argmin(y_cand[:, i])
            X_next[i, :] = deepcopy(X_cand[indbest, :])
            y_cand[indbest, :] = np.inf
        return X_next

    def log_active_subspace_pert(self, X_next):
        incumbent = to_unit_cube(deepcopy(self.X[self.fX.argmin():self.fX.argmin()+1]), self.lb, self.ub)
        candidate = deepcopy(X_next)
        perturbation = incumbent - candidate
        active_subspace_perturbation = perturbation[0, :2]
        info(f'{active_subspace_perturbation=}')
        info(f'active subspace perturbation size: {np.linalg.norm(active_subspace_perturbation)}')
        if self.n_evals % 20 == 0:
        
            rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
            rc('text', usetex=True)
            font_scale = 1.4
            fontdict = {"fontsize": 20*font_scale}
            font = font_manager.FontProperties(size=18*font_scale)
            
            marker_scale = 2.5
            
            unwarped_incumbent = self.X[self.fX.argmin()]
            fig, axes = plt.subplots(1, 1, figsize=(9, 7))
            samples = axes.scatter(self.X[:,0], self.X[:,1], c=np.arange(len(self.X)), cmap='Greys', edgecolors='black', s=50*marker_scale, label='Samples')
            axes.scatter(unwarped_incumbent[0], unwarped_incumbent[1], color='pink', marker='s', s=80*marker_scale, edgecolors='black', label='Incumbent')
            # plot optima
            z1 = (-np.pi, 12.275)
            z2 = (np.pi, 2.275)
            z3 = (3*np.pi, 2.475)
            zs = [z2, z3]
            axes.scatter(z1[0], z1[1], marker='x', color='r', s=80*marker_scale, label='Optima') # just label one optima
            for z in zs:
                axes.scatter(z[0], z[1], marker='x', color='r', s=80*marker_scale)
            
            cb = fig.colorbar(samples)
            cb.set_label(label='Sample Number', **fontdict)
            cb.ax.tick_params(labelsize=16*font_scale)
            axes.tick_params(width=1.5, axis='both', which='major', labelsize=16*font_scale)
            axes.tick_params(width=1.5, axis='both', which='minor', labelsize=16*font_scale)
            plt.xlim([-5, 15])
            plt.ylim([-5, 15])
            axes.set_xticks([-5, 0, 5, 10, 15])
            axes.set_yticks([-5, 0, 5, 10, 15])
            plt.xlabel(r"$x_1$", **fontdict)
            plt.ylabel(r"$x_2$", **fontdict)
            plt.legend(prop=font, framealpha=1.0)
            plt.tight_layout()
            plot_dir = Path(self.logdir) / 'plots'
            plot_dir.mkdir(exist_ok=True)
            plt.savefig(plot_dir / f'active_ss_projection_{self.n_evals}_evals.png')
            
        # time.sleep(1)

    def log_sample_uncertainty(self, X_next, gp):
        
        incumbent = to_unit_cube(deepcopy(self.X[self.fX.argmin():self.fX.argmin()+1]), self.lb, self.ub)
        candidate = deepcopy(X_next)
        
        perturbation = incumbent - candidate
        active_subspace_perturbation = perturbation[0, :2]
        
        pert_mag = np.linalg.norm(perturbation)
        Active_SS_pert_mag = np.linalg.norm(active_subspace_perturbation)
        
        candidate_torch = torch.tensor(candidate).to(device=self.device, dtype=self.dtype)
            
        posterior = gp.likelihood(gp(candidate_torch))
        uncertainty = posterior.variance.detach().cpu().numpy().item()
        
        info(f'{pert_mag=}')
        info(f'{Active_SS_pert_mag=}')
        info(f'{uncertainty=}')
        # time.sleep(1)
        self.uncertainty_log['pert_mag'].append(pert_mag)
        self.uncertainty_log['ss_pert_mag'].append(Active_SS_pert_mag)
        self.uncertainty_log['uncertainty'].append(uncertainty)

    def optimize(self):
        """Run the full optimization process."""
        while self.n_evals < self.max_evals:
            if len(self._fX) > 0 and self.verbose:
                n_evals, fbest = self.n_evals, self._fX.min()
                info(f"{n_evals}) Restarting with fbest = {fbest:.4}")
                sys.stdout.flush()

            # Initialize parameters
            self._restart()

            # Generate and evalute initial design points
            X_init = latin_hypercube(self.n_init, self.dim)
            X_init = from_unit_cube(X_init, self.lb, self.ub)
            t0 = time.time()
            fX_init = np.array([[self.f(x)] for x in X_init])
            t1 = time.time()
            elapsed = t1-t0
            debug(f'time for {X_init.shape[0]} init: {elapsed} s')
            debug(f'{elapsed/X_init.shape[0]} seconds per eval')

            # Update budget and set as initial data for this TR
            self.n_evals += self.n_init
            self._X = deepcopy(X_init)
            self._fX = deepcopy(fX_init)

            # Append data to the global history
            self.X = np.vstack((self.X, deepcopy(X_init)))
            self.fX = np.vstack((self.fX, deepcopy(fX_init)))

            if self.verbose:
                fbest = self._fX.min()
                info(f"Starting from fbest = {fbest:.4}")
                sys.stdout.flush()

            # Thompson sample to get next suggestions
            while self.n_evals < self.max_evals and self.length >= self.length_min:
                # Warp inputs
                X = to_unit_cube(deepcopy(self._X), self.lb, self.ub)

                # Standardize values
                fX = deepcopy(self._fX).ravel()

                # Create th next batch
                if self.log_uncertainty:
                    X_cand, y_cand, _, gp = self._create_candidates(
                        X, fX, length=self.length, n_training_steps=self.n_training_steps, hypers={}
                    )
                else:
                    X_cand, y_cand, _ = self._create_candidates(
                        X, fX, length=self.length, n_training_steps=self.n_training_steps, hypers={}
                    )
                X_next = self._select_candidates(X_cand, y_cand)

                if self.log_active_subspace:
                    # log active subspace perturbation
                    self.log_active_subspace_pert(X_next)
                
                if self.log_uncertainty:
                    self.log_sample_uncertainty(X_next, gp)
                
                # Undo the warping
                X_next = from_unit_cube(X_next, self.lb, self.ub)

                # Evaluate batch
                fX_next = np.array([[self.f(x)] for x in X_next])

                # Update trust region
                self._adjust_length(fX_next)

                # Update budget and append data
                self.n_evals += self.batch_size
                self._X = np.vstack((self._X, X_next))
                self._fX = np.vstack((self._fX, fX_next))

                if self.verbose and fX_next.min() < self.fX.min():
                    n_evals, fbest = self.n_evals, fX_next.min()
                    info(f"{n_evals}) New best: {fbest:.4}")
                    sys.stdout.flush()

                # Append data to the global history
                self.X = np.vstack((self.X, deepcopy(X_next)))
                self.fX = np.vstack((self.fX, deepcopy(fX_next)))

        if self.log_uncertainty:
            df = pd.DataFrame.from_dict(self.uncertainty_log)
            csv_path = Path(self.logdir) / 'uncertainty.csv'
            df.to_csv(csv_path)