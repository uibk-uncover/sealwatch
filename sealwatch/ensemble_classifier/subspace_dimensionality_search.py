import numpy as np


class SubspaceDimensionalitySearch(object):
    def __init__(self, d_sub_step, max_dim, oob_error_tolerance=0.02):
        """
        Search for optimal subspace dimensionality `d_sub`.
        Seek the minimum detector errors through a one-dimensional search over `d_sub`. Approach the minimum "from the left", because the FLD base learner's complexity increases for higher `d_sub`.

        :param d_sub_step: search step size; denoted as delta_d in the paper
        :param max_dim: maximum number of dimensions
        :param oob_error_tolerance: tolerance for early stopping; denoted as tau in the paper
        """
        # Initialize step size
        self.d_sub_step = d_sub_step
        if d_sub_step >= max_dim / 4:
            # Curb overly large step size
            # The maximum step size should be a quarter of the number of features
            self.d_sub_step = max_dim // 4
        if max_dim < 10:
            # For small feature spaces, use a step size of 1
            self.d_sub_step = 1

        # Copy remaining arguments
        self.max_dim = max_dim
        self.oob_error_tolerance = oob_error_tolerance

        # Candidates for d_sub
        # Should always be sorted
        self.d_sub_candidates = np.array([1, 2, 3], dtype=int) * self.d_sub_step

        # Special handling in case there are only two dimensions
        if self.max_dim == 2:
            self.d_sub_candidates = np.array([1, 2], dtype=int)

        # OOB error corresponding to d_sub
        # Mark unexplored d_sub candidates with an OOB error of -1
        self.oob_errors = -1 * np.ones(len(self.d_sub_candidates))

        # Number of search iterations
        self.search_counter = 0

        # Robustness to statistical variations; controls when to move from stage 1 to stage 2
        self.epsilon = 5e-3

        # Keep track of best d_sub candidate
        self.optimal_d_sub = None
        self.min_oob_error = None

        self.search_in_progress = True
        self.next_d_sub = int(self.d_sub_candidates[0])

    def _add_gridpoints(self, new_d_sub_candidates):
        if np.isscalar(new_d_sub_candidates):
            raise ValueError("Expected ndarray of new gridpoints")

        # Find indices where to insert new candidate points
        ii = np.searchsorted(self.d_sub_candidates, new_d_sub_candidates)

        # Insert d_sub candidates
        self.d_sub_candidates = np.insert(self.d_sub_candidates, ii, new_d_sub_candidates)

        # Insert empty places for corresponding OOB error
        new_candidates_oob_errors = -1 * np.ones(len(new_d_sub_candidates))
        self.oob_errors = np.insert(self.oob_errors, ii, new_candidates_oob_errors)

    def update(self, d_sub, current_error):
        """
        This method should be called after training on a specific subspace has been completed.

        If training should terminate, self.search_in_progress is set to False.
        Otherwise, self.next_d_sub is set to the next candidate.

        See Algorithm 2 in the paper for a reference, although the code diverges slightly.

        :param d_sub: subspace dimensionality that was currently evaluated
        :param current_error: corresponding out-of-bag error estimate
        """
        self.search_counter += 1
        if not self.search_in_progress:
            return

        # Find position
        d_sub_idx = np.where(self.d_sub_candidates == d_sub)[0]
        assert len(d_sub_idx) == 1, "Given `d_sub` must be exactly one of the candidates"
        d_sub_idx = d_sub_idx[0]

        # Save error
        self.oob_errors[d_sub_idx] = current_error

        # Check whether we have any other candidates left to explore
        unfinished = np.where(self.oob_errors == -1)[0]
        if len(unfinished) > 0:
            # Set d_sub to the next value that we haven't explored so far
            self.next_d_sub = int(self.d_sub_candidates[unfinished[0]])
            return

        # No unfinished values
        # Find the current minimum
        min_idx = np.argmin(self.oob_errors)
        min_oob_error = self.oob_errors[min_idx]

        # Have we reached zero error or the smallest possible step size?
        if self.d_sub_step == 1 or np.isclose(min_oob_error, 0):
            # Terminate search because we have reached zero error or the smallest possible step size
            self.search_in_progress = False
            self.optimal_d_sub = self.d_sub_candidates[min_idx]
            self.min_oob_error = self.oob_errors[min_idx]
            return

        if min_idx == 0:
            # Smallest candidate is the best
            # Reduce step size
            self.d_sub_step = self.d_sub_step // 2

            # Explore two gridpoints around the previous minimum d_sub
            new_d_sub_candidates = self.d_sub_candidates[0] + self.d_sub_step * np.array([-1, 1])
            self._add_gridpoints(new_d_sub_candidates)

        elif min_idx == len(self.d_sub_candidates) - 1:
            # Largest candidate is the best

            if (self.d_sub_candidates[-1] + self.d_sub_step <= self.max_dim
                    and np.min(np.abs(self.d_sub_candidates[-1] + self.d_sub_step - self.d_sub_candidates)) > self.d_sub_step / 2):
                # Condition 0: We can still take a step without exceeding the maximum number of dimensions
                # Condition 1: One more step to the right is not too close to a previous step

                # Continue to the right
                self._add_gridpoints(self.d_sub_candidates[-1:] + self.d_sub_step)

            else:
                # Hitting the full dimensionality
                if (min_oob_error / self.oob_errors[-2] >= 1 - self.oob_error_tolerance  # Desired tolerance fulfilled
                        or self.oob_errors[-2] - min_oob_error < self.epsilon  # maximal precision in terms of error set to 0.5%
                        or self.d_sub_step < self.d_sub_candidates[min_idx] * 0.05):  # step is smaller than 5% of the optimal value of k

                    # Stopping criterion is met
                    self.search_in_progress = False
                    self.optimal_d_sub = self.d_sub_candidates[min_idx]
                    self.min_oob_error = self.oob_errors[min_idx]
                    return

                else:
                    # Reduce step
                    self.d_sub_step = self.d_sub_step // 2
                    if self.d_sub_candidates[-1] + self.d_sub_step <= self.max_dim:
                        # If we can still increase the dimensionality, explore both higher and lower candidates
                        self._add_gridpoints(self.d_sub_candidates[-1] + self.d_sub_step * np.array([-1, 1]))
                    else:
                        # Only explore a lower candidate
                        self._add_gridpoints(self.d_sub_candidates[-1:] - self.d_sub_step)

        elif (min_idx == len(self.d_sub_candidates) - 2
              and self.d_sub_candidates[min_idx] + self.d_sub_step <= self.max_dim
              and np.min(np.abs(self.d_sub_candidates[min_idx] + self.d_sub_step - self.d_sub_candidates)) > self.d_sub_step / 2
              and not (self.oob_errors[-1] > self.oob_errors[-2] and self.oob_errors[-1] > self.oob_errors[-3])
        ):
            # Condition 1: If lowest is the second to last
            # Condition 2: One more step to the right is still possible (less than or equal to max_dim)
            # Condition 3: One more step to the right is not too close to any other candidate
            # Condition 4: The last point is not worse than the two previous ones

            # Robustness ensurance, try one more step to the right
            self._add_gridpoints(np.array([d_sub + self.d_sub_step]))

        else:
            # Best candidate is not at the edge of the grid (and robustness is resolved)
            oob_error_around = (self.oob_errors[min_idx - 1] + self.oob_errors[min_idx + 1]) / 2

            if (min_oob_error / oob_error_around >= 1 - self.oob_error_tolerance
                    or oob_error_around - min_oob_error < self.epsilon
                    or self.d_sub_step < self.d_sub_candidates[min_idx] * 0.05):
                # Condition 1: Desired tolerance fulfilled
                # Condition 2: Maximal precision in terms of error set to 0.5%
                # Condition 3: Step is smaller than 5% of the optimal value of d_sub

                # Stopping criterion met
                self.search_in_progress = False
                self.optimal_d_sub = self.d_sub_candidates[min_idx]
                self.min_oob_error = self.oob_errors[min_idx]
                return

            else:
                # Reduce step
                self.d_sub_step = self.d_sub_step // 2
                self._add_gridpoints(self.d_sub_candidates[min_idx] + self.d_sub_step * np.array([-1, 1]))

        # Find the next unfinished candidate
        unfinished = np.where(self.oob_errors == -1)[0]
        self.next_d_sub = int(self.d_sub_candidates[unfinished[0]])


class FixedDimensionalityDummySearch(object):
    def __init__(self, d_sub):
        """
        Dummy search object for a fixed subspace dimensionality
        :param d_sub: subspace dimensionality
        """
        self.next_d_sub = d_sub
        self.optimal_oob_error = None
        self.search_in_progress = True

    def update(self, d_sub, current_error):
        assert d_sub == self.next_d_sub
        self.optimal_oob_error = current_error
        self.search_in_progress = False
