# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
from typing import Optional, Union
from torch import Tensor
from torch.distributions import Independent, Normal, OneHotCategoricalStraightThrough, Categorical
from torch.nn import functional as F


class MyDist:
    def __init__(self, *args, **kwargs) -> None:
        self.distribution = None

    def log_prob(self, sample: Tensor) -> Tensor:
        pass

    def sample(self, deterministic: Union[bool, Tensor]) -> Tensor:
        pass

    def repeat_interleave_(self, repeats: int, dim: int) -> None:
        pass


class DiagGaussian(MyDist):
    def __init__(self, mean: Tensor, log_std: Tensor, valid: Optional[Tensor] = None) -> None:
        """
        mean: [n_sc, n_ag, (k_pred), out_dim]
        """
        super().__init__()
        self.mean = mean
        self.valid = valid
        self.distribution = Independent(Normal(self.mean, log_std.exp()), 1)
        self.stddev = self.distribution.stddev

    def log_prob(self, sample: Tensor) -> Tensor:
        """
        log_prob: [n_sc, n_ag]
        """
        return self.distribution.log_prob(sample)

    def sample(self, deterministic: Union[bool, Tensor]) -> Tensor:
        """
        Args:
            deterministic: bool, or Tensor for sampling relevant agents and determistic other agents.
        Returns:
            sample: [n_sc, n_ag, out_dim]
        """
        if type(deterministic) is Tensor:
            det_sample = self.distribution.mean
            rnd_sample = self.distribution.rsample()
            sample = det_sample.masked_fill(~deterministic.unsqueeze(-1), 0) + rnd_sample.masked_fill(
                deterministic.unsqueeze(-1), 0
            )
        else:
            if deterministic:
                sample = self.distribution.mean
            else:
                sample = self.distribution.rsample()
        return sample

    def repeat_interleave_(self, repeats: int, dim: int) -> None:
        self.mean = self.mean.repeat_interleave(repeats, dim)
        self.stddev = self.stddev.repeat_interleave(repeats, dim)
        self.distribution = Independent(Normal(self.mean, self.stddev), 1)
        if self.valid is not None:
            self.valid = self.valid.repeat_interleave(repeats, dim)


class MultiCategorical(MyDist):
    def __init__(self, logits: Tensor, valid: Optional[Tensor] = None):
        """
        logits: [n_sc, n_ag, n_cat, n_class]
        """
        super().__init__()
        self.logits = logits
        self.distribution = Independent(OneHotCategoricalStraightThrough(logits=self.logits), 1)
        self.n_cat = self.logits.shape[-2]
        self.n_class = self.logits.shape[-1]
        self._dtype = self.logits.dtype
        self.valid = valid

    def log_prob(self, sample: Tensor) -> Tensor:
        # [n_sc, n_ag]
        return self.distribution.log_prob(sample.view(*sample.shape[:-1], self.n_cat, self.n_class))

    def sample(self, deterministic: Union[bool, Tensor]) -> Tensor:
        """
        Args:
            deterministic: bool, or Tensor for sampling relevant agents and determistic other agents.
        Returns:
            sample: [n_sc, n_ag, out_dim]
        """
        # [n_sc, n_ag, n_cat, n_class]
        if type(deterministic) is Tensor:
            det_sample = (
                F.one_hot(self.distribution.base_dist.probs.argmax(-1), num_classes=self.n_class)
                .type(self._dtype)
                .flatten(start_dim=-2, end_dim=-1)
            )
            rnd_sample = self.distribution.rsample().flatten(start_dim=-2, end_dim=-1)
            sample = det_sample.masked_fill(~deterministic.unsqueeze(-1), 0) + rnd_sample.masked_fill(
                deterministic.unsqueeze(-1), 0
            )
        else:
            if deterministic:
                sample = (
                    F.one_hot(self.distribution.base_dist.probs.argmax(-1), num_classes=self.n_class)
                    .type(self._dtype)
                    .flatten(start_dim=-2, end_dim=-1)
                )
            else:
                sample = self.distribution.rsample().flatten(start_dim=-2, end_dim=-1)
        return sample

    def repeat_interleave_(self, repeats: int, dim: int) -> None:
        self.logits = self.logits.repeat_interleave(repeats, dim)
        self.distribution = Independent(OneHotCategoricalStraightThrough(logits=self.logits), 1)
        self.n_cat = self.logits.shape[-2]
        self.n_class = self.logits.shape[-1]
        self._dtype = self.logits.dtype
        if self.valid is not None:
            self.valid = self.valid.repeat_interleave(repeats, dim)


class DestCategorical(MyDist):
    def __init__(self, probs: Optional[Tensor] = None, logits: Optional[Tensor] = None, valid: Optional[Tensor] = None):
        """
        probs: [n_sc, n_ag, n_mp] >= 0, sum up to 1.
        """
        super().__init__()
        if probs is None:
            assert logits is not None
            self.distribution = Categorical(logits=logits)
            self.probs = self.distribution.probs
        else:
            assert probs is not None
            self.distribution = Categorical(probs=probs)
            self.probs = self.distribution.probs

        self.valid = valid

    def log_prob(self, sample: Tensor) -> Tensor:
        return self.distribution.log_prob(sample)  # [n_sc, n_ag]

    def sample(self, deterministic: Union[bool, Tensor]) -> Tensor:
        """
        Args:
            deterministic: bool, or Tensor for sampling relevant agents and determistic other agents.
        Returns:
            sample: [n_sc, n_ag]
        """
        if type(deterministic) is Tensor:
            det_sample = self.distribution.probs.argmax(-1)
            rnd_sample = self.distribution.sample()
            sample = det_sample.masked_fill(~deterministic, 0) + rnd_sample.masked_fill(deterministic, 0)
        else:
            if deterministic:
                sample = self.distribution.probs.argmax(-1)
            else:
                sample = self.distribution.sample()
        return sample

    def repeat_interleave_(self, repeats: int, dim: int) -> None:
        self.probs = self.probs.repeat_interleave(repeats, dim)
        self.distribution = Categorical(probs=self.probs)
        if self.valid is not None:
            self.valid = self.valid.repeat_interleave(repeats, dim)
