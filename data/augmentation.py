import numpy as np
from numpy.typing import NDArray
from typing import Optional, Tuple
from scipy.interpolate import CubicSpline


class TimeSeriesAugmenter:
    """Time series data augmentation for training."""

    def __init__(
        self,
        jitter_std: float = 0.03,
        scaling_range: Tuple[float, float] = (0.8, 1.2),
        time_warp_strength: float = 0.2,
        window_slice_ratio: float = 0.9,
        rotation_prob: float = 0.5,
        augment_prob: float = 0.5
    ) -> None:
        """Initialize the TimeSeriesAugmenter.

        Args:
            jitter_std: Standard deviation for Gaussian noise.
            scaling_range: Range for magnitude scaling (min, max).
            time_warp_strength: Strength of time warping distortion.
            window_slice_ratio: Ratio of original length to keep when slicing.
            rotation_prob: Probability of applying rotation (multivariate only).
            augment_prob: Probability of applying augmentation to each sample.
        """
        self.jitter_std = jitter_std
        self.scaling_range = scaling_range
        self.time_warp_strength = time_warp_strength
        self.window_slice_ratio = window_slice_ratio
        self.rotation_prob = rotation_prob
        self.augment_prob = augment_prob

    def jitter(self, x: NDArray[np.float32]) -> NDArray[np.float32]:
        """Add Gaussian noise to the time series.

        Args:
            x: Input time series of shape (..., length).

        Returns:
            Augmented time series with added noise.
        """
        noise = np.random.normal(0, self.jitter_std, x.shape)
        return x + noise.astype(np.float32)

    def scaling(self, x: NDArray[np.float32]) -> NDArray[np.float32]:
        """Apply random magnitude scaling to the time series.

        Args:
            x: Input time series of shape (..., length).

        Returns:
            Scaled time series.
        """
        scale_factor = np.random.uniform(self.scaling_range[0], self.scaling_range[1])
        return (x * scale_factor).astype(np.float32)

    def time_warp(self, x: NDArray[np.float32]) -> NDArray[np.float32]:
        """Apply smooth time distortion using cubic splines.

        Args:
            x: Input time series of shape (..., length).

        Returns:
            Time-warped time series.
        """
        orig_shape = x.shape
        length = orig_shape[-1]
        
        # Flatten to 2D for processing
        if x.ndim == 1:
            x = x.reshape(1, -1)
        elif x.ndim > 2:
            x = x.reshape(-1, length)
        
        # Create warped time indices
        num_knots = max(4, length // 10)
        knot_indices = np.linspace(0, length - 1, num_knots)
        knot_values = knot_indices + np.random.uniform(
            -self.time_warp_strength * length / num_knots,
            self.time_warp_strength * length / num_knots,
            num_knots
        )
        knot_values = np.clip(knot_values, 0, length - 1)
        
        # Interpolate to get smooth warping
        cs = CubicSpline(knot_indices, knot_values)
        warped_indices = cs(np.arange(length))
        warped_indices = np.clip(warped_indices, 0, length - 1)
        
        # Apply warping to each channel
        warped = np.zeros_like(x)
        for i in range(x.shape[0]):
            warped[i] = np.interp(np.arange(length), warped_indices, x[i])
        
        return warped.reshape(orig_shape).astype(np.float32)

    def window_slice(self, x: NDArray[np.float32]) -> NDArray[np.float32]:
        """Random cropping with padding to maintain original length.

        Args:
            x: Input time series of shape (..., length).

        Returns:
            Sliced and padded time series.
        """
        orig_shape = x.shape
        length = orig_shape[-1]
        
        # Calculate slice length
        slice_length = int(length * self.window_slice_ratio)
        if slice_length >= length:
            return x
        
        # Random start position
        start = np.random.randint(0, length - slice_length + 1)
        end = start + slice_length
        
        # Extract slice
        if x.ndim == 1:
            sliced = x[start:end]
        elif x.ndim == 2:
            sliced = x[:, start:end]
        else:
            sliced = x[..., start:end]
        
        # Pad to original length
        pad_before = (length - slice_length) // 2
        pad_after = length - slice_length - pad_before
        
        if x.ndim == 1:
            padded = np.pad(sliced, (pad_before, pad_after), mode='edge')
        elif x.ndim == 2:
            padded = np.pad(sliced, ((0, 0), (pad_before, pad_after)), mode='edge')
        else:
            pad_width = [(0, 0)] * (x.ndim - 1) + [(pad_before, pad_after)]
            padded = np.pad(sliced, pad_width, mode='edge')
        
        return padded.astype(np.float32)

    def rotation(self, x: NDArray[np.float32]) -> NDArray[np.float32]:
        """Apply rotation (channel mixing) for multivariate time series.

        Args:
            x: Input time series of shape (channels, length).

        Returns:
            Rotated time series.
        """
        if x.ndim < 2:
            return x  # Cannot rotate univariate series
        
        if np.random.random() > self.rotation_prob:
            return x
        
        # Apply random rotation to channels
        num_channels = x.shape[0] if x.ndim == 2 else x.shape[-2]
        
        if num_channels == 1:
            return x
        
        # Generate random rotation matrix
        rotation_matrix = self._random_rotation_matrix(num_channels)
        
        if x.ndim == 2:
            # Shape: (channels, length)
            return (rotation_matrix @ x).astype(np.float32)
        else:
            # Handle higher dimensions
            orig_shape = x.shape
            x_reshaped = x.reshape(-1, num_channels, orig_shape[-1])
            rotated = np.zeros_like(x_reshaped)
            for i in range(x_reshaped.shape[0]):
                rotated[i] = rotation_matrix @ x_reshaped[i]
            return rotated.reshape(orig_shape).astype(np.float32)

    def _random_rotation_matrix(self, n: int) -> NDArray[np.float32]:
        """Generate a random rotation matrix.

        Args:
            n: Dimension of the rotation matrix.

        Returns:
            Random orthogonal rotation matrix.
        """
        # QR decomposition of random matrix gives orthogonal matrix
        random_matrix = np.random.randn(n, n)
        q, r = np.linalg.qr(random_matrix)
        # Ensure proper rotation (det = 1)
        d = np.diag(np.sign(np.diag(r)))
        return (q @ d).astype(np.float32)

    def augment(self, x: NDArray[np.float32], is_multivariate: bool = False) -> NDArray[np.float32]:
        """Apply random augmentation to a time series sample.

        Args:
            x: Input time series.
            is_multivariate: Whether the series is multivariate.

        Returns:
            Augmented time series.
        """
        # Apply augmentation with probability
        if np.random.random() > self.augment_prob:
            return x
        
        # Randomly select one augmentation technique
        augmentations = [
            self.jitter,
            self.scaling,
            self.time_warp,
            self.window_slice
        ]
        
        # Add rotation for multivariate series
        if is_multivariate:
            augmentations.append(self.rotation)
        
        # Randomly select and apply one augmentation
        aug_func = np.random.choice(augmentations)
        return aug_func(x)

    def augment_batch(
        self,
        X: NDArray[np.float32],
        is_multivariate: bool = False
    ) -> NDArray[np.float32]:
        """Apply augmentation to a batch of time series.

        Args:
            X: Batch of time series of shape (batch_size, ...).
            is_multivariate: Whether the series are multivariate.

        Returns:
            Augmented batch.
        """
        augmented = np.zeros_like(X)
        for i in range(len(X)):
            augmented[i] = self.augment(X[i], is_multivariate)
        return augmented
