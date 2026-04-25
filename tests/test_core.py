import numpy as np
import pytest

from FourierMesh import cartesian_DFT_dirac, normalize_points


def test_cartesian_dft_dirac_output_shape():
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ]
    )
    kx = np.array([-1.0, 0.0, 1.0])
    ky = np.array([0.0, 2.0])
    kz = np.array([-2.0, 0.0])

    spectrum = cartesian_DFT_dirac(points, kx, ky, kz)

    assert spectrum.shape == (3, 2, 2)


def test_cartesian_dft_dirac_dc_equals_number_of_points():
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 2.0, 3.0],
            [-1.0, 0.5, 2.0],
            [4.0, -2.0, 1.0],
        ]
    )
    k0 = np.array([0.0])

    spectrum = cartesian_DFT_dirac(points, k0, k0, k0)

    assert spectrum.shape == (1, 1, 1)
    assert spectrum[0, 0, 0] == pytest.approx(4.0 + 0.0j)


def test_cartesian_dft_dirac_rejects_invalid_point_shape():
    invalid_points = np.array([1.0, 2.0, 3.0])
    k = np.array([0.0, 1.0])

    with pytest.raises(ValueError):
        cartesian_DFT_dirac(invalid_points, k, k, k)


def test_normalize_points_centers_and_scales_to_unit_box():
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [2.0, 4.0, 6.0],
            [1.0, 2.0, 3.0],
        ]
    )

    normalized = normalize_points(points)

    mins = normalized.min(axis=0)
    maxs = normalized.max(axis=0)

    assert np.allclose((mins + maxs) / 2.0, 0.0)
    assert np.max(maxs - mins) == pytest.approx(2.0)
