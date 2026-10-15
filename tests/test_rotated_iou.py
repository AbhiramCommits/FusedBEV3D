"""Rotated BEV IoU: hand-checkable polygon-clipping cases."""

from __future__ import annotations

import math

import numpy as np

from fusedbev3d.eval.detection import (
    box_corners,
    polygon_area,
    polygon_clip,
    rotated_iou_bev,
)


def test_identical_boxes_have_iou_one():
    """A box against itself is a perfect overlap."""
    box = np.array([[0.0, 0.0, 4.0, 1.8, 0.3]])
    assert math.isclose(float(rotated_iou_bev(box, box)[0, 0]), 1.0, abs_tol=1e-9)


def test_disjoint_boxes_have_iou_zero():
    """Far apart boxes share no area."""
    a = np.array([[0.0, 0.0, 4.0, 1.8, 0.0]])
    b = np.array([[20.0, 0.0, 4.0, 1.8, 0.0]])
    assert float(rotated_iou_bev(a, b)[0, 0]) == 0.0


def test_touching_boxes_have_iou_zero():
    """Boxes that only share an edge have zero intersection area."""
    a = np.array([[0.0, 0.0, 2.0, 2.0, 0.0]])
    b = np.array([[2.0, 0.0, 2.0, 2.0, 0.0]])
    assert float(rotated_iou_bev(a, b)[0, 0]) == 0.0


def test_forty_five_degree_analytic_case():
    """A square rotated by 45 degrees about its centre overlaps with IoU = 1/sqrt(2).

    The intersection of a square of side ``s`` with its 45-degree rotation is a
    regular octagon of area ``2(sqrt(2) - 1) s^2``; the union is ``2 s^2`` minus
    that, which simplifies to an IoU of exactly ``1 / sqrt(2)``.
    """
    square = np.array([[0.0, 0.0, 2.0, 2.0, 0.0]])
    rotated = np.array([[0.0, 0.0, 2.0, 2.0, math.pi / 4]])
    iou = float(rotated_iou_bev(square, rotated)[0, 0])
    assert abs(iou - 1.0 / math.sqrt(2.0)) < 1e-3


def test_ninety_degree_rectangle_overlap_is_analytic():
    """A 4x2 rectangle rotated 90 degrees overlaps a 2x2 square by 2 m^2."""
    rect = np.array([[0.0, 0.0, 4.0, 2.0, 0.0]])
    turned = np.array([[0.0, 0.0, 4.0, 2.0, math.pi / 2]])
    iou = float(rotated_iou_bev(rect, turned)[0, 0])
    # intersection is the 2x2 square at the origin: 4 m^2 of 8 m^2 each
    assert math.isclose(iou, 4.0 / (8.0 + 8.0 - 4.0), abs_tol=1e-6)


def test_iou_matrix_is_symmetric_and_bounded():
    """The full pairwise matrix is symmetric with values in [0, 1]."""
    rng = np.random.default_rng(0)
    a = np.stack(
        [rng.uniform(-5, 5, 6), rng.uniform(-5, 5, 6), rng.uniform(2, 5, 6),
         rng.uniform(1, 2, 6), rng.uniform(0, math.pi, 6)],
        axis=1,
    )
    m = rotated_iou_bev(a, a)
    assert np.allclose(m, m.T, atol=1e-9)
    assert np.all(m >= 0) and np.all(m <= 1 + 1e-9)
    assert np.allclose(np.diag(m), 1.0, atol=1e-9)


def test_eleven_column_layout_is_accepted():
    """The 11-column box layout is interpreted identically to the 5-column one."""
    full = np.array([[1.0, 2.0, 0.9, 4.2, 1.8, 1.6, 0.3, 1.0, 0.0, 0.0, 5.0]])
    compact = np.array([[1.0, 2.0, 4.2, 1.8, 0.3]])
    assert math.isclose(
        float(rotated_iou_bev(full, full)[0, 0]),
        float(rotated_iou_bev(compact, compact)[0, 0]),
        abs_tol=1e-9,
    )
    assert math.isclose(
        float(rotated_iou_bev(full, np.array([[3.0, 2.0, 0.9, 4.2, 1.8, 1.6, 0.3, 0.0, 0.0, 0.0, 6.0]]))[0, 0]),
        float(rotated_iou_bev(compact, np.array([[3.0, 2.0, 4.2, 1.8, 0.3]]))[0, 0]),
        abs_tol=1e-9,
    )


def test_polygon_helpers_are_consistent():
    """Clipping a square by itself reproduces its area; corners are counter-clockwise."""
    square = np.array([[0.0, 0.0, 2.0, 2.0, 0.0]])
    corners = box_corners(square)[0]
    assert math.isclose(polygon_area(corners), 4.0, abs_tol=1e-9)
    assert math.isclose(polygon_area(polygon_clip(corners, corners)), 4.0, abs_tol=1e-9)
    assert polygon_clip(corners, box_corners(np.array([[50.0, 50.0, 2.0, 2.0, 0.0]]))[0]).shape[0] == 0
