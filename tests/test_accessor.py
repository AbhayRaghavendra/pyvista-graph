import jraph
import numpy as np
import pyvista as pv
import pytest

import pyvista_graph  # noqa: F401


def test_mesh_to_jraph_and_back():
    mesh = pv.PolyData(
        [[0.0, 0, 0], [1, 0, 0], [0, 1, 0]],
        lines=[2, 0, 1, 2, 2, 1],
    )
    mesh.point_data["weight"] = np.array([1.0, 2.0, 3.0])
    mesh.cell_data["cost"] = np.array([0.5, 1.5])
    mesh.field_data["label"] = np.array([[7]])

    graph = mesh.graph.to_jraph()

    assert isinstance(graph, jraph.GraphsTuple)
    np.testing.assert_array_equal(graph.n_node, [3])
    np.testing.assert_array_equal(graph.n_edge, [2])
    np.testing.assert_array_equal(graph.senders, [0, 2])
    np.testing.assert_array_equal(graph.receivers, [1, 1])
    np.testing.assert_array_equal(graph.nodes["positions"], mesh.points)
    np.testing.assert_array_equal(graph.nodes["weight"], [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(graph.edges["cost"], [0.5, 1.5])
    np.testing.assert_array_equal(graph.globals["label"], [[7]])

    restored = mesh.graph.to_polydata(graph)
    np.testing.assert_array_equal(restored.points, mesh.points)
    np.testing.assert_array_equal(restored.lines, mesh.lines)
    np.testing.assert_array_equal(restored.point_data["weight"], [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(restored.cell_data["cost"], [0.5, 1.5])
    np.testing.assert_array_equal(restored.field_data["label"], [7])
    np.testing.assert_array_equal(
        restored.graph.to_jraph().globals["label"], [[7]]
    )


def test_to_jraph_rejects_non_line_cells():
    mesh = pv.PolyData(
        [[0.0, 0, 0], [1, 0, 0], [0, 1, 0]],
        faces=[3, 0, 1, 2],
    )

    with pytest.raises(ValueError, match="only line cells"):
        mesh.graph.to_jraph()


def test_to_polydata_rejects_batched_graphs():
    graph = jraph.GraphsTuple(
        nodes=None,
        edges=None,
        receivers=np.empty(0, dtype=np.int32),
        senders=np.empty(0, dtype=np.int32),
        globals=None,
        n_node=np.array([1, 1]),
        n_edge=np.array([0, 0]),
    )

    with pytest.raises(ValueError, match="one-element"):
        pv.PolyData().graph.to_polydata(graph)
