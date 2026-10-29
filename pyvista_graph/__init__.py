"""PyVista accessor for converting meshes to and from jraph graphs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import jraph
import numpy as np
import pyvista as pv

_POSITIONS_KEY = "pos"


def _feature_mapping(
    features: Any,
    count: int,
    association: str,
) -> dict[str, np.ndarray]:
    """Convert flat jraph feature arrays to named PyVista data arrays."""
    if features is None:
        return {}
    if isinstance(features, Mapping):
        mapped = dict(features)
    elif (
        association == "point"
        and np.asarray(features).ndim == 2
        and np.asarray(features).shape[1] == 3
    ):
        return {}
    else:
        mapped = {f"{association}_features": features}

    arrays: dict[str, np.ndarray] = {}
    for name, values in mapped.items():
        if not isinstance(name, str):
            raise TypeError(f"{association} feature names must be strings.")
        array = np.asarray(values)
        if array.ndim == 0 or array.shape[0] != count:
            raise ValueError(
                f"{association} feature {name!r} must have {count} entries."
            )
        arrays[name] = array
    return arrays


@pv.register_dataset_accessor("graph", pv.DataSet)
class GraphAccessor:
    """Convert between a PyVista mesh and a single-graph jraph ``GraphsTuple``.

    Mesh points are graph nodes and are included in ``nodes["pos"]``.
    Explicit two-point line cells are directed edges, with each line's point
    order defining its sender and receiver. Point, cell, and field data map to
    node, edge, and global features, respectively.
    """

    def __init__(self, dataset: pv.DataSet) -> None:
        self._dataset = dataset

    def to_jraph(self) -> jraph.GraphsTuple:
        """Convert this line-only ``PolyData`` mesh to a single ``GraphsTuple``."""
        mesh = self._dataset
        if not isinstance(mesh, pv.PolyData):
            raise TypeError("to_jraph() requires a pyvista.PolyData mesh.")
        if mesh.n_cells != mesh.n_lines:
            raise ValueError(
                "to_jraph() requires a mesh containing only line cells; "
                "vertices, polygons, and strips are not graph edges."
            )
        if _POSITIONS_KEY in mesh.point_data:
            raise ValueError(
                f"point_data name {_POSITIONS_KEY!r} is reserved for mesh coordinates."
            )

        senders: list[int] = []
        receivers: list[int] = []
        lines = np.asarray(mesh.lines, dtype=np.int64)
        offset = 0
        while offset < len(lines):
            size = int(lines[offset])
            if size != 2:
                raise ValueError(
                    "to_jraph() requires every line cell to contain exactly two points."
                )
            if offset + size >= len(lines):
                raise ValueError("Malformed line connectivity in PolyData.")
            senders.append(int(lines[offset + 1]))
            receivers.append(int(lines[offset + 2]))
            offset += size + 1

        nodes = {_POSITIONS_KEY: np.asarray(mesh.points)}
        nodes.update(
            {name: np.asarray(mesh.point_data[name]) for name in mesh.point_data}
        )
        edges = {name: np.asarray(mesh.cell_data[name]) for name in mesh.cell_data}
        globals_ = {}
        for name in mesh.field_data:
            values = np.asarray(mesh.field_data[name])
            if values.ndim == 1:
                values = values[np.newaxis, :]
            elif values.ndim == 0 or values.shape[0] != 1:
                raise ValueError(
                    f"field_data array {name!r} must describe exactly one graph."
                )
            globals_[name] = values

        return jraph.GraphsTuple(
            nodes=nodes,
            edges=edges or None,
            receivers=np.asarray(receivers, dtype=np.int64),
            senders=np.asarray(senders, dtype=np.int64),
            globals=globals_ or None,
            n_node=np.asarray([mesh.n_points], dtype=np.int32),
            n_edge=np.asarray([len(senders)], dtype=np.int32),
        )

    @staticmethod
    def to_polydata(graph: jraph.GraphsTuple) -> pv.PolyData:
        """Convert a single-graph ``GraphsTuple`` to a new ``PolyData`` mesh.

        Batched tuples are not supported. Node coordinates must be provided
        under ``graph.nodes["pos"]`` with shape ``(n_node, 3)``.
        """
        if not isinstance(graph, jraph.GraphsTuple):
            raise TypeError("to_polydata() requires a jraph.GraphsTuple.")

        n_node = GraphAccessor._single_count(graph.n_node, "n_node")
        n_edge = GraphAccessor._single_count(graph.n_edge, "n_edge")
        if n_node < 0 or n_edge < 0:
            raise ValueError("n_node and n_edge must be non-negative.")

        senders = GraphAccessor._indices(graph.senders, n_edge, "senders")
        receivers = GraphAccessor._indices(graph.receivers, n_edge, "receivers")
        if np.any(senders < 0) or np.any(senders >= n_node):
            raise ValueError("senders contains an index outside the graph's nodes.")
        if np.any(receivers < 0) or np.any(receivers >= n_node):
            raise ValueError("receivers contains an index outside the graph's nodes.")

        node_features = graph.nodes
        if isinstance(node_features, Mapping):
            if _POSITIONS_KEY not in node_features:
                raise ValueError(
                    f"graph.nodes must include the {_POSITIONS_KEY!r} position feature."
                )
            node_arrays = _feature_mapping(node_features, n_node, "point")
            positions = node_arrays.pop(_POSITIONS_KEY)
        else:
            raise ValueError(
                f"graph.nodes must be a mapping containing the "
                f"{_POSITIONS_KEY!r} position feature."
            )

        positions = np.asarray(positions)
        if positions.shape != (n_node, 3):
            raise ValueError(
                f"graph.nodes[{_POSITIONS_KEY!r}] must have shape ({n_node}, 3), "
                f"got {positions.shape}."
            )

        edge_arrays = _feature_mapping(graph.edges, n_edge, "cell")
        global_arrays = GraphAccessor._global_mapping(graph.globals)

        connectivity = np.column_stack(
            (np.full(n_edge, 2, dtype=np.int64), senders, receivers)
        ).ravel()
        mesh = pv.PolyData(positions, lines=connectivity)
        for name, values in node_arrays.items():
            mesh.point_data[name] = values
        for name, values in edge_arrays.items():
            mesh.cell_data[name] = values
        for name, values in global_arrays.items():
            mesh.field_data[name] = values
        return mesh

    @staticmethod
    def _single_count(values: Any, name: str) -> int:
        counts = np.asarray(values)
        if counts.shape != (1,) or not np.issubdtype(counts.dtype, np.integer):
            raise ValueError(f"{name} must be a one-element integer array.")
        return int(counts[0])

    @staticmethod
    def _indices(values: Any, count: int, name: str) -> np.ndarray:
        if values is None:
            if count:
                raise ValueError(f"{name} cannot be None when n_edge is non-zero.")
            return np.empty(0, dtype=np.int64)
        indices = np.asarray(values)
        if indices.shape != (count,) or not np.issubdtype(indices.dtype, np.integer):
            raise ValueError(f"{name} must be an integer array of shape ({count},).")
        return indices.astype(np.int64, copy=False)

    @staticmethod
    def _global_mapping(globals_: Any) -> dict[str, np.ndarray]:
        if globals_ is None:
            return {}
        if isinstance(globals_, Mapping):
            mapped = dict(globals_)
        else:
            mapped = {"global_features": globals_}

        arrays: dict[str, np.ndarray] = {}
        for name, values in mapped.items():
            if not isinstance(name, str):
                raise TypeError("Global feature names must be strings.")
            array = np.asarray(values)
            if array.ndim == 0 or array.shape[0] != 1:
                raise ValueError(
                    f"Global feature {name!r} must have a leading graph dimension of 1."
                )
            arrays[name] = array
        return arrays


__all__ = ["GraphAccessor"]
