# pyvista-graph

`pyvista-graph` adds a `graph` accessor to PyVista datasets. Import the package
to register it:

```python
import pyvista as pv
import pyvista_graph

mesh = pv.PolyData(
    [[0, 0, 0], [1, 0, 0]],
    lines=[2, 0, 1],
)
graph = mesh.graph.to_jraph()
mesh_again = mesh.graph.to_polydata(graph)
```

`to_jraph()` produces a `jraph.GraphsTuple` for one line-only `PolyData` mesh.
Mesh points become nodes and are stored as the `pos` node feature;
two-point line cells become directed edges, with point order specifying
sender-to-receiver direction. PyVista `point_data`, `cell_data`, and
`field_data` become node, edge, and global feature dictionaries. The
`pos` key is reserved for mesh coordinates. Polygons,
strips, vertices, and batched `GraphsTuple` values are not supported.
`to_polydata()` requires `graph.nodes["pos"]` to have shape
`(num_nodes, 3)`.

Install with `pip install pyvista-graph`. The accessor is registered by
`import pyvista_graph`, and requires PyVista 0.48 or later.
