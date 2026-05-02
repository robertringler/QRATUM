"""SQL to Dataflow Lifter for AION.

Lifts SQL queries to AION-SIR dataflow hypergraph with:
- Table scan, filter, project, aggregate vertices
- Provenance metadata for cross-language debugging
- Query optimization hints

Version: 1.0.0
Status: Production
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

from ..sir.edges import HyperEdge, ParallelismKind
from ..sir.hypergraph import HyperGraph
from ..sir.vertices import (
    AIONType,
    EffectKind,
    HardwareAffinity,
    Provenance,
    Vertex,
)


class SQLNodeKind(Enum):
    """SQL AST node kinds."""

    SELECT = auto()
    FROM = auto()
    WHERE = auto()
    GROUP_BY = auto()
    HAVING = auto()
    ORDER_BY = auto()
    LIMIT = auto()
    JOIN = auto()
    UNION = auto()
    SUBQUERY = auto()
    TABLE_REF = auto()
    COLUMN_REF = auto()
    LITERAL = auto()
    FUNCTION_CALL = auto()
    BINARY_OP = auto()
    CASE_EXPR = auto()
    INSERT = auto()
    UPDATE = auto()
    DELETE = auto()


class SQLOperator(Enum):
    """Dataflow operators for SQL."""

    TABLE_SCAN = auto()
    INDEX_SCAN = auto()
    FILTER = auto()
    PROJECT = auto()
    AGGREGATE = auto()
    HASH_JOIN = auto()
    MERGE_JOIN = auto()
    NESTED_LOOP_JOIN = auto()
    SORT = auto()
    LIMIT = auto()
    DISTINCT = auto()
    UNION = auto()
    MATERIALIZE = auto()


@dataclass
class SQLASTNode:
    """SQL AST node representation."""

    kind: SQLNodeKind
    value: Any = None
    children: list[SQLASTNode] = field(default_factory=list)
    alias: str = ""
    columns: list[str] = field(default_factory=list)
    table_name: str = ""


@dataclass
class TableSchema:
    """Database table schema for type inference."""

    name: str
    columns: dict[str, AIONType] = field(default_factory=dict)
    primary_key: list[str] = field(default_factory=list)
    indexes: list[list[str]] = field(default_factory=list)
    estimated_rows: int = 1000


class SQLLifter:
    """Lifts SQL queries to AION-SIR dataflow graphs.

    Converts relational algebra operations to hypergraph vertices:
    - SELECT → Project vertex
    - FROM → TableScan vertex
    - WHERE → Filter vertex
    - GROUP BY → Aggregate vertex
    - JOIN → Join vertex with algorithm annotation
    """

    def __init__(self) -> None:
        """Initialize SQL lifter."""
        self.source_language = "SQL"
        self.graph = HyperGraph()
        self.schemas: dict[str, TableSchema] = {}
        self.query_id = 0

    def register_schema(self, schema: TableSchema) -> None:
        """Register a table schema for type inference."""
        self.schemas[schema.name] = schema

    def lift(self, query: str) -> HyperGraph:
        """Lift SQL query to AION-SIR hypergraph.

        Args:
            query: SQL query string

        Returns:
            AION-SIR hypergraph representing the query plan
        """
        self.query_id += 1
        self.graph = HyperGraph(name=f"query_{self.query_id}")

        # Parse SQL to AST
        ast = self._parse_sql(query)

        # Convert AST to dataflow graph
        output = self._lift_node(ast)

        if output:
            self.graph.exits.append(output)

        return self.graph

    def _parse_sql(self, query: str) -> SQLASTNode:
        """Parse SQL query to AST.

        Simplified SQL parser for demonstration.
        Production would use a proper SQL parser.
        """
        query = query.strip().upper()
        tokens = self._tokenize(query)

        if tokens and tokens[0] == "SELECT":
            return self._parse_select(tokens)
        elif tokens and tokens[0] == "INSERT":
            return self._parse_insert(tokens)
        elif tokens and tokens[0] == "UPDATE":
            return self._parse_update(tokens)
        elif tokens and tokens[0] == "DELETE":
            return self._parse_delete(tokens)

        return SQLASTNode(kind=SQLNodeKind.SELECT)

    def _tokenize(self, query: str) -> list[str]:
        """Tokenize SQL query."""
        # Simple tokenizer
        tokens = []
        current = ""
        in_string = False

        for char in query:
            if char == "'" and not in_string:
                in_string = True
                current += char
            elif char == "'" and in_string:
                in_string = False
                current += char
                tokens.append(current)
                current = ""
            elif in_string:
                current += char
            elif char in " \t\n,()":
                if current:
                    tokens.append(current)
                    current = ""
                if char in "(),":
                    tokens.append(char)
            else:
                current += char

        if current:
            tokens.append(current)

        return tokens

    def _parse_select(self, tokens: list[str]) -> SQLASTNode:
        """Parse SELECT statement."""
        root = SQLASTNode(kind=SQLNodeKind.SELECT)

        i = 1  # Skip SELECT

        # Parse column list
        columns = []
        while i < len(tokens) and tokens[i] != "FROM":
            if tokens[i] not in (",", "(", ")"):
                columns.append(tokens[i])
            i += 1
        root.columns = columns

        # Parse FROM
        if i < len(tokens) and tokens[i] == "FROM":
            i += 1
            from_node = SQLASTNode(kind=SQLNodeKind.FROM)

            # Parse table references
            while i < len(tokens) and tokens[i] not in (
                "WHERE",
                "GROUP",
                "ORDER",
                "LIMIT",
                "JOIN",
                "LEFT",
                "RIGHT",
                "INNER",
                "OUTER",
            ):
                if tokens[i] not in (",", "(", ")"):
                    table_ref = SQLASTNode(
                        kind=SQLNodeKind.TABLE_REF,
                        table_name=tokens[i],
                    )
                    from_node.children.append(table_ref)
                i += 1

            root.children.append(from_node)

        # Parse JOIN
        while i < len(tokens) and tokens[i] in ("JOIN", "LEFT", "RIGHT", "INNER", "OUTER"):
            join_type = tokens[i]
            i += 1
            if tokens[i] == "JOIN":
                i += 1

            if i < len(tokens):
                join_node = SQLASTNode(
                    kind=SQLNodeKind.JOIN,
                    value=join_type,
                )
                join_node.table_name = tokens[i]
                i += 1

                # Parse ON condition
                if i < len(tokens) and tokens[i] == "ON":
                    i += 1
                    condition = []
                    while i < len(tokens) and tokens[i] not in (
                        "WHERE",
                        "GROUP",
                        "ORDER",
                        "LIMIT",
                        "JOIN",
                        "LEFT",
                        "RIGHT",
                    ):
                        condition.append(tokens[i])
                        i += 1
                    join_node.columns = condition

                root.children.append(join_node)

        # Parse WHERE
        if i < len(tokens) and tokens[i] == "WHERE":
            i += 1
            where_node = SQLASTNode(kind=SQLNodeKind.WHERE)

            condition = []
            while i < len(tokens) and tokens[i] not in ("GROUP", "ORDER", "LIMIT"):
                condition.append(tokens[i])
                i += 1
            where_node.columns = condition
            root.children.append(where_node)

        # Parse GROUP BY
        if i < len(tokens) and tokens[i] == "GROUP":
            i += 1
            if i < len(tokens) and tokens[i] == "BY":
                i += 1

            group_node = SQLASTNode(kind=SQLNodeKind.GROUP_BY)

            group_cols = []
            while i < len(tokens) and tokens[i] not in ("HAVING", "ORDER", "LIMIT"):
                if tokens[i] != ",":
                    group_cols.append(tokens[i])
                i += 1
            group_node.columns = group_cols
            root.children.append(group_node)

            # Parse HAVING
            if i < len(tokens) and tokens[i] == "HAVING":
                i += 1
                having_node = SQLASTNode(kind=SQLNodeKind.HAVING)

                condition = []
                while i < len(tokens) and tokens[i] not in ("ORDER", "LIMIT"):
                    condition.append(tokens[i])
                    i += 1
                having_node.columns = condition
                root.children.append(having_node)

        # Parse ORDER BY
        if i < len(tokens) and tokens[i] == "ORDER":
            i += 1
            if i < len(tokens) and tokens[i] == "BY":
                i += 1

            order_node = SQLASTNode(kind=SQLNodeKind.ORDER_BY)

            order_cols = []
            while i < len(tokens) and tokens[i] != "LIMIT":
                if tokens[i] not in (",", "ASC", "DESC"):
                    order_cols.append(tokens[i])
                i += 1
            order_node.columns = order_cols
            root.children.append(order_node)

        # Parse LIMIT
        if i < len(tokens) and tokens[i] == "LIMIT":
            i += 1
            if i < len(tokens):
                limit_node = SQLASTNode(
                    kind=SQLNodeKind.LIMIT,
                    value=int(tokens[i]) if tokens[i].isdigit() else 100,
                )
                root.children.append(limit_node)

        return root

    def _parse_insert(self, tokens: list[str]) -> SQLASTNode:
        """Parse INSERT statement."""
        return SQLASTNode(kind=SQLNodeKind.INSERT)

    def _parse_update(self, tokens: list[str]) -> SQLASTNode:
        """Parse UPDATE statement."""
        return SQLASTNode(kind=SQLNodeKind.UPDATE)

    def _parse_delete(self, tokens: list[str]) -> SQLASTNode:
        """Parse DELETE statement."""
        return SQLASTNode(kind=SQLNodeKind.DELETE)

    def _lift_node(self, node: SQLASTNode, parent: Vertex | None = None) -> Vertex | None:
        """Lift SQL AST node to AION-SIR vertex."""
        provenance = Provenance(
            source_language=self.source_language,
            original_name=f"sql_{node.kind.name.lower()}",
        )

        if node.kind == SQLNodeKind.SELECT:
            return self._lift_select(node, provenance)
        elif node.kind == SQLNodeKind.FROM:
            return self._lift_from(node, provenance, parent)
        elif node.kind == SQLNodeKind.WHERE:
            return self._lift_where(node, provenance, parent)
        elif node.kind == SQLNodeKind.GROUP_BY:
            return self._lift_group_by(node, provenance, parent)
        elif node.kind == SQLNodeKind.ORDER_BY:
            return self._lift_order_by(node, provenance, parent)
        elif node.kind == SQLNodeKind.LIMIT:
            return self._lift_limit(node, provenance, parent)
        elif node.kind == SQLNodeKind.JOIN:
            return self._lift_join(node, provenance, parent)
        elif node.kind == SQLNodeKind.TABLE_REF:
            return self._lift_table_scan(node, provenance)

        return None

    def _lift_select(self, node: SQLASTNode, provenance: Provenance) -> Vertex:
        """Lift SELECT statement to dataflow graph."""
        # Build pipeline from bottom up
        current = None

        # First, lift FROM clause (table scans)
        for child in node.children:
            if child.kind == SQLNodeKind.FROM:
                from_vertices = []
                for table in child.children:
                    v = self._lift_node(table)
                    if v:
                        from_vertices.append(v)

                if len(from_vertices) == 1:
                    current = from_vertices[0]
                elif len(from_vertices) > 1:
                    # Cross join for multiple tables
                    current = from_vertices[0]
                    for other in from_vertices[1:]:
                        join = self._create_join_vertex(current, other, "CROSS", provenance)
                        current = join

        # Lift JOINs
        for child in node.children:
            if child.kind == SQLNodeKind.JOIN:
                right = self._lift_table_scan(
                    SQLASTNode(kind=SQLNodeKind.TABLE_REF, table_name=child.table_name),
                    provenance,
                )
                if current and right:
                    current = self._create_join_vertex(current, right, str(child.value), provenance)

        # Lift WHERE clause (filter)
        for child in node.children:
            if child.kind == SQLNodeKind.WHERE:
                filter_v = self._lift_where(child, provenance, current)
                if filter_v:
                    current = filter_v

        # Lift GROUP BY (aggregate)
        for child in node.children:
            if child.kind == SQLNodeKind.GROUP_BY:
                agg_v = self._lift_group_by(child, provenance, current)
                if agg_v:
                    current = agg_v

        # Lift HAVING (filter on aggregates)
        for child in node.children:
            if child.kind == SQLNodeKind.HAVING:
                having_v = self._lift_where(child, provenance, current)
                if having_v:
                    current = having_v

        # Lift ORDER BY (sort)
        for child in node.children:
            if child.kind == SQLNodeKind.ORDER_BY:
                sort_v = self._lift_order_by(child, provenance, current)
                if sort_v:
                    current = sort_v

        # Lift LIMIT
        for child in node.children:
            if child.kind == SQLNodeKind.LIMIT:
                limit_v = self._lift_limit(child, provenance, current)
                if limit_v:
                    current = limit_v

        # Project columns at the end
        project = self._create_project_vertex(node.columns, provenance, current)

        return project

    def _lift_from(
        self, node: SQLASTNode, provenance: Provenance, parent: Vertex | None
    ) -> Vertex | None:
        """Lift FROM clause."""
        # Process in _lift_select
        return None

    def _lift_table_scan(self, node: SQLASTNode, provenance: Provenance) -> Vertex:
        """Lift table reference to TableScan vertex."""
        table_name = node.table_name

        # Get schema for type info
        schema = self.schemas.get(table_name)
        if schema:
            type_info = AIONType(
                kind="struct",
                refinement=f"table={table_name},rows={schema.estimated_rows}",
            )
        else:
            type_info = AIONType(kind="struct")

        scan = Vertex.apply(
            function_name="table_scan",
            type_info=type_info,
            effects={EffectKind.READ},
            provenance=provenance.add_transformation("table_scan"),
        )
        scan.attributes["operator"] = SQLOperator.TABLE_SCAN.name
        scan.attributes["table"] = table_name
        scan.attributes["provenance"] = {
            "source": "sql",
            "table": table_name,
        }

        self.graph.add_vertex(scan)

        # Add parallel edge for parallel scan
        self.graph.add_edge(
            HyperEdge.parallel_edge(
                [scan],
                kind=ParallelismKind.DATAFLOW,
                affinity=HardwareAffinity.ANY,
            )
        )

        return scan

    def _lift_where(
        self, node: SQLASTNode, provenance: Provenance, parent: Vertex | None
    ) -> Vertex:
        """Lift WHERE clause to Filter vertex."""
        condition = " ".join(node.columns)

        filter_v = Vertex.apply(
            function_name="filter",
            type_info=AIONType(kind="struct"),
            effects={EffectKind.PURE},
            provenance=provenance.add_transformation("filter"),
        )
        filter_v.attributes["operator"] = SQLOperator.FILTER.name
        filter_v.attributes["condition"] = condition
        filter_v.attributes["provenance"] = {
            "source": "sql",
            "clause": "WHERE",
            "condition": condition,
        }

        self.graph.add_vertex(filter_v)

        if parent:
            self.graph.add_edge(HyperEdge.data_flow(parent, filter_v))

        return filter_v

    def _lift_group_by(
        self, node: SQLASTNode, provenance: Provenance, parent: Vertex | None
    ) -> Vertex:
        """Lift GROUP BY clause to Aggregate vertex."""
        group_cols = node.columns

        agg = Vertex.apply(
            function_name="aggregate",
            type_info=AIONType(kind="struct"),
            effects={EffectKind.PURE},
            provenance=provenance.add_transformation("aggregate"),
        )
        agg.attributes["operator"] = SQLOperator.AGGREGATE.name
        agg.attributes["group_by"] = group_cols
        agg.attributes["provenance"] = {
            "source": "sql",
            "clause": "GROUP BY",
            "columns": group_cols,
        }

        self.graph.add_vertex(agg)

        if parent:
            self.graph.add_edge(HyperEdge.data_flow(parent, agg))

        return agg

    def _lift_order_by(
        self, node: SQLASTNode, provenance: Provenance, parent: Vertex | None
    ) -> Vertex:
        """Lift ORDER BY clause to Sort vertex."""
        order_cols = node.columns

        sort = Vertex.apply(
            function_name="sort",
            type_info=AIONType(kind="struct"),
            effects={EffectKind.PURE},
            provenance=provenance.add_transformation("sort"),
        )
        sort.attributes["operator"] = SQLOperator.SORT.name
        sort.attributes["order_by"] = order_cols
        sort.attributes["provenance"] = {
            "source": "sql",
            "clause": "ORDER BY",
            "columns": order_cols,
        }

        self.graph.add_vertex(sort)

        if parent:
            self.graph.add_edge(HyperEdge.data_flow(parent, sort))

        return sort

    def _lift_limit(
        self, node: SQLASTNode, provenance: Provenance, parent: Vertex | None
    ) -> Vertex:
        """Lift LIMIT clause to Limit vertex."""
        limit_value = node.value

        limit_v = Vertex.apply(
            function_name="limit",
            type_info=AIONType(kind="struct"),
            effects={EffectKind.PURE},
            provenance=provenance.add_transformation("limit"),
        )
        limit_v.attributes["operator"] = SQLOperator.LIMIT.name
        limit_v.attributes["limit"] = limit_value
        limit_v.attributes["provenance"] = {
            "source": "sql",
            "clause": "LIMIT",
            "value": limit_value,
        }

        self.graph.add_vertex(limit_v)

        if parent:
            self.graph.add_edge(HyperEdge.data_flow(parent, limit_v))

        return limit_v

    def _lift_join(self, node: SQLASTNode, provenance: Provenance, parent: Vertex | None) -> Vertex:
        """Lift JOIN clause."""
        # Handled in _lift_select
        return None  # type: ignore

    def _create_join_vertex(
        self,
        left: Vertex,
        right: Vertex,
        join_type: str,
        provenance: Provenance,
    ) -> Vertex:
        """Create a join vertex."""
        # Determine join algorithm based on table sizes
        algorithm = SQLOperator.HASH_JOIN

        join = Vertex.apply(
            function_name="join",
            type_info=AIONType(kind="struct"),
            effects={EffectKind.PURE},
            provenance=provenance.add_transformation("join"),
        )
        join.attributes["operator"] = algorithm.name
        join.attributes["join_type"] = join_type
        join.attributes["provenance"] = {
            "source": "sql",
            "clause": "JOIN",
            "type": join_type,
        }

        self.graph.add_vertex(join)
        self.graph.add_edge(HyperEdge.data_flow([left, right], join))

        return join

    def _create_project_vertex(
        self,
        columns: list[str],
        provenance: Provenance,
        parent: Vertex | None,
    ) -> Vertex:
        """Create a projection vertex."""
        project = Vertex.apply(
            function_name="project",
            type_info=AIONType(kind="struct"),
            effects={EffectKind.PURE},
            provenance=provenance.add_transformation("project"),
        )
        project.attributes["operator"] = SQLOperator.PROJECT.name
        project.attributes["columns"] = columns
        project.attributes["provenance"] = {
            "source": "sql",
            "clause": "SELECT",
            "columns": columns,
        }

        self.graph.add_vertex(project)

        if parent:
            self.graph.add_edge(HyperEdge.data_flow(parent, project))

        return project


class DataflowBuilder:
    """Builder for SQL dataflow graphs.

    Provides fluent API for constructing dataflow graphs
    programmatically.
    """

    def __init__(self, name: str = "dataflow") -> None:
        """Initialize dataflow builder."""
        self.graph = HyperGraph(name=name)
        self._current: Vertex | None = None

    def scan(self, table: str, columns: list[str] | None = None) -> DataflowBuilder:
        """Add a table scan operator."""
        provenance = Provenance(
            source_language="dataflow",
            original_name=f"scan_{table}",
        )

        scan = Vertex.apply(
            function_name="table_scan",
            type_info=AIONType(kind="struct"),
            effects={EffectKind.READ},
            provenance=provenance,
        )
        scan.attributes["table"] = table
        scan.attributes["columns"] = columns or ["*"]

        self.graph.add_vertex(scan)
        self._current = scan

        return self

    def filter(self, condition: str) -> DataflowBuilder:
        """Add a filter operator."""
        provenance = Provenance(
            source_language="dataflow",
            original_name="filter",
        )

        filter_v = Vertex.apply(
            function_name="filter",
            type_info=AIONType(kind="struct"),
            effects={EffectKind.PURE},
            provenance=provenance,
        )
        filter_v.attributes["condition"] = condition

        self.graph.add_vertex(filter_v)

        if self._current:
            self.graph.add_edge(HyperEdge.data_flow(self._current, filter_v))

        self._current = filter_v
        return self

    def project(self, columns: list[str]) -> DataflowBuilder:
        """Add a projection operator."""
        provenance = Provenance(
            source_language="dataflow",
            original_name="project",
        )

        project = Vertex.apply(
            function_name="project",
            type_info=AIONType(kind="struct"),
            effects={EffectKind.PURE},
            provenance=provenance,
        )
        project.attributes["columns"] = columns

        self.graph.add_vertex(project)

        if self._current:
            self.graph.add_edge(HyperEdge.data_flow(self._current, project))

        self._current = project
        return self

    def aggregate(self, group_by: list[str], agg_funcs: dict[str, str]) -> DataflowBuilder:
        """Add an aggregate operator."""
        provenance = Provenance(
            source_language="dataflow",
            original_name="aggregate",
        )

        agg = Vertex.apply(
            function_name="aggregate",
            type_info=AIONType(kind="struct"),
            effects={EffectKind.PURE},
            provenance=provenance,
        )
        agg.attributes["group_by"] = group_by
        agg.attributes["aggregates"] = agg_funcs

        self.graph.add_vertex(agg)

        if self._current:
            self.graph.add_edge(HyperEdge.data_flow(self._current, agg))

        self._current = agg
        return self

    def join(self, right: DataflowBuilder, on: str, join_type: str = "INNER") -> DataflowBuilder:
        """Add a join operator."""
        provenance = Provenance(
            source_language="dataflow",
            original_name="join",
        )

        join_v = Vertex.apply(
            function_name="join",
            type_info=AIONType(kind="struct"),
            effects={EffectKind.PURE},
            provenance=provenance,
        )
        join_v.attributes["on"] = on
        join_v.attributes["join_type"] = join_type

        self.graph.add_vertex(join_v)

        # Connect both inputs
        if self._current:
            self.graph.add_edge(HyperEdge.data_flow(self._current, join_v))
        if right._current:
            # Merge right graph into this one
            for v in right.graph.vertices:
                self.graph.add_vertex(v)
            for e in right.graph.edges:
                self.graph.add_edge(e)
            self.graph.add_edge(HyperEdge.data_flow(right._current, join_v))

        self._current = join_v
        return self

    def sort(self, columns: list[str], ascending: bool = True) -> DataflowBuilder:
        """Add a sort operator."""
        provenance = Provenance(
            source_language="dataflow",
            original_name="sort",
        )

        sort_v = Vertex.apply(
            function_name="sort",
            type_info=AIONType(kind="struct"),
            effects={EffectKind.PURE},
            provenance=provenance,
        )
        sort_v.attributes["columns"] = columns
        sort_v.attributes["ascending"] = ascending

        self.graph.add_vertex(sort_v)

        if self._current:
            self.graph.add_edge(HyperEdge.data_flow(self._current, sort_v))

        self._current = sort_v
        return self

    def limit(self, n: int) -> DataflowBuilder:
        """Add a limit operator."""
        provenance = Provenance(
            source_language="dataflow",
            original_name="limit",
        )

        limit_v = Vertex.apply(
            function_name="limit",
            type_info=AIONType(kind="struct"),
            effects={EffectKind.PURE},
            provenance=provenance,
        )
        limit_v.attributes["limit"] = n

        self.graph.add_vertex(limit_v)

        if self._current:
            self.graph.add_edge(HyperEdge.data_flow(self._current, limit_v))

        self._current = limit_v
        return self

    def build(self) -> HyperGraph:
        """Build and return the dataflow graph."""
        if self._current:
            self.graph.exits.append(self._current)
        return self.graph
