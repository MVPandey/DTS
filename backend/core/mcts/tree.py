"""MCTS Tree container and operations."""

from __future__ import annotations

import uuid
from typing import Iterator

from pydantic import BaseModel, Field

from backend.core.mcts.types import MCTSNode, NodeStatus


def generate_node_id() -> str:
    """Generate a unique node ID."""
    return str(uuid.uuid4())


class MCTSTree(BaseModel):
    """Container for MCTS tree with node management operations."""

    root_id: str
    nodes: dict[str, MCTSNode] = Field(default_factory=dict)

    model_config = {"arbitrary_types_allowed": True}

    @classmethod
    def create(cls, root_node: MCTSNode) -> "MCTSTree":
        """Create a new tree with the given root node."""
        tree = cls(root_id=root_node.id)
        tree.nodes[root_node.id] = root_node
        return tree

    def get(self, node_id: str) -> MCTSNode:
        """Get a node by ID. Raises KeyError if not found."""
        if node_id not in self.nodes:
            raise KeyError(f"Node {node_id} not found in tree")
        return self.nodes[node_id]

    def get_root(self) -> MCTSNode:
        """Get the root node."""
        return self.get(self.root_id)

    def add_node(self, node: MCTSNode) -> None:
        """Add a node to the tree."""
        self.nodes[node.id] = node

    def add_child(self, parent_id: str, child: MCTSNode) -> None:
        """Add a child node under a parent."""
        parent = self.get(parent_id)
        child.parent_id = parent_id
        child.depth = parent.depth + 1
        self.nodes[child.id] = child
        parent.children.append(child.id)

    def remove_node(self, node_id: str) -> None:
        """Remove a node from the tree (does not remove children)."""
        if node_id in self.nodes:
            node = self.nodes[node_id]
            if node.parent_id:
                parent = self.get(node.parent_id)
                if node_id in parent.children:
                    parent.children.remove(node_id)
            del self.nodes[node_id]

    def all_nodes(self) -> list[MCTSNode]:
        """Get all nodes in the tree."""
        return list(self.nodes.values())

    def active_nodes(self) -> list[MCTSNode]:
        """Get all active (non-pruned, non-error) nodes."""
        return [n for n in self.nodes.values() if n.status == NodeStatus.ACTIVE]

    def active_leaves(self) -> list[MCTSNode]:
        """Get all active leaf nodes (nodes with no children)."""
        return [
            n
            for n in self.nodes.values()
            if n.status == NodeStatus.ACTIVE and len(n.children) == 0
        ]

    def leaves_at_depth(self, depth: int) -> list[MCTSNode]:
        """Get all leaf nodes at a specific depth."""
        return [
            n
            for n in self.nodes.values()
            if n.depth == depth and len(n.children) == 0
        ]

    def path_to_root(self, node_id: str) -> list[MCTSNode]:
        """Get the path from a node to the root (inclusive)."""
        path = []
        current_id: str | None = node_id
        while current_id is not None:
            node = self.get(current_id)
            path.append(node)
            current_id = node.parent_id
        return path

    def path_from_root(self, node_id: str) -> list[MCTSNode]:
        """Get the path from root to a node (inclusive)."""
        return list(reversed(self.path_to_root(node_id)))

    def backpropagate(self, node_id: str, score: float) -> None:
        """
        Propagate a score from a leaf node up to the root.
        Updates visits, value_sum, and value_mean for each ancestor.
        """
        current_id: str | None = node_id
        while current_id is not None:
            node = self.get(current_id)
            node.stats.visits += 1
            node.stats.value_sum += score
            node.stats.value_mean = node.stats.value_sum / node.stats.visits
            current_id = node.parent_id

    def prune_node(self, node_id: str, reason: str | None = None) -> None:
        """Mark a node as pruned."""
        node = self.get(node_id)
        node.status = NodeStatus.PRUNED
        node.prune_reason = reason

    def prune_subtree(self, node_id: str, reason: str | None = None) -> int:
        """
        Mark a node and all its descendants as pruned.
        Returns the number of nodes pruned.
        """
        count = 0

        def _prune_recursive(nid: str) -> None:
            nonlocal count
            node = self.get(nid)
            if node.status != NodeStatus.PRUNED:
                node.status = NodeStatus.PRUNED
                node.prune_reason = reason
                count += 1
            for child_id in node.children:
                _prune_recursive(child_id)

        _prune_recursive(node_id)
        return count

    def descendants(self, node_id: str) -> Iterator[MCTSNode]:
        """Iterate over all descendants of a node (not including the node itself)."""
        node = self.get(node_id)
        for child_id in node.children:
            child = self.get(child_id)
            yield child
            yield from self.descendants(child_id)

    def subtree_size(self, node_id: str) -> int:
        """Get the number of nodes in the subtree rooted at node_id (including the node)."""
        return 1 + sum(1 for _ in self.descendants(node_id))

    def max_depth(self) -> int:
        """Get the maximum depth of any node in the tree."""
        if not self.nodes:
            return 0
        return max(n.depth for n in self.nodes.values())

    def best_leaf(self) -> MCTSNode | None:
        """Get the active leaf with the highest value_mean."""
        leaves = self.active_leaves()
        if not leaves:
            return None
        return max(leaves, key=lambda n: n.stats.value_mean)

    def best_leaf_by_score(self) -> MCTSNode | None:
        """Get the active leaf with the highest aggregated_score."""
        leaves = self.active_leaves()
        if not leaves:
            return None
        return max(leaves, key=lambda n: n.stats.aggregated_score)

    def statistics(self) -> dict:
        """Get tree statistics."""
        all_nodes = list(self.nodes.values())
        active = [n for n in all_nodes if n.status == NodeStatus.ACTIVE]
        pruned = [n for n in all_nodes if n.status == NodeStatus.PRUNED]
        leaves = self.active_leaves()

        return {
            "total_nodes": len(all_nodes),
            "active_nodes": len(active),
            "pruned_nodes": len(pruned),
            "active_leaves": len(leaves),
            "max_depth": self.max_depth(),
            "total_visits": sum(n.stats.visits for n in all_nodes),
        }
