"""Tests for backend/core/dts/tree.py."""

import pytest

from backend.core.dts.tree import DialogueTree, generate_node_id
from backend.core.dts.types import DialogueNode, NodeStatus, Strategy

# -----------------------------------------------------------------------------
# generate_node_id Tests
# -----------------------------------------------------------------------------


class TestGenerateNodeId:
    """Tests for generate_node_id function."""

    def test_generates_unique_ids(self) -> None:
        """Test that generated IDs are unique."""
        ids = [generate_node_id() for _ in range(100)]
        assert len(ids) == len(set(ids))

    def test_generates_valid_uuid(self) -> None:
        """Test that generated ID is a valid UUID string."""
        node_id = generate_node_id()
        assert isinstance(node_id, str)
        assert len(node_id) == 36  # UUID format: xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx


# -----------------------------------------------------------------------------
# DialogueTree Creation Tests
# -----------------------------------------------------------------------------


class TestDialogueTreeCreation:
    """Tests for DialogueTree creation and basic operations."""

    def test_create_tree(self) -> None:
        """Test creating a tree with a root node."""
        root = DialogueNode(id="root-1", depth=0)
        tree = DialogueTree.create(root)

        assert tree.root_id == "root-1"
        assert "root-1" in tree.nodes
        assert tree.get("root-1") is root

    def test_get_root(self) -> None:
        """Test getting the root node."""
        root = DialogueNode(id="root-1", depth=0)
        tree = DialogueTree.create(root)

        assert tree.get_root() is root

    def test_get_existing_node(self) -> None:
        """Test getting an existing node."""
        root = DialogueNode(id="root-1", depth=0)
        tree = DialogueTree.create(root)

        assert tree.get("root-1") is root

    def test_get_nonexistent_node_raises(self) -> None:
        """Test getting a non-existent node raises KeyError."""
        root = DialogueNode(id="root-1", depth=0)
        tree = DialogueTree.create(root)

        with pytest.raises(KeyError) as exc_info:
            tree.get("nonexistent")

        assert "nonexistent" in str(exc_info.value)


# -----------------------------------------------------------------------------
# Node Management Tests
# -----------------------------------------------------------------------------


class TestNodeManagement:
    """Tests for adding and removing nodes."""

    def test_add_node(self) -> None:
        """Test adding a node to the tree."""
        root = DialogueNode(id="root", depth=0)
        tree = DialogueTree.create(root)

        node = DialogueNode(id="child", depth=1)
        tree.add_node(node)

        assert "child" in tree.nodes
        assert tree.get("child") is node

    def test_add_child(self) -> None:
        """Test adding a child node under a parent."""
        root = DialogueNode(id="root", depth=0)
        tree = DialogueTree.create(root)

        child = DialogueNode(id="child")
        tree.add_child("root", child)

        assert child.parent_id == "root"
        assert child.depth == 1
        assert "child" in root.children
        assert tree.get("child") is child

    def test_add_multiple_children(self) -> None:
        """Test adding multiple children to a parent."""
        root = DialogueNode(id="root", depth=0)
        tree = DialogueTree.create(root)

        for i in range(3):
            child = DialogueNode(id=f"child-{i}")
            tree.add_child("root", child)

        assert len(root.children) == 3
        for i in range(3):
            assert f"child-{i}" in root.children

    def test_remove_node(self) -> None:
        """Test removing a node from the tree."""
        root = DialogueNode(id="root", depth=0)
        tree = DialogueTree.create(root)

        child = DialogueNode(id="child")
        tree.add_child("root", child)
        tree.remove_node("child")

        assert "child" not in tree.nodes
        assert "child" not in root.children

    def test_remove_nonexistent_node(self) -> None:
        """Test removing a non-existent node does nothing."""
        root = DialogueNode(id="root", depth=0)
        tree = DialogueTree.create(root)

        # Should not raise
        tree.remove_node("nonexistent")


# -----------------------------------------------------------------------------
# Node Query Tests
# -----------------------------------------------------------------------------


class TestNodeQueries:
    """Tests for querying nodes in the tree."""

    @pytest.fixture
    def sample_tree(self) -> DialogueTree:
        """Create a sample tree for testing."""
        root = DialogueNode(id="root", depth=0)
        tree = DialogueTree.create(root)

        # Add children
        for i in range(3):
            child = DialogueNode(
                id=f"child-{i}",
                strategy=Strategy(tagline=f"Strategy {i}", description="desc"),
            )
            tree.add_child("root", child)

        # Add grandchildren to first child
        for j in range(2):
            grandchild = DialogueNode(
                id=f"grandchild-{j}",
                strategy=Strategy(tagline=f"Grandchild {j}", description="desc"),
            )
            tree.add_child("child-0", grandchild)

        return tree

    def test_all_nodes(self, sample_tree: DialogueTree) -> None:
        """Test getting all nodes."""
        all_nodes = sample_tree.all_nodes()

        assert len(all_nodes) == 6  # root + 3 children + 2 grandchildren

    def test_active_nodes(self, sample_tree: DialogueTree) -> None:
        """Test getting active nodes."""
        # Prune one child
        sample_tree.get("child-1").status = NodeStatus.PRUNED

        active = sample_tree.active_nodes()

        assert len(active) == 5
        assert all(n.status == NodeStatus.ACTIVE for n in active)

    def test_active_leaves(self, sample_tree: DialogueTree) -> None:
        """Test getting active leaf nodes."""
        leaves = sample_tree.active_leaves()

        # Leaves are: child-1, child-2, grandchild-0, grandchild-1
        assert len(leaves) == 4
        assert all(len(n.children) == 0 for n in leaves)

    def test_leaves_at_depth(self, sample_tree: DialogueTree) -> None:
        """Test getting leaves at specific depth."""
        depth_1_leaves = sample_tree.leaves_at_depth(1)
        depth_2_leaves = sample_tree.leaves_at_depth(2)

        assert len(depth_1_leaves) == 2  # child-1, child-2
        assert len(depth_2_leaves) == 2  # grandchild-0, grandchild-1


# -----------------------------------------------------------------------------
# Path Tests
# -----------------------------------------------------------------------------


class TestTreePaths:
    """Tests for path operations."""

    @pytest.fixture
    def linear_tree(self) -> DialogueTree:
        """Create a linear tree (root -> child -> grandchild)."""
        root = DialogueNode(id="root", depth=0)
        tree = DialogueTree.create(root)

        child = DialogueNode(id="child")
        tree.add_child("root", child)

        grandchild = DialogueNode(id="grandchild")
        tree.add_child("child", grandchild)

        return tree

    def test_path_to_root(self, linear_tree: DialogueTree) -> None:
        """Test getting path from node to root."""
        path = linear_tree.path_to_root("grandchild")

        assert len(path) == 3
        assert path[0].id == "grandchild"
        assert path[1].id == "child"
        assert path[2].id == "root"

    def test_path_from_root(self, linear_tree: DialogueTree) -> None:
        """Test getting path from root to node."""
        path = linear_tree.path_from_root("grandchild")

        assert len(path) == 3
        assert path[0].id == "root"
        assert path[1].id == "child"
        assert path[2].id == "grandchild"


# -----------------------------------------------------------------------------
# Backpropagation Tests
# -----------------------------------------------------------------------------


class TestBackpropagation:
    """Tests for score backpropagation."""

    def test_backpropagate_updates_ancestors(self) -> None:
        """Test that backpropagate updates all ancestors."""
        root = DialogueNode(id="root", depth=0)
        tree = DialogueTree.create(root)

        child = DialogueNode(id="child")
        tree.add_child("root", child)

        grandchild = DialogueNode(id="grandchild")
        tree.add_child("child", grandchild)

        tree.backpropagate("grandchild", 8.0)

        assert grandchild.stats.visits == 1
        assert grandchild.stats.value_sum == 8.0
        assert grandchild.stats.value_mean == 8.0

        assert child.stats.visits == 1
        assert child.stats.value_sum == 8.0

        assert root.stats.visits == 1
        assert root.stats.value_sum == 8.0

    def test_backpropagate_multiple_times(self) -> None:
        """Test multiple backpropagations accumulate."""
        root = DialogueNode(id="root", depth=0)
        tree = DialogueTree.create(root)

        child = DialogueNode(id="child")
        tree.add_child("root", child)

        tree.backpropagate("child", 6.0)
        tree.backpropagate("child", 8.0)

        assert child.stats.visits == 2
        assert child.stats.value_sum == 14.0
        assert child.stats.value_mean == 7.0


# -----------------------------------------------------------------------------
# Pruning Tests
# -----------------------------------------------------------------------------


class TestPruning:
    """Tests for pruning operations."""

    def test_prune_node(self) -> None:
        """Test pruning a single node."""
        root = DialogueNode(id="root", depth=0)
        tree = DialogueTree.create(root)

        child = DialogueNode(id="child")
        tree.add_child("root", child)

        tree.prune_node("child", reason="score too low")

        assert child.status == NodeStatus.PRUNED
        assert child.prune_reason == "score too low"

    def test_prune_subtree(self) -> None:
        """Test pruning an entire subtree."""
        root = DialogueNode(id="root", depth=0)
        tree = DialogueTree.create(root)

        child = DialogueNode(id="child")
        tree.add_child("root", child)

        grandchild1 = DialogueNode(id="gc1")
        tree.add_child("child", grandchild1)

        grandchild2 = DialogueNode(id="gc2")
        tree.add_child("child", grandchild2)

        count = tree.prune_subtree("child", reason="branch failed")

        assert count == 3
        assert child.status == NodeStatus.PRUNED
        assert grandchild1.status == NodeStatus.PRUNED
        assert grandchild2.status == NodeStatus.PRUNED

    def test_prune_subtree_already_pruned(self) -> None:
        """Test pruning subtree when some nodes already pruned."""
        root = DialogueNode(id="root", depth=0)
        tree = DialogueTree.create(root)

        child = DialogueNode(id="child")
        tree.add_child("root", child)

        # Pre-prune the child
        child.status = NodeStatus.PRUNED

        count = tree.prune_subtree("child", reason="test")

        assert count == 0  # Already pruned


# -----------------------------------------------------------------------------
# Descendants Tests
# -----------------------------------------------------------------------------


class TestDescendants:
    """Tests for descendant operations."""

    def test_descendants(self) -> None:
        """Test iterating over descendants."""
        root = DialogueNode(id="root", depth=0)
        tree = DialogueTree.create(root)

        child = DialogueNode(id="child")
        tree.add_child("root", child)

        gc1 = DialogueNode(id="gc1")
        tree.add_child("child", gc1)

        gc2 = DialogueNode(id="gc2")
        tree.add_child("child", gc2)

        descendants = list(tree.descendants("root"))

        assert len(descendants) == 3
        ids = [d.id for d in descendants]
        assert "child" in ids
        assert "gc1" in ids
        assert "gc2" in ids

    def test_subtree_size(self) -> None:
        """Test getting subtree size."""
        root = DialogueNode(id="root", depth=0)
        tree = DialogueTree.create(root)

        child = DialogueNode(id="child")
        tree.add_child("root", child)

        gc1 = DialogueNode(id="gc1")
        tree.add_child("child", gc1)

        assert tree.subtree_size("root") == 3
        assert tree.subtree_size("child") == 2
        assert tree.subtree_size("gc1") == 1


# -----------------------------------------------------------------------------
# Statistics Tests
# -----------------------------------------------------------------------------


class TestTreeStatistics:
    """Tests for tree statistics."""

    def test_max_depth(self) -> None:
        """Test getting maximum depth."""
        root = DialogueNode(id="root", depth=0)
        tree = DialogueTree.create(root)

        child = DialogueNode(id="child")
        tree.add_child("root", child)

        grandchild = DialogueNode(id="grandchild")
        tree.add_child("child", grandchild)

        assert tree.max_depth() == 2

    def test_max_depth_empty(self) -> None:
        """Test max depth of empty tree."""
        tree = DialogueTree(root_id="none", nodes={})
        assert tree.max_depth() == 0

    def test_best_leaf(self) -> None:
        """Test getting best leaf by value_mean."""
        root = DialogueNode(id="root", depth=0)
        tree = DialogueTree.create(root)

        for i, score in enumerate([5.0, 8.0, 6.0]):
            child = DialogueNode(id=f"child-{i}")
            child.stats.value_mean = score
            tree.add_child("root", child)

        best = tree.best_leaf()

        assert best is not None
        assert best.id == "child-1"
        assert best.stats.value_mean == 8.0

    def test_best_leaf_by_score(self) -> None:
        """Test getting best leaf by aggregated_score."""
        root = DialogueNode(id="root", depth=0)
        tree = DialogueTree.create(root)

        for i, score in enumerate([5.0, 8.0, 6.0]):
            child = DialogueNode(id=f"child-{i}")
            child.stats.aggregated_score = score
            tree.add_child("root", child)

        best = tree.best_leaf_by_score()

        assert best is not None
        assert best.id == "child-1"

    def test_best_leaf_no_leaves(self) -> None:
        """Test best_leaf returns None when no active leaves."""
        root = DialogueNode(id="root", depth=0)
        root.status = NodeStatus.PRUNED
        tree = DialogueTree.create(root)

        assert tree.best_leaf() is None

    def test_statistics(self) -> None:
        """Test getting tree statistics."""
        root = DialogueNode(id="root", depth=0)
        tree = DialogueTree.create(root)

        child1 = DialogueNode(id="child1")
        tree.add_child("root", child1)

        child2 = DialogueNode(id="child2")
        child2.status = NodeStatus.PRUNED
        tree.add_child("root", child2)

        stats = tree.statistics()

        assert stats["total_nodes"] == 3
        assert stats["active_nodes"] == 2
        assert stats["pruned_nodes"] == 1
        assert stats["active_leaves"] == 1
        assert stats["max_depth"] == 1
