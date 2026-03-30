"""Tests for FX graph canonicalization passes (AC-3c).

Verifies _simplify_fx_graph, _break_fx_diamonds, and _fold_scalar_inputs
work correctly as separate preprocessing passes.
"""

import torch
import torch.fx as fx
import pytest


def test_simplify_fx_graph_removes_cat_empty():
    """_simplify_fx_graph: cat([torch.tensor([]), x]) → x."""
    from tilelang.torch_compile.analysis import _simplify_fx_graph

    graph = fx.Graph()
    x = graph.placeholder("x")
    x.meta["example_value"] = torch.randn(4)
    empty = graph.call_function(torch.tensor, args=([],))
    empty.meta["example_value"] = torch.tensor([])
    cat = graph.call_function(torch.cat, args=([empty, x],))
    cat.meta["example_value"] = torch.randn(4)
    graph.output(cat)

    gm = fx.GraphModule(torch.nn.Module(), graph)
    _simplify_fx_graph(gm)
    cat_nodes = [
        n for n in gm.graph.nodes
        if n.op == "call_function" and getattr(n.target, "__name__", "") == "cat"
    ]
    assert len(cat_nodes) == 0


def test_break_fx_diamonds_splits_shared_node():
    """_break_fx_diamonds: cheap shared node gets duplicated for each user."""
    from tilelang.torch_compile.compiler import _break_fx_diamonds

    graph = fx.Graph()
    x = graph.placeholder("x")
    x.meta["example_value"] = torch.randn(4)
    # neg is cheap and has 2 users → should be duplicated
    neg = graph.call_function(torch.neg, args=(x,))
    neg.meta["example_value"] = torch.randn(4)
    add1 = graph.call_function(torch.add, args=(neg, x))
    add1.meta["example_value"] = torch.randn(4)
    add2 = graph.call_function(torch.add, args=(neg, x))
    add2.meta["example_value"] = torch.randn(4)
    out = graph.call_function(torch.add, args=(add1, add2))
    out.meta["example_value"] = torch.randn(4)
    graph.output(out)

    gm = fx.GraphModule(torch.nn.Module(), graph)
    # Before: neg has 2 users
    neg_nodes_before = [
        n for n in gm.graph.nodes
        if n.op == "call_function" and n.target == torch.neg
    ]
    assert len(neg_nodes_before) == 1
    assert len(neg_nodes_before[0].users) == 2

    _break_fx_diamonds(gm)

    # After: neg should be duplicated (2 separate neg nodes, each with 1 user)
    neg_nodes_after = [
        n for n in gm.graph.nodes
        if n.op == "call_function" and n.target == torch.neg
    ]
    assert len(neg_nodes_after) == 2
    for n in neg_nodes_after:
        assert len(n.users) == 1


def test_fold_scalar_inputs_folds_0d_tensor():
    """_fold_scalar_inputs: 0-d tensor placeholder → folded scalar."""
    from tilelang.torch_compile.compiler import _fold_scalar_inputs

    graph = fx.Graph()
    x = graph.placeholder("x")
    x.meta["example_value"] = torch.randn(4, 4)
    eps = graph.placeholder("eps")
    eps.meta["example_value"] = torch.tensor(1e-6)
    # eps.item() → should be folded
    eps_item = graph.call_method("item", args=(eps,))
    eps_item.meta["example_value"] = 1e-6
    add = graph.call_function(torch.add, args=(x, eps_item))
    add.meta["example_value"] = torch.randn(4, 4)
    graph.output(add)

    gm = fx.GraphModule(torch.nn.Module(), graph)
    concrete_args = (torch.randn(4, 4), torch.tensor(1e-6))
    folded_gm = _fold_scalar_inputs(gm, concrete_args)

    # After folding, eps placeholder should be removed or unused.
    placeholders = [n for n in folded_gm.graph.nodes if n.op == "placeholder"]
    # The eps placeholder may still exist but should have no users.
    eps_nodes = [n for n in placeholders if n.name == "eps"]
    if eps_nodes:
        assert len(eps_nodes[0].users) == 0
