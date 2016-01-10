# decision-trees
A Python library that implements sklearn-compatible decision trees (using CART algorithm).

## Rationale

Wait... Why should anyone want to write a DT package from scratch? That's simple:

- To better understand DTs and how they are constructed - there's no better way to understand something than to implement it
- To have more freedom in trying different approaches in DTs construction (e.g. oblivious trees, etc). 
With the trees from sklearn one would have to code it in Cython, which is nasty.

## Limitations & roadmap

- Implement tree pruning
- Add more parameters to tweak, make the interface more like `sklearn.tree.DecisionTree*` classes
- Speed things up?
