# Pines: a Decision Trees ML library
A Python library that implements sklearn-compatible decision trees. 

Check the `examples` directory to see how to use the package for both classification and regression problems.

## List of things implemented
- CART algorithm for building regression and classification trees
- Oblivious Trees 
- Different splitting criteria: Gini index, entropy

## Rationale

Wait... Why would anyone want to implement his own DT package from scratch, while there's such thing as `slklearn`? That's some of the reasons:

- To better understand DTs and how they are constructed - there's no better way to understand something than to implement it;
- To have more freedom in trying different approaches in DTs construction (e.g. oblivious trees, etc). 
With the trees from sklearn one would have to code it in Cython, which is nasty.

## Limitations & roadmap

- Implement tree pruning
- Add more parameters to tweak, make the interface more like `sklearn.tree.DecisionTree*` classes
- Speed things up, maybe switch to Cython? 
