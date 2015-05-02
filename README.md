# 349BasicDecisionTree

```
Usage:
  id3.pl h
    provides usage help
  id3.pl t trainfile [pruning maxlevels]
    creates a decision tree based on data in trainfile
    prints tree to stdout
  id3.pl e trainfile testfile [pruning maxlevels]
    creates a decision tree based on data in trainfile
    tests decision tree on data in testfile
    prints expected classes to stdout with other test data in csv format
  id3.pl v trainfile validatefile [pruning maxlevels]
    creates a decision tree based on data inf trainfile
    tests decision tree on  data in validatefile
    compares results from tree with results in validatefile
    prints accuracy
  id3.pl l trainfile validatefile percent-of-tree [pruning maxlevels]
    validation on the full validation set but using different subsets of the training set to create the tree.
    produces and average result for different runs on different subsets of the training set.
    size of the subset is a percentage of the overall training set so .1 is 10% of the training set.
    
  append pruning maxlevels to any of the above to specify pruning as a boolean and maxlevels as an integer
```
