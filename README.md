# 349BasicDecisionTree

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

  append pruning maxlevels to any of the above to specify pruning as a boolean and maxlevels as an integer
