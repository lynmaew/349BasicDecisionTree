#!/usr/bin/perl
use feature ':5.10';
use strict;
use POSIX;

#tree data
my @tree_A;
my @tree_B;
my @tree_index;
my @tree_value;
my @tree_attribute;
my @tree_leaf;
my $attribute_line;
my @attribute_list;

#other global data
my $num_train_lines = 0;
my $num_attributes = 0;
my @train_data;
my @averages;
my @test_data;
my $num_test_lines = 0;
my @validate_data;
my $num_validate_lines = 0;

#parameters
my $max_levels = 12;
my $pruning = 1;
my $majority_threshold = 0.9;
my $share_threshold = 0.1;
my $reuse_attributes = 0;

#calls tree_gen to generate tree
sub train {
  my ($start, $end) = @_;


  #read in data, find number of attributes (should be at least 2)
  #assume first line is label line
  open (TRAIN_FILE, $ARGV[1]) or die "training file $ARGV[1] could not be read";
  my @train_file_lines = <TRAIN_FILE>;
  close (TRAIN_FILE);
  $attribute_line = $train_file_lines[0];
  splice(@train_file_lines, 0, 1);
  @attribute_list = split(',', $attribute_line);
  $num_attributes = $#attribute_list;
  if ($num_attributes < 1) {
    die "Not enough attributes to train on";
  }

  if ($start != 0 && $end != 0)
  {
    @train_file_lines = @train_file_lines[$start..$end];
  }


  #parse file
  #assume each line has same number of entries as first
  my $i = -1;
  foreach my $train_file_line (@train_file_lines) {
    $i = $i + 1;
    my @split_line = split(',', $train_file_line);
    for my $j (0..$num_attributes) {
      $train_data[$i][$j] = $split_line[$j];
    }
  }

  #clean input data
  #assign missing values
  $num_train_lines = $#train_file_lines;
  foreach my $j (0..($num_attributes - 1)) {
    my $sum = 0;
    foreach my $i (0..$num_train_lines) {
      if ($train_data[$i][$j] != "?") {
        $sum = $sum + $train_data[$i][$j];
      }
    }
    $averages[$j] = $sum / $num_train_lines;
    foreach my $i (0..$num_train_lines) {
      if($train_data[$i][$j] == "?") {
        $train_data[$i][$j] = $averages[$j];
      }
    }
  }

  my $upper_left = 0;
  my $upper_right = $num_train_lines;
  my $upper_node_index = 1;
  my @upper_unused_attributes = (0..($num_attributes-1));

  tree_gen($upper_left, $upper_right, $upper_node_index, $num_attributes, @upper_unused_attributes);
}

#a method for generating decision tree
sub tree_gen {
  my $left = $_[0];
  my $right = $_[1];
  my $node_index = $_[2];
  # my $num_attributes = $_[3]
  my @unused_attributes = @_[4..($_[3]+4)];

  my $count_A = 0;
  my $count_B = 0;
  foreach my $i ($left..$right) {
    if ($train_data[$i][$num_attributes] == 0) {
      $count_A = $count_A + 1;
    } elsif ($train_data[$i][$num_attributes] == 1) {
      $count_B = $count_B + 1;
    }
  }

  my $best_index = -1;
  my $best_entropy = 1;
  my $best_attribute = -1;

  foreach my $j (@unused_attributes) {
    @train_data[$left..$right] = sort { $a->[$j] <=> $b->[$j] } @train_data[$left..$right];
    
    my $local_A = 0;
    my $local_B = 0;
    my $prev_val = $train_data[$left][$j];
    foreach my $i ($left..$right) {
      if ($train_data[$i][$num_attributes] == 0) {
        $local_A = $local_A + 1;
      } elsif ($train_data[$i][$num_attributes] == 1) {
        $local_B = $local_B + 1;
      }

      my $p1 = $local_A / ($i - $left + 1);
      my $p2 = $local_B / ($i - $left + 1);
      my $p3 = ($count_A - $local_A) / ($right - $i + 1);
      my $p4 = ($count_B - $local_B) / ($right - $i + 1);
      my $e1 = 0;
      if ($p1 > 0) {
        $e1 = - (($i - $left + 1) / ($right - $left + 1)) * $p1 * log($p1);
      }
      my $e2 = 0;
      if ($p2 > 0) {
        $e2 = - (($i - $left + 1) / ($right - $left + 1)) * $p2 * log($p2);
      }
      my $e3 = 0;
      if ($p3 > 0) {
        $e3 = - (($right - $i + 1) / ($right - $left + 1)) * $p3 * log($p3);
      }
      my $e4 = 0;
      if ($p4 > 0) {
        $e4 = - (($right - $i + 1) / ($right - $left + 1)) * $p4 * log($p4);
      }
      my $local_entropy = $e1 + $e2 + $e3 + $e4;
      if (($local_entropy < $best_entropy) && (($train_data[$i][$j] != $prev_val) || ($i == $left))) {
        $best_entropy = $local_entropy;
        $best_index = $i;
        $best_attribute = $j;
      }
      $prev_val = $train_data[$i][$j];
    }
  }

  $tree_index[$node_index] = $best_index;
  $tree_attribute[$node_index] = $best_attribute;
  $tree_value[$node_index] = $train_data[$best_index][$best_attribute];
  $tree_A[$node_index] = $count_A;
  $tree_B[$node_index] = $count_B;
  $tree_leaf[$node_index] = 0;

  if ($pruning == 1) {
    #prune on size of example set for node
    if ($count_A + $count_B < $share_threshold * $num_train_lines) {
      $tree_leaf[$node_index] = 1;
    }
    #prune on ratio of classes
    if ((($count_A / ($count_A + $count_B)) > $majority_threshold) || (($count_A / ($count_A + $count_B)) < (1 - $majority_threshold))) {
      $tree_leaf[$node_index] = 1;
    }
  }
  if (($tree_leaf[$node_index] == 0) && ($node_index < 2**$max_levels)) {
    my @next_unused_array;
    if ($reuse_attributes == 1) {
      @next_unused_array = @unused_attributes;
    } else {
      @next_unused_array = grep {$_ != $best_attribute} @unused_attributes;
    }
    tree_gen($left, $best_index-1, ($node_index * 2), $num_attributes - 1, @next_unused_array);
    tree_gen($best_index, $right, (($node_index * 2) + 1), $num_attributes - 1, @next_unused_array);
  } else {
    $tree_leaf[$node_index] = 1;
  }
}

#prints tree to standard output
sub tree_print {
  my $node_index = $_[0];
  foreach my $i (1 .. floor(log($node_index)/log(2))) {
    print "|";
  }
  if ($tree_leaf[$node_index] == 0) {
    print "= Split on attribute ", $attribute_list[$tree_attribute[$node_index]];
    print " < ", $tree_value[$node_index], " into left subtree, else right subtree\n";
    tree_print($node_index * 2);
    tree_print($node_index * 2 + 1);
  } else {
    if($tree_A[$node_index] > $tree_B[$node_index]) {
      print "= Estimate class 0\n";
    } else {
      print "= Estimate class 1\n";
    }
  }
}

sub tree_count {
  my $node_index = $_[0];
  if($tree_leaf[$node_index] == 0) {
    return 1 + tree_count($node_index * 2) + tree_count($node_index * 2 + 1);
  }
  return 1;
}

#clean tree leaves from bottom up
sub tree_clean {
  my $node_index = $_[0];
  if ($tree_leaf[$node_index] == 0) {
    tree_clean($node_index * 2);
    tree_clean($node_index * 2 + 1);
    if ($tree_leaf[$node_index * 2] == 1 && $tree_leaf[$node_index * 2 + 1] == 1) {
      if (($tree_A[$node_index * 2] > $tree_B[$node_index * 2]) == ($tree_A[$node_index * 2 + 1] > $tree_B[$node_index * 2 + 1])) {
        $tree_leaf[$node_index] = 1;
      }
    }
  }
}


#routine to use test data
sub test {
  open (TEST_FILE, $ARGV[2]) or die "testing file $ARGV[2] could not be read";
  my @test_file_lines = <TEST_FILE>;
  close (TEST_FILE);
  splice(@test_file_lines, 0, 1);

  #parse file
  my $i = -1;
  foreach my $test_file_line (@test_file_lines) {
    $i = $i + 1;
    my @split_line = split(',', $test_file_line);
    for my $j (0..$num_attributes) {
      $test_data[$i][$j] = $split_line[$j];
    }
  }

  #clean input data
  #assign missing values
  $num_test_lines = $#test_file_lines;
  foreach my $j (0..($num_attributes - 1)) {
    foreach my $i (0..$num_test_lines) {
      if($test_data[$i][$j] == "?") {
        $test_data[$i][$j] = $averages[$j];
      }
    }
  }

  #estimate each value
  foreach my $i (0..$num_test_lines) {
    my $node_index = 1;
    while ($tree_leaf[$node_index] != 1) {
      if ($test_data[$i][$tree_attribute[$node_index]] < $tree_value[$node_index]) {
        $node_index = $node_index * 2;
      } else {
        $node_index = $node_index * 2 + 1;
      }
    }
    foreach my $j (0..$num_attributes) {
      print $test_data[$i][$j], ",";
    }
    if ($tree_A[$node_index] > $tree_B[$node_index]) {
      print "0\n";
    } else {
      print "1\n";
    }
  }
}

#routine to use validate data
sub validate {

  my $num_correct = 0;

  open (VALIDATE_FILE, $ARGV[2]) or die "validation file $ARGV[2] could not be read";
  my @validate_file_lines = <VALIDATE_FILE>;
  close (VALIDATE_FILE);
  splice(@validate_file_lines, 0, 1);

  #parse file
  my $i = -1;
  foreach my $validate_file_line (@validate_file_lines) {
    $i = $i + 1;
    my @split_line = split(',', $validate_file_line);
    for my $j (0..$num_attributes) {
      $validate_data[$i][$j] = $split_line[$j];
    }
  }

  #clean input data
  #assign missing values
  $num_validate_lines = $#validate_file_lines;
  foreach my $j (0..($num_attributes - 1)) {
    foreach my $i (0..$num_validate_lines) {
      if($validate_data[$i][$j] == "?") {
        $validate_data[$i][$j] = $averages[$j];
      }
    }
  }

  #estimate each value
  foreach my $i (0..$num_validate_lines) {
    my $node_index = 1;
    while ($tree_leaf[$node_index] != 1) {
      if ($validate_data[$i][$tree_attribute[$node_index]] < $tree_value[$node_index]) {
        $node_index = $node_index * 2;
      } else {
        $node_index = $node_index * 2 + 1;
      }
    }
    if ($tree_A[$node_index] > $tree_B[$node_index]) {
      #estimate 0
      if ($validate_data[$i][$num_attributes] == 0) {
        $num_correct = $num_correct + 1;
      }
    } else {
      #estimate 1
      if ($validate_data[$i][$num_attributes] == 1) {
        $num_correct = $num_correct + 1;
      }
    }
  }

  my $accuracy = $num_correct / $num_validate_lines;
  print "Accuracy on validation file $ARGV[2]: $accuracy\n";
  return $accuracy;

}

sub learningcurve {
  my $multiplier = @_[0];

  open (TRAIN_FILE, $ARGV[1]) or die "training file $ARGV[1] could not be read";
  my @train_file_lines = <TRAIN_FILE>;
  close (TRAIN_FILE);
  my $attribute_line = $train_file_lines[0];
  splice(@train_file_lines, 0, 1);
  @attribute_list = split(',', $attribute_line);
  $num_attributes = $#attribute_list;
  if ($num_attributes < 1) {
    die "Not enough attributes to train on";
  }

  my $samplesize = ceil($multiplier * $#train_file_lines);

  my $numbins = ceil($#train_file_lines / $samplesize);

  my $results;
  if ($numbins >= 5)
  {
    foreach my $i (0..4) 
    {
      train($i * $samplesize, $samplesize * $i + $samplesize);
      my $node_index = 1;
      tree_clean($node_index);
      $results += validate();
    }
    return $results / 5;
  }
  elsif ($numbins >= 2) 
  {
    foreach my $i (0..4) {
      train(ceil($i * $samplesize / 4), ceil($samplesize / 4 * $i + $samplesize));
      my $node_index = 1;
      tree_clean($node_index);
      $results += validate();
      
    }
    return $results / 5;
  }
  elsif ($numbins > 1)
  {
    foreach my $i (0..4) {
      train(ceil($i * $samplesize / 400), ceil($samplesize / 400 * $i + $samplesize));
      my $node_index = 1;
      tree_clean($node_index);
      $results += validate();
    }
    return $results / 5;
  }
  else
  {
    train(0,0);
    my $node_index = 1;
    tree_clean($node_index);
    return validate();
  }
  
}

#main

if ($#ARGV < 1) {
  print "Usage:\n  id3.pl h\n    provides usage help\n";
  print "  id3.pl t trainfile [pruning maxlevels]\n    creates a decision tree based on data in trainfile\n";
  print "    prints tree to stdout\n";
  print "  id3.pl e trainfile testfile [pruning maxlevels]\n";
  print "    creates a decision tree based on data in trainfile\n";
  print "    tests decision tree on data in testfile\n";
  print "    prints expected classes to stdout with other test data in csv format\n";
  print "  id3.pl v trainfile validatefile [pruning maxlevels]\n";
  print "    creates a decision tree based on data inf trainfile\n";
  print "    tests decision tree on  data in validatefile\n";
  print "    compares results from tree with results in validatefile\n";
  print "    prints accuracy\n";
  print "append pruning maxlevels to specify pruning as a boolean and maxlevels as an integer\n";
  print "Example: id3.pl t btrain.csv 1 12 for training with pruning and 12 levels\n";
} elsif ($ARGV[0] eq "t") {
  train(0,0);

  my $node_index = 1;
  print "Size of uncleaned tree: ", tree_count(1), "\n";
  tree_clean($node_index);
  tree_print($node_index);
  print "Size of cleaned tree: ", tree_count(1), "\n";
} elsif ($ARGV[0] eq "e") {
  train(0,0);
  if ($#ARGV > 2) {
    $pruning = $ARGV[3];
    $max_levels = $ARGV[4];
  }
  train();
  my $node_index = 1;
  tree_clean($node_index);
  test();
} elsif ($ARGV[0] eq "v") {
  train(0,0);
  my $node_index = 1;
  tree_clean($node_index);
  validate();
} elsif ($ARGV[0] eq "l") {
  print learningcurve($ARGV[3]) . "\n";
  
} else {
  print "Unable to parse command\n";
  print "For help:\n  id3.pl h\n";
}
