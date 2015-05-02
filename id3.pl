#!/usr/bin/perl
use feature ':5.10';
use strict;
use POSIX;
use List::Util qw(min max);

#tree data
my @tree_A;
my @tree_B;
my @tree_index;
my @tree_value;
my @tree_attribute;
my @tree_leaf;
my @attribute_list;

#other global data
my $num_train_lines = 0;
my $num_attributes = 0;
my @train_data;

#parameters
my $max_levels = 10;
my $pruning = 0;
my $majority_threshold = 0.9;
my $share_threshold = 0.005;

#calls tree_gen to generate tree
sub train() {

  #read in data, find number of attributes (should be at least 2)
  #assume first line is label line
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
    my $average = $sum / $num_train_lines;
    foreach my $i (0..$num_train_lines) {
      if($train_data[$i][$j] == "?") {
        $train_data[$i][$j] = $average;
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
      if ($local_entropy < $best_entropy) {
        $best_entropy = $local_entropy;
        $best_index = $i;
        $best_attribute = $j;
      }
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
    my @next_unused_array = grep {$_ != $best_attribute} @unused_attributes;
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

#create routine to clean tree leaves
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

sub classify {
  my ($index, $testline) = @_;

  my @testarr = split(",", $testline);
  #print "The current index is: " . $index . "\n";
  #print "The current attribute is: " . $tree_attribute[$index] . "\n";
  #print "The current tree value is: " . $tree_value[$index] . "\n";
  #print "The current test value is: " . $testarr[$tree_attribute[$index]] . "\n";

  if ($tree_leaf[$index])
  {
    if ($tree_A[$index] > $tree_B[$index])
    {
      return 0;
    }
    else
    {
      return 1;
    }
  }
  else
  {
    # check for missing attribute value
    if (@testarr[$tree_attribute[$index]] == "?")
    {
      # get tree_A and tree_B on left
      my $sum_left = $tree_A[$index * 2] + $tree_B[$index * 2];
      my $sum_right = $tree_A[$index * 2 + 1] + $tree_B[$index * 2 + 1];
      # go to left if left is better
      if ($sum_left > $sum_right)
      {
        classify($index * 2, $testline);
      }
      # go to right if right is better
      else
      {
        classify($index * 2 + 1, $testline);
      }
    }
    # compare attribute with that in the tree
    else
    {
      if ($tree_value[$index] > @testarr[$tree_attribute[$index]])
      {  
        classify($index * 2, $testline);
      }
      else
      {
        classify($index * 2 + 1, $testline);
      }
    }
  }
}

sub validate {
  my ($index, $testline) = @_;

  my @testarr = split(",", $testline);
  #print "The current index is: " . $index . "\n";
  #print "The current attribute is: " . $tree_attribute[$index] . "\n";
  #print "The current tree value is: " . $tree_value[$index] . "\n";
  #print "The current test value is: " . $testarr[$tree_attribute[$index]] . "\n";

  if ($tree_leaf[$index])
  {
    if ($tree_A[$index] > $tree_B[$index])
    {
      if (@testarr[$#testarr] == 0)
      {
        return 1;
      }
      else
      {
        return 0;
      }
    }
    else
    {
      if (@testarr[$#testarr] == 1)
      {
        return 1;
      }
      else
      {
        return 0;
      }
    }
  }
  else
  {
    # check for missing attribute value
    if (@testarr[$tree_attribute[$index]] == "?")
    {
      # get tree_A and tree_B on left
      my $sum_left = $tree_A[$index * 2] + $tree_B[$index * 2];
      my $sum_right = $tree_A[$index * 2 + 1] + $tree_B[$index * 2 + 1];
      # go to left if left is better
      if ($sum_left > $sum_right)
      {
        validate($index * 2, $testline);
      }
      # go to right if right is better
      else
      {
        validate($index * 2 + 1, $testline);
      }
    }
    # compare attribute with that in the tree
    else
    {
      if ($tree_value[$index] > @testarr[$tree_attribute[$index]])
      {  
        validate($index * 2, $testline);
      }
      else
      {
        validate($index * 2 + 1, $testline);
      }
    }
  }
}

#main
if ($#ARGV < 1) {
  print "Usage:\n  id3.pl h\n    provides usage help\n";
  print "  id3.pl t trainfile\n    creates a decision tree based on data in trainfile\n";
  print "    prints tree to stdout\n";
} elsif ($ARGV[0] eq "t") {
  train();
  my $node_index = 1;
  tree_clean($node_index);
  #tree_print($node_index);
} elsif ($ARGV[0] eq "v") {
  train();
  my $node_index = 1;
  tree_clean($node_index);
  #print tree_print($node_index);


  # test the tree on validation set
  open (TEST_FILE, $ARGV[1]) or die "test file $ARGV[1] could not be read";
  my @test_file_lines = <TEST_FILE>;
  close (TEST_FILE);
  my $attribute_line = @test_file_lines[0];
  splice(@test_file_lines, 0, 1);
  if ($num_attributes < 1) {
    die "Not enough attributes to test on";
  }

  my $i = 0;
  my @results = [];
  foreach my $testline (@test_file_lines) 
  {
    @results[$i] = validate(1, $testline);
    $i = $i + 1;
  }


  my $len = scalar(@results);
  print $len . "\n";
  my $numcorrect;
  my $numwrong;
  foreach my $result (@results)
  {
    if ($result == 1)
    {
      $numcorrect++;
    }
    else
    {
      $numwrong++;
    }
  }
  print "number correct: " . $numcorrect . "\n";
  print "number wrong: " . $numwrong . "\n";
  print "percent correct: " . $numcorrect/$len . "\n";

  #print classify(1, $test_file_lines[4]);
} else {
  print "Unable to parse command\n";
  print "For help:\n  id3.pl h\n";
}
