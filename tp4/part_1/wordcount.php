<?php

define("YES", 1);
define("NO", 0);


$inword = NO;
$nl = 0;
$nw = 0;
$nc = 0;
$ni = 0;

$fp = fopen('file.txt', 'r');
$c = fgetc($fp);
while ( $c !== false ) {
  $nc = $nc + 1;
  if ( $c == "\n" )
    $nl = $nl + 1;
  if ( $c != " " )
    if ( $c != "\n" )
      if ( $c != "\t" )
        $ni = $ni +1;
      else
        $inword = NO;
    else
      $inword = NO;
  else
    $inword = NO;
  if (( $c != " " ) && ( $c != "\n" ) && ( $c != "\t" ))
    if( $inword == NO ) {
      $inword = YES;
      $nw = $nw + 1;
    }
  $c = fgetc($fp);
}

echo $nl . "\n";
echo $nw . "\n";
echo $nc . "\n";
echo $ni . "\n";


