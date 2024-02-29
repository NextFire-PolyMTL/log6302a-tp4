<?php

function read_file($filename) {
  $fp = fopen($filename,"r");
  $out = '';

  $line=fgets($fp);
  while($line != false) {
    $out = $out . $line;
    $line=fgets($fp);
  }

  fclose($fp);
  $fp = null;
  return $out;
}
