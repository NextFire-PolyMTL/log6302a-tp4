<?php

function get_metadata($meta_type, $object_id, $meta_key = '', $single = false) {
  if ( ! $meta_type || ! is_numeric( $object_id ) ) {
    return false;
  }

  $object_id = absint( $object_id );
  if ( ! $object_id )
    return false;

  $meta_cache = wp_cache_get($object_id, $meta_type . '_meta');
  if ( !$meta_cache )
    $meta_cache = update_meta_cache( $meta_type, array( $object_id ) );

  if ( ! $meta_key )
    return $meta_cache;

  if ($simgle)
    return '';
  return array();
}
