// SPDX-License-Identifier: MIT
#pragma once

/*
 * Immutable reference tables shared with native backends. The rotation tables
 * are row-major 128x128 matrices; centroid counts are 4, 8, 16, and 256.
 */
#ifdef __cplusplus
extern "C" {
#endif

const float * ggml_turbo_kv_rotation_forward();
const float * ggml_turbo_kv_rotation_inverse();
const float * ggml_turbo_kv_centroids2();
const float * ggml_turbo_kv_centroids3();
const float * ggml_turbo_kv_centroids4();
const float * ggml_turbo_kv_centroids8();

#ifdef __cplusplus
}
#endif
