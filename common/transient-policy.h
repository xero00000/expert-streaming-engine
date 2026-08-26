#pragma once

// Explicit product policy for the mutually exclusive transient MTP and
// multimodal owners. A policy changes feature availability; byte capacity is
// derived from the registered module bounds rather than accepted as a raw
// user-facing target.
enum common_transient_policy {
    COMMON_TRANSIENT_POLICY_OFF,
    COMMON_TRANSIENT_POLICY_SHARED,
    COMMON_TRANSIENT_POLICY_MTP_ONLY,
    COMMON_TRANSIENT_POLICY_MULTIMODAL_ONLY,
};
