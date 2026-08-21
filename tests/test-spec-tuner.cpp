#include "spec-tuner.h"

#include <cassert>
#include <iostream>

static void sample_arm(spec_tuner & tuner, int arm, double tps) {
    tuner.coords[0].current_idx = arm;
    tuner.accept_feedback(arm == 0 ? 0 : 1, arm, tps);
}

int main() {
    common_params_speculative params;
    params.n_max = 3;
    params.n_min = 0;

    spec_tuner losing;
    losing.init(COMMON_SPECULATIVE_TYPE_MTP, params, nullptr);
    assert(losing.has_target_only_arm());
    assert(losing.coords.size() == 1);
    assert(losing.coords[0].arms.size() == 4);
    assert((int) losing.coords[0].arms[0].value == 0);

    for (int arm = 0; arm < 4; ++arm) {
        for (int sample = 0; sample < 3; ++sample) {
            sample_arm(losing, arm, arm == 0 ? 100.0 : 70.0 - arm);
        }
    }
    assert(losing.coords[0].best_idx == 0);
    for (int arm = 1; arm < 4; ++arm) {
        assert(losing.target_only_quarantined[arm]);
    }

    common_params_speculative proposed = params;
    losing.propose(proposed);
    assert(proposed.n_max == 0);
    assert(losing.n_target_only_selections == 1);

    spec_tuner winning;
    winning.init(COMMON_SPECULATIVE_TYPE_MTP, params, nullptr);
    for (int arm = 0; arm < 4; ++arm) {
        for (int sample = 0; sample < 3; ++sample) {
            sample_arm(winning, arm, arm == 2 ? 130.0 : 100.0 - arm);
        }
    }
    assert(winning.coords[0].best_idx == 2);
    proposed = params;
    winning.propose(proposed);
    assert(proposed.n_max == 2);

    std::cout << "PASS: adaptive MTP depth includes a measured target-only arm\n";
    return 0;
}
