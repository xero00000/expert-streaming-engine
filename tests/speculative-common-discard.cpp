#include "../common/speculative.h"

bool test_common_checkpoint_discard(llama_context * ctx, llama_seq_id seq_id, int mode);
bool test_common_checkpoint_discard(llama_context * ctx, llama_seq_id seq_id, int mode) {
    common_speculative_checkpoint owner;
    owner.valid = true;
    owner.seq_id = seq_id;
    owner.mode = mode;
    common_speculative_checkpoint_discard(owner, ctx);
    return !owner.valid && owner.seq_id == -1;
}
