#pragma once
#include "llama.h"
#include <memory>
#include <string>
#include <vector>

struct llama_model;
namespace deepseek_v41 {
// Experimental greedy DSpark executor. The target owns the shared embedding
// and head and must outlive this object. Only accepted target features enter
// the draft window; proposing tokens never mutates it.
class dspark {
public:
    dspark(llama_model & target, const std::string & path, int first_device = 0, int last_device = 1);
    ~dspark();
    const std::vector<int32_t> & target_layers() const;
    void ingest(const std::vector<float> & features, int position, int count);
    std::vector<llama_token> propose(llama_token known, int position);
private:
    struct impl;
    std::unique_ptr<impl> p;
};
}
