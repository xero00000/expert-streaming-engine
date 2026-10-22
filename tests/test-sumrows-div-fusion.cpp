#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

static void require(bool value, const std::string & message) {
    if (!value) throw std::runtime_error(message);
}

static std::vector<float> get(ggml_tensor * tensor) {
    std::vector<float> result(ggml_nelements(tensor));
    require(tensor->nb[0] == sizeof(float), "packed elements");
    size_t index = 0;
    for (int64_t i3=0; i3<tensor->ne[3]; ++i3)
        for (int64_t i2=0; i2<tensor->ne[2]; ++i2)
            for (int64_t i1=0; i1<tensor->ne[1]; ++i1) {
                ggml_backend_tensor_get(tensor, result.data()+index,
                        i3*tensor->nb[3]+i2*tensor->nb[2]+i1*tensor->nb[1],
                        tensor->ne[0]*sizeof(float));
                index += tensor->ne[0];
            }
    return result;
}

static void exact(const std::vector<float> & actual, const std::vector<float> & expected,
        const std::string & label) {
    require(actual.size() == expected.size(), label+": length");
    for (size_t i=0; i<actual.size(); ++i) {
        if (std::isnan(actual[i]) && std::isnan(expected[i])) continue;
        if (std::memcmp(&actual[i], &expected[i], sizeof(float)) != 0) {
            char numbers[160];
            std::snprintf(numbers,sizeof(numbers)," actual=%a expected=%a",
                    double(actual[i]),double(expected[i]));
            throw std::runtime_error(label+": mismatch at "+std::to_string(i)+
                    numbers);
        }
    }
}

static void run_case(ggml_backend_t backend, int width, int dim, bool strided, int pattern) {
    ggml_context * ctx = ggml_init({1<<20, nullptr, true});
    require(ctx != nullptr, "metadata context");
    ggml_backend_buffer_t buffer = nullptr;
    try {
        const int64_t n1=3, n2=2, n3=2;
        const int64_t storage0=width+(strided?5:0), storage1=n1+(strided?2:0), storage2=n2+(strided?1:0);
        auto * backing=ggml_new_tensor_4d(ctx, GGML_TYPE_F32, storage0, storage1, storage2, n3);
        auto * input=strided?ggml_view_4d(ctx,backing,width,n1,n2,n3,
                backing->nb[1],backing->nb[2],backing->nb[3],0):backing;
        auto * sum=ggml_sum_rows_ext(ctx,input,dim);
        auto * output=ggml_div(ctx,input,sum);
        // A requested intermediate remains observable even when the pair is fused.
        ggml_set_output(sum);
        ggml_set_output(output);
        auto * graph=ggml_new_graph_custom(ctx,32,false);
        ggml_build_forward_expand(graph,output);
        buffer=ggml_backend_alloc_ctx_tensors(ctx,backend);
        require(buffer!=nullptr,"tensor buffers");
        std::vector<float> values(ggml_nelements(backing),-12345.0f);
        for(int64_t i3=0;i3<n3;++i3)for(int64_t i2=0;i2<n2;++i2)for(int64_t i1=0;i1<n1;++i1)
            for(int j=0;j<width;++j) {
                float value=0.01f*float(1+j)+0.001f*float(i1+3*i2+6*i3);
                if(pattern==1)value=-value;
                if(pattern==2)value=j==0?1.0e8f:j==width-1?-1.0e8f:0.1f;
                if(pattern==3)value=(j&1)?-1.0f:1.0f;
                if(pattern==4)value=1.0f;
                values[i3*storage2*storage1*storage0+i2*storage1*storage0+i1*storage0+j]=value;
            }
        ggml_backend_tensor_set(backing,values.data(),0,values.size()*sizeof(float));
        std::vector<float> sentinel(ggml_nelements(sum),-9876.0f);
        ggml_backend_tensor_set(sum,sentinel.data(),0,sentinel.size()*sizeof(float));
        require(ggml_backend_graph_compute(backend,graph)==GGML_STATUS_SUCCESS,"complete graph");
        ggml_backend_synchronize(backend);
        const auto fused=get(output),fused_sum=get(sum);
        int boundary=-1;
        for(int i=0;i<graph->n_nodes;++i)if(graph->nodes[i]==sum)boundary=i;
        require(boundary>=0,"sum graph node");
        ggml_backend_tensor_set(sum,sentinel.data(),0,sentinel.size()*sizeof(float));
        auto first=ggml_graph_view(graph,0,boundary+1);
        auto second=ggml_graph_view(graph,boundary+1,graph->n_nodes);
        require(ggml_backend_graph_compute(backend,&first)==GGML_STATUS_SUCCESS,"sum stage");
        require(ggml_backend_graph_compute(backend,&second)==GGML_STATUS_SUCCESS,"divide stage");
        ggml_backend_synchronize(backend);
        const std::string label="width="+std::to_string(width)+" dim="+std::to_string(dim)+
                " strided="+std::to_string(strided)+" pattern="+std::to_string(pattern);
        exact(fused,get(output),label+" output");
        exact(fused_sum,get(sum),label+" sum");
        exact(get(backing),values,label+" input and gaps");
        ggml_backend_buffer_free(buffer);buffer=nullptr;
        ggml_free(ctx);
    } catch(...) {
        if(buffer)ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
        throw;
    }
}

int main(int argc, char ** argv) {
    auto backend=ggml_backend_cpu_init();
    if(!backend)return 1;
    try {
        if(argc==6 && std::string(argv[1])=="--case") {
            ggml_backend_cpu_set_n_threads(backend,1);
            run_case(backend,std::stoi(argv[2]),std::stoi(argv[3]),std::stoi(argv[4])!=0,std::stoi(argv[5]));
            ggml_backend_free(backend);
            std::puts("PASS: selected complete/staged sum-divide control");
            return 0;
        }
        require(argc==1,"usage: test-sumrows-div-fusion [--case WIDTH DIM STRIDED PATTERN]");
        size_t cases=0;
        for(int threads:{1,2,8}) {
            ggml_backend_cpu_set_n_threads(backend,threads);
            for(int width:{10,32})for(int dim:{0,1,2,3})for(bool strided:{false,true})
                for(int pattern:{0,1,2,3,4}) {
                    run_case(backend,width,dim,strided,pattern);
                    ++cases;
                }
        }
        std::printf("PASS: %zu complete/staged sum-divide cases; outputs, sums, inputs and gaps exact\n",cases);
        ggml_backend_free(backend);
        return 0;
    } catch(const std::exception & error) {
        std::fprintf(stderr,"FAIL: %s\n",error.what());
        ggml_backend_free(backend);
        return 1;
    }
}
