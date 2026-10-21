#llama_add_compile_flags()

function(llama_build source)
    set(TEST_SOURCES ${source} ${ARGN})

    if (DEFINED LLAMA_TEST_NAME)
        set(TEST_TARGET ${LLAMA_TEST_NAME})
    else()
        get_filename_component(TEST_TARGET ${source} NAME_WE)
    endif()

    add_executable(${TEST_TARGET} ${TEST_SOURCES})
    target_link_libraries(${TEST_TARGET} PRIVATE common)
    install(TARGETS ${TEST_TARGET} RUNTIME)
endfunction()

function(llama_test target)
    include(CMakeParseArguments)
    set(options)
    set(oneValueArgs NAME LABEL WORKING_DIRECTORY)
    set(multiValueArgs ARGS)
    cmake_parse_arguments(LLAMA_TEST "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if (NOT DEFINED LLAMA_TEST_LABEL)
        set(LLAMA_TEST_LABEL "main")
    endif()
    if (NOT DEFINED LLAMA_TEST_WORKING_DIRECTORY)
        set(LLAMA_TEST_WORKING_DIRECTORY .)
    endif()
    if (DEFINED LLAMA_TEST_NAME)
        set(TEST_NAME ${LLAMA_TEST_NAME})
    else()
        set(TEST_NAME ${target})
    endif()

    set(TEST_TARGET ${target})

    add_test(
        NAME ${TEST_NAME}
        WORKING_DIRECTORY ${LLAMA_TEST_WORKING_DIRECTORY}
        COMMAND $<TARGET_FILE:${TEST_TARGET}>
        ${LLAMA_TEST_ARGS})

    set_property(TEST ${TEST_NAME} PROPERTY LABELS ${LLAMA_TEST_LABEL})
endfunction()

# Builds and runs a test source file.
# Optional args:
# - NAME: name of the executable & test target (defaults to the source file name without extension)
# - LABEL: label for the test (defaults to main)
# - ARGS: arguments to pass to the test executable
# - WORKING_DIRECTORY
function(llama_target_and_test source)
    include(CMakeParseArguments)
    set(options)
    set(oneValueArgs NAME LABEL WORKING_DIRECTORY)
    set(multiValueArgs ARGS)
    cmake_parse_arguments(LLAMA_TEST "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    set(TEST_SOURCES ${source} ${LLAMA_TEST_UNPARSED_ARGUMENTS} get-model.cpp)

    if (NOT DEFINED LLAMA_TEST_LABEL)
        set(LLAMA_TEST_LABEL "main")
    endif()
    if (NOT DEFINED LLAMA_TEST_WORKING_DIRECTORY)
        set(LLAMA_TEST_WORKING_DIRECTORY .)
    endif()
    if (DEFINED LLAMA_TEST_NAME)
        set(TEST_TARGET ${LLAMA_TEST_NAME})
    else()
        get_filename_component(TEST_TARGET ${source} NAME_WE)
    endif()

    add_executable(${TEST_TARGET} ${TEST_SOURCES})
    install(TARGETS ${TEST_TARGET} RUNTIME)
    target_link_libraries(${TEST_TARGET} PRIVATE common)
    add_test(
        NAME ${TEST_TARGET}
        WORKING_DIRECTORY ${LLAMA_TEST_WORKING_DIRECTORY}
        COMMAND $<TARGET_FILE:${TEST_TARGET}>
        ${LLAMA_TEST_ARGS})

    set_property(TEST ${TEST_TARGET} PROPERTY LABELS ${LLAMA_TEST_LABEL})
endfunction()

function(llama_build_and_test source)
    include(CMakeParseArguments)
    set(options)
    set(oneValueArgs NAME LABEL WORKING_DIRECTORY)
    set(multiValueArgs ARGS)
    cmake_parse_arguments(LLAMA_TEST "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    set(TEST_SOURCES ${source} ${LLAMA_TEST_UNPARSED_ARGUMENTS} get-model.cpp)

    if (NOT DEFINED LLAMA_TEST_LABEL)
        set(LLAMA_TEST_LABEL "main")
    endif()
    if (NOT DEFINED LLAMA_TEST_WORKING_DIRECTORY)
        set(LLAMA_TEST_WORKING_DIRECTORY .)
    endif()
    if (DEFINED LLAMA_TEST_NAME)
        set(TEST_TARGET ${LLAMA_TEST_NAME})
    else()
        get_filename_component(TEST_TARGET ${source} NAME_WE)
    endif()

    add_executable(${TEST_TARGET} ${TEST_SOURCES})
    if (LLAMA_TESTS_INSTALL)
        install(TARGETS ${TEST_TARGET} RUNTIME)
    endif()
    target_link_libraries(${TEST_TARGET} PRIVATE common)

    add_test(
        NAME ${TEST_TARGET}
        WORKING_DIRECTORY ${LLAMA_TEST_WORKING_DIRECTORY}
        COMMAND $<TARGET_FILE:${TEST_TARGET}>
        ${LLAMA_TEST_ARGS})

    set_property(TEST ${TEST_TARGET} PROPERTY LABELS ${LLAMA_TEST_LABEL})
endfunction()

# build test-tokenizer-0 target once and add many tests
add_executable(test-tokenizer-0 test-tokenizer-0.cpp)
target_link_libraries(test-tokenizer-0 PRIVATE common)
install(TARGETS test-tokenizer-0 RUNTIME)

llama_test(test-tokenizer-0 NAME test-tokenizer-0-bert-bge          ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-bert-bge.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-command-r         ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-command-r.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-deepseek-coder    ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-deepseek-coder.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-deepseek-llm      ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-deepseek-llm.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-falcon            ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-falcon.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-gpt-2             ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-gpt-2.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-llama-bpe         ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-llama-bpe.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-llama-spm         ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-llama-spm.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-mpt               ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-mpt.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-phi-3             ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-phi-3.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-qwen2             ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-qwen2.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-refact            ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-refact.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-starcoder         ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-starcoder.gguf)

if (LLAMA_LLGUIDANCE)
    llama_target_and_test(test-grammar-llguidance.cpp ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-llama-bpe.gguf)
endif ()

if (NOT WIN32)
    # these tests are disabled on Windows because they use internal functions not exported with LLAMA_API
    #llama_target_and_test(test-sampling.cpp)
    #llama_target_and_test(test-grammar-parser.cpp)
    #llama_target_and_test(test-grammar-integration.cpp)
    #llama_target_and_test(test-llama-grammar.cpp)
    #llama_target_and_test(test-chat.cpp)
    # TODO: disabled on loongarch64 because the ggml-ci node lacks Python 3.8
    #if (NOT ${CMAKE_SYSTEM_PROCESSOR} MATCHES "loongarch64")
    #    llama_target_and_test(test-json-schema-to-grammar.cpp   WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/..)
    #    target_include_directories(test-json-schema-to-grammar PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../examples/server)
    #endif()


    # build test-tokenizer-1-bpe target once and add many tests
    # add_executable(test-tokenizer-1-bpe test-tokenizer-1-bpe.cpp)
    # target_link_libraries(test-tokenizer-1-bpe PRIVATE common)
    # install(TARGETS test-tokenizer-1-bpe RUNTIME)

    # TODO: disabled due to slowness
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-aquila    ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-aquila.gguf)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-falcon    ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-falcon.gguf)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-gpt-2     ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-gpt-2.gguf)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-gpt-neox  ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-gpt-neox.gguf)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-llama-bpe ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-llama-bpe.gguf --ignore-merges)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-mpt       ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-mpt.gguf)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-refact    ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-refact.gguf)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-starcoder ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-starcoder.gguf)

    # build test-tokenizer-1-spm target once and add many tests
    # add_executable(test-tokenizer-1-spm test-tokenizer-1-spm.cpp)
    # target_link_libraries(test-tokenizer-1-spm PRIVATE common)
    # install(TARGETS test-tokenizer-1-spm RUNTIME)

    # llama_test(test-tokenizer-1-spm  NAME test-tokenizer-1-llama-spm ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-llama-spm.gguf)
    #llama_test(test-tokenizer-1-spm  NAME test-tokenizer-1-baichuan  ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-baichuan.gguf)

    # llama_target_and_test(test-double-float.cpp) # SLOW
endif()

#llama_build_and_test(test-chat-peg-parser.cpp peg-parser/simple-tokenize.cpp)
llama_build_and_test(test-jinja.cpp)
llama_test(test-jinja NAME test-jinja-py ARGS -py LABEL python)
llama_build_and_test(test-chat-auto-parser.cpp WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
llama_build_and_test(test-chat-template.cpp )
llama_build_and_test(test-json-partial.cpp )
#llama_build_and_test(test-log.cpp)
llama_build_and_test(
    test-peg-parser.cpp
    peg-parser/simple-tokenize.cpp
    peg-parser/test-basic.cpp
    peg-parser/test-gbnf-generation.cpp
    peg-parser/test-json-parser.cpp
    peg-parser/test-json-serialization.cpp
    peg-parser/test-python-dict-parser.cpp
    peg-parser/test-unicode.cpp
    peg-parser/tests.h
)
llama_build_and_test(test-regex-partial.cpp)

# llama_target_and_test(test-opt.cpp) # SLOW

llama_target_and_test(test-model-load-cancel.cpp  LABEL "model")
llama_target_and_test(test-autorelease.cpp        LABEL "model")
llama_build(test-qwen4exp-probe.cpp)
llama_test(test-qwen4exp-probe LABEL "model")
llama_build(test-hybrid-state-copy.cpp)
llama_test(test-hybrid-state-copy LABEL "model")
llama_build(test-qwen4exp-output-ledger.cpp)
llama_test(test-qwen4exp-output-ledger LABEL "model")
llama_build(test-qwen4exp-hybrid-retier.cpp)
llama_test(test-qwen4exp-hybrid-retier LABEL "model")
llama_build(test-model-loader-splits.cpp)
target_include_directories(test-model-loader-splits PRIVATE ${CMAKE_SOURCE_DIR}/src)
llama_test(test-model-loader-splits LABEL "model")
llama_build(test-speculative-sequence-isolation.cpp speculative-common-discard.cpp)
llama_build(test-bf16-stable-batch.cpp)
if (GGML_CUDA)
    find_package(CUDAToolkit REQUIRED)
    target_link_libraries(test-speculative-sequence-isolation PRIVATE CUDA::cuda_driver)
    target_link_libraries(test-bf16-stable-batch PRIVATE CUDA::cuda_driver)
    llama_build(test-cuda-nccl-lifecycle.cpp)
    llama_test(test-cuda-nccl-lifecycle LABEL "model")
    set_tests_properties(test-cuda-nccl-lifecycle PROPERTIES RUN_SERIAL TRUE SKIP_RETURN_CODE 77 TIMEOUT 120)
    llama_build(test-cuda-device-context.cpp)
    target_link_libraries(test-cuda-device-context PRIVATE CUDA::cudart CUDA::cuda_driver)
    llama_test(test-cuda-device-context LABEL "model")
    llama_build(test-context-device-selection.cpp test-native-device-planning.cpp)
    target_link_libraries(test-context-device-selection PRIVATE CUDA::cuda_driver)
    llama_test(test-context-device-selection LABEL "model")
    llama_test(test-context-device-selection NAME test-native-single-device-primary-contexts
        LABEL "model" ARGS --primary-contexts)
    set_tests_properties(test-native-single-device-primary-contexts PROPERTIES RUN_SERIAL TRUE)
endif()
llama_test(test-speculative-sequence-isolation LABEL "model")
llama_test(test-bf16-stable-batch LABEL "model")
llama_build(test-q2-0.cpp)
llama_test(test-q2-0 LABEL "model")
llama_build(test-gguf-unsupported-type.cpp)
llama_test(test-gguf-unsupported-type LABEL "model")
llama_build(test-deepseek-v41-engram.cpp)
llama_build(test-deepseek-v41-expert.cpp)
llama_build(test-deepseek-v41-ops.cpp)
llama_build(test-deepseek-v41-layout.cpp)
llama_build(test-deepseek-v41-index.cpp)
llama_build(test-deepseek-v41-cache.cpp)
llama_build(test-deepseek-v41-model.cpp)
llama_build(test-deepseek-v41-forward.cpp)
llama_build(test-deepseek-v41-moe.cpp)
llama_build(test-deepseek-v41-runtime.cpp)
llama_build(test-deepseek-v41-tokenizer.cpp)
llama_test(test-deepseek-v41-runtime LABEL "model" ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../docs/deepseek-v41/block-reference.json)
llama_test(test-deepseek-v41-moe LABEL "model" ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../docs/deepseek-v41/moe-reference.json)
llama_test(test-deepseek-v41-forward NAME test-deepseek-v41-token LABEL "model"
    ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../docs/deepseek-v41/block-reference.json cpu --whole-model)
llama_test(test-deepseek-v41-forward NAME test-deepseek-v41-block LABEL "model"
    ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../docs/deepseek-v41/block-reference.json cpu)
llama_test(test-deepseek-v41-forward LABEL "model"
    ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../docs/deepseek-v41/forward-reference.json cpu)
llama_test(test-deepseek-v41-forward NAME test-deepseek-v41-forward-bf16 LABEL "model"
    ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../docs/deepseek-v41/forward-bf16-reference.json cpu)
llama_test(test-deepseek-v41-model LABEL "model"
    ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../docs/deepseek-v41/config.json
         ${CMAKE_CURRENT_SOURCE_DIR}/../docs/deepseek-v41/mixedq2-gguf-inventory.json)
llama_test(test-deepseek-v41-cache LABEL "model"
    ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../docs/deepseek-v41/config.json
         ${CMAKE_CURRENT_SOURCE_DIR}/../docs/deepseek-v41/qat-reference.json)
llama_test(test-deepseek-v41-index LABEL "model"
    ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../docs/deepseek-v41/index-reference.json)
llama_test(test-deepseek-v41-ops NAME test-deepseek-v41-index-scores LABEL "model"
    ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../docs/deepseek-v41/index-reference.json cpu)
llama_test(test-deepseek-v41-layout LABEL "model"
    ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../docs/deepseek-v41/config.json)
llama_test(test-deepseek-v41-ops LABEL "model"
    ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../docs/deepseek-v41/ops-reference.json cpu)
option(LLAMA_CUDA_MEMCHECK_TESTS "Register CUDA memory checks (requires a visible GPU)" OFF)
if (LLAMA_CUDA_MEMCHECK_TESTS)
    if (NOT GGML_CUDA)
        message(FATAL_ERROR "LLAMA_CUDA_MEMCHECK_TESTS requires GGML_CUDA=ON")
    endif()
    find_program(LLAMA_COMPUTE_SANITIZER compute-sanitizer
        HINTS ${CUDAToolkit_BIN_DIR} REQUIRED)
    add_test(NAME test-cuda-shutdown-memcheck
        COMMAND ${LLAMA_COMPUTE_SANITIZER} --tool memcheck --leak-check full --error-exitcode 99
            $<TARGET_FILE:test-deepseek-v41-ops>
            ${CMAKE_CURRENT_SOURCE_DIR}/../docs/deepseek-v41/ops-reference.json 0)
    set_tests_properties(test-cuda-shutdown-memcheck PROPERTIES
        LABELS "model;cuda;memcheck" RUN_SERIAL TRUE TIMEOUT 300)
    add_test(NAME test-cuda-nccl-lifecycle-memcheck
        COMMAND ${LLAMA_COMPUTE_SANITIZER} --tool memcheck --leak-check full --error-exitcode 99
            $<TARGET_FILE:test-cuda-nccl-lifecycle>)
    set_tests_properties(test-cuda-nccl-lifecycle-memcheck PROPERTIES
        LABELS "model;cuda;memcheck" RUN_SERIAL TRUE SKIP_RETURN_CODE 77 TIMEOUT 300)
endif()
llama_test(test-deepseek-v41-ops NAME test-deepseek-v41-qat LABEL "model"
    ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../docs/deepseek-v41/qat-reference.json cpu)
llama_test(test-deepseek-v41-ops NAME test-deepseek-v41-attention LABEL "model"
    ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../docs/deepseek-v41/attention-reference.json cpu)
llama_test(test-deepseek-v41-engram LABEL "model"
    ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../docs/deepseek-v41)
llama_build(test-qwen4exp-ple.cpp)
llama_test(test-qwen4exp-ple LABEL "model"
    ARGS
        ${CMAKE_CURRENT_SOURCE_DIR}/../docs/qwen4exp/ple-reference-trace.json
        ${CMAKE_CURRENT_SOURCE_DIR}/../docs/qwen4exp/ple-iq4nl-row-fixture.json)
llama_build(test-qwen4exp-ple-storage.cpp)
llama_test(test-qwen4exp-ple-storage LABEL "model")
llama_build(test-qwen4exp-ple-cache.cpp)
llama_test(test-qwen4exp-ple-cache LABEL "model")
llama_build(test-graph-arena-budget.cpp)
llama_test(test-graph-arena-budget LABEL "model")
llama_build(test-prefill-backoff.cpp)
llama_test(test-prefill-backoff LABEL "model")
llama_build(test-decode-status.cpp)
llama_test(test-decode-status LABEL "model")
llama_build(test-copy-transpose-layout.cpp)
llama_test(test-copy-transpose-layout)
llama_build(test-cuda-pool-memory.cpp)
llama_test(test-cuda-pool-memory LABEL "model")
llama_build(test-shared-memory-budget.cpp)
llama_test(test-shared-memory-budget LABEL "model")

llama_build(test-handoff-allocation.cpp)
llama_test(test-handoff-allocation LABEL "model")

llama_build(test-qwen4exp-ple-provider-metadata.cpp)
llama_test(test-qwen4exp-ple-provider-metadata LABEL "model")

llama_build(test-qwen4exp-ple-row-store.cpp)
llama_test(test-qwen4exp-ple-row-store LABEL "model"
    ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../docs/qwen4exp/ple-iq4nl-row-fixture.json)
llama_build(test-qwen4exp-ple-page-store.cpp)
llama_test(test-qwen4exp-ple-page-store LABEL "model"
    ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../docs/qwen4exp/ple-iq4nl-row-fixture.json)
llama_build(test-qwen4exp-schema.cpp)
llama_test(test-qwen4exp-schema LABEL "model"
    ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../docs/qwen4exp/ud-iq1-s.manifest.json)
llama_build(test-qwen4exp-ops.cpp)
llama_build(test-qwen4exp-prefix-state.cpp)
if (GGML_CUDA)
    target_link_libraries(test-qwen4exp-prefix-state PRIVATE CUDA::cuda_driver)
endif()
llama_test(test-qwen4exp-prefix-state LABEL "model")
llama_build(test-qwen4exp-ple-journal.cpp)
if (GGML_CUDA)
    target_link_libraries(test-qwen4exp-ple-journal PRIVATE CUDA::cuda_driver)
endif()
llama_test(test-qwen4exp-ple-journal LABEL "model")
# Reuse the bundled SHA-256 implementation to identify bounded real GGUF payloads.
target_sources(test-qwen4exp-ops PRIVATE ../examples/gguf-hash/deps/sha256/sha256.c)
target_include_directories(test-qwen4exp-ops PRIVATE ../examples/gguf-hash/deps)
llama_test(test-qwen4exp-ops LABEL "model"
    ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../docs/qwen4exp/operator-reference-fixture.json)
# Tiny synthetic qwen4exp GGUF generated in-test: exercises the metadata path
# without the 58 GiB production quant and keeps the production gate closed.
llama_build(test-qwen4exp-fixture.cpp)
llama_test(test-qwen4exp-fixture LABEL "model")

llama_build(test-qwen4exp-cache-sizing.cpp)
llama_test(test-qwen4exp-cache-sizing LABEL "model")

llama_build(test-model-resource-bytes.cpp)
llama_test(test-model-resource-bytes LABEL "model")

llama_build(test-sumrows-div-fusion.cpp)
llama_test(test-sumrows-div-fusion LABEL "model")

llama_build(test-ssm-conv-boundary.cpp)
llama_test(test-ssm-conv-boundary LABEL "model")

# Requires caller-supplied GGUF shard paths; exercised manually against the
# maintainer's local artifact (see test-qwen4exp-artifact.cpp usage).
llama_build(test-qwen4exp-artifact.cpp)



# Phase 1 Turbo KV: internal type metadata and checked CPU dispatch.
llama_build(test-turbo-kv-core.cpp)
llama_test(test-turbo-kv-core)

llama_build(test-turbo-kv-policy.cpp)
llama_test(test-turbo-kv-policy)

# Phase 2 expert hierarchy: checked descriptors and bounded lease lifetime.
llama_build(test-expert-cache.cpp)
if (GGML_EXPERT_CHUNKING AND GGML_IQK_MUL_MAT)
    target_compile_definitions(test-expert-cache PRIVATE ESE_TEST_EXPERT_LEASE_CHUNKING)
endif()
llama_test(test-expert-cache)
if (GGML_CUDA)
    add_test(NAME test-model-quantized-cache-geometry COMMAND test-expert-cache --model-quant-geometry-only)
    set_tests_properties(test-model-quantized-cache-geometry PROPERTIES
        ENVIRONMENT "ESE_CUDA_STABLE_EXPERT_REDUCTION=1" RUN_SERIAL TRUE TIMEOUT 120)
endif()
llama_build(test-expert-cache-transaction.cpp)

# Phase 3 transient residency: bounded, failure-atomic module transactions.
llama_build(test-transient-module-manager.cpp)
llama_test(test-transient-module-manager)

llama_build(test-spec-tuner.cpp)
llama_test(test-spec-tuner)

llama_build(test-speculative-mtp-transaction.cpp)
llama_test(test-speculative-mtp-transaction)

llama_build(test-resource-planner.cpp)
llama_test(test-resource-planner)

llama_build(test-kda-delta-net.cpp)
llama_test(test-kda-delta-net)

llama_build(test-delta-net-layout.cpp)
llama_test(test-delta-net-layout)

llama_build(test-mapped-draft.cpp)
llama_test(test-mapped-draft)

# Requires a caller-supplied GGUF model; exercised by the Phase 1 validation script.
llama_build(test-turbo-kv-lifecycle.cpp)
llama_build(test-turbo-kv-quality.cpp)
llama_build(test-turbo-kv-recurrent.cpp)

if (GGML_CUDA)
    llama_build(test-turbo-kv-cuda.cpp)
    llama_test(test-turbo-kv-cuda)

    # ggml enables CUDA in its own directory; initialize it in this scope too.
    enable_language(CUDA)
    add_executable(test-iq3-cuda-dot test-iq3-cuda-dot.cu)
    target_include_directories(test-iq3-cuda-dot PRIVATE ../ggml/src ../ggml/src/ggml-cuda)
    target_compile_definitions(test-iq3-cuda-dot PRIVATE
        GGML_CUDA_FUSION=${GGML_CUDA_FUSION}
        GGML_CUDA_MIN_BATCH_OFFLOAD=${GGML_CUDA_MIN_BATCH_OFFLOAD})
    target_compile_options(test-iq3-cuda-dot PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>)
    target_link_libraries(test-iq3-cuda-dot PRIVATE ggml CUDA::cudart CUDA::cuda_driver)
    set_target_properties(test-iq3-cuda-dot PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
    llama_test(test-iq3-cuda-dot)
    set_tests_properties(test-iq3-cuda-dot PROPERTIES SKIP_RETURN_CODE 77)
endif()

llama_build(test-get-rows-bf16.cpp)
llama_build(test-expert-f32.cpp)
llama_build(test-expert-iq-formats.cpp)
target_include_directories(test-expert-iq-formats PRIVATE ../ggml/src)
llama_test(test-expert-iq-formats)
llama_build(test-shared-f32.cpp)
llama_build(test-q6-direct.cpp)
if (GGML_CUDA)
    target_link_libraries(test-shared-f32 PRIVATE CUDA::cuda_driver)
    target_link_libraries(test-q6-direct PRIVATE CUDA::cuda_driver)
endif()
target_include_directories(test-expert-f32 PRIVATE ../ggml/src)
if (GGML_CUDA)
    target_link_libraries(test-expert-f32 PRIVATE CUDA::cuda_driver)
    llama_build(test-expert-f32-graph.cpp)
    target_include_directories(test-expert-f32-graph PRIVATE ../ggml/src)
    target_link_libraries(test-expert-f32-graph PRIVATE CUDA::cuda_driver)
    llama_build(test-expert-f32-cache.cpp)
    target_include_directories(test-expert-f32-cache PRIVATE ../ggml/src)
    target_link_libraries(test-expert-f32-cache PRIVATE CUDA::cuda_driver)
endif()
llama_test(test-get-rows-bf16)
llama_build(test-bf16-head.cpp)
llama_test(test-bf16-head)
llama_build(test-scalar-row-projection.cpp)
if (GGML_CUDA)
    target_link_libraries(test-scalar-row-projection PRIVATE CUDA::cuda_driver)
endif()
llama_test(test-scalar-row-projection LABEL "model")
llama_build(test-rms-layout.cpp)
if (GGML_CUDA)
    target_link_libraries(test-rms-layout PRIVATE CUDA::cuda_driver)
endif()
llama_test(test-rms-layout LABEL "model")
llama_build(test-attention-precision.cpp)
if (GGML_CUDA)
    target_link_libraries(test-attention-precision PRIVATE CUDA::cuda_driver)
endif()
llama_test(test-attention-precision LABEL "model")
llama_build(test-attention-batch.cpp)
if (GGML_CUDA)
    target_link_libraries(test-attention-batch PRIVATE CUDA::cuda_driver)
endif()
llama_test(test-attention-batch LABEL "model")
set_tests_properties(test-attention-batch PROPERTIES ENVIRONMENT "ESE_CUDA_ATTENTION_STABLE_BATCH=1")
llama_build(test-qwen4exp-cache-order.cpp)
if (GGML_CUDA)
    target_link_libraries(test-qwen4exp-cache-order PRIVATE CUDA::cuda_driver)
endif()
llama_test(test-qwen4exp-cache-order LABEL "model")
llama_build(test-qwen4exp-attention-layout.cpp)
llama_test(test-qwen4exp-attention-layout LABEL "model")
llama_build(test-cuda-graph-layout.cpp)
if (GGML_CUDA)
    target_link_libraries(test-cuda-graph-layout PRIVATE CUDA::cuda_driver)
endif()
llama_test(test-cuda-graph-layout LABEL "model")
llama_build(test-qwen4exp-qsa-selection.cpp)
if (GGML_CUDA)
    target_link_libraries(test-qwen4exp-qsa-selection PRIVATE CUDA::cuda_driver)
endif()
llama_test(test-qwen4exp-qsa-selection LABEL "model")
llama_build(test-fused-dense-batch.cpp)
if (GGML_CUDA)
    target_link_libraries(test-fused-dense-batch PRIVATE CUDA::cuda_driver)
endif()
llama_test(test-fused-dense-batch LABEL "model")
set_tests_properties(test-fused-dense-batch PROPERTIES ENVIRONMENT "ESE_CUDA_FUSED_BATCH_VECTOR=1")
llama_build(test-explicit-projection.cpp)
llama_test(test-explicit-projection)
if (GGML_CUDA)
    set_tests_properties(test-explicit-projection PROPERTIES ENVIRONMENT "ESE_CUDA_EXPLICIT_AUTOTUNE=1;CUDA_DEVICE_ORDER=PCI_BUS_ID")
endif()

llama_build(test-iq4-xs-cpu.cpp)
target_include_directories(test-iq4-xs-cpu PRIVATE ../ggml/src)
llama_test(test-iq4-xs-cpu)

# dummy executable - not installed
get_filename_component(TEST_TARGET test-c.c NAME_WE)
add_executable(${TEST_TARGET} test-c.c)
target_link_libraries(${TEST_TARGET} PRIVATE llama)
