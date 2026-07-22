# test-philox: Philox RNG reference dumper.
add_executable(test-philox test-philox.cpp)
target_include_directories(test-philox PRIVATE ${CMAKE_SOURCE_DIR}/src)
if(NOT MSVC)
    target_link_libraries(test-philox PRIVATE m)
endif()

# test-model-store: exercise ModelStore load/evict policies against real GGUFs.
add_executable(test-model-store test-model-store.cpp)
target_link_libraries(test-model-store PRIVATE acestep-core)
link_ggml_backends(test-model-store)

# test-lm-prompt: byte for byte check of Phase 2 LM prompt builders
# against the Qwen3 chat template reference.
add_executable(test-lm-prompt test-lm-prompt.cpp)
target_link_libraries(test-lm-prompt PRIVATE acestep-core)
link_ggml_backends(test-lm-prompt)

# test-dit-resume: pause the flow matching loop mid-schedule and prove the
# resumed run reaches byte-identical latents.
add_executable(test-dit-resume test-dit-resume.cpp)
target_link_libraries(test-dit-resume PRIVATE acestep-core)
link_ggml_backends(test-dit-resume)
