find_package(OpenMP REQUIRED)


set(TENSORFLOW_DIR ../third_party/tensorflow)
# link dynamic tensorflow lib.
include_directories( 
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${TENSORFLOW_DIR}/include
  ${TENSORFLOW_DIR}/bazel_include
  ${TENSORFLOW_DIR}/include/nsync/public
  ${TENSORFLOW_DIR}/include/com_google_protobuf/src
) 
link_directories(${TENSORFLOW_DIR}/bin)

add_library (deep_cfr_base OBJECT
  device_manager.h
  vpevaluator.h
  vpevaluator.cc
  vpnet.h
  vpnet.cc
  universal_poker_exploitability.h
  universal_poker_exploitability.cc
)
target_include_directories (deep_cfr_base PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(deep_cfr_base PUBLIC tensorflow_cc OpenMP::OpenMP_CXX)

# SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--allow-multiple-definition -Wl,--whole-archive")

add_executable(run_ach ach.cc run_deep_cfr.cc  deep_cfr.h 
    ach_solver.cc  ach_solver.h
    ${OPEN_SPIEL_OBJECTS})
target_link_libraries(run_ach deep_cfr_base)

add_executable(run_a2c a2c.cc run_deep_cfr.cc  deep_cfr.h 
    ach_solver.cc  ach_solver.h
    ${OPEN_SPIEL_OBJECTS})
target_link_libraries(run_a2c deep_cfr_base)

add_executable(run_neurd neurd.cc run_deep_cfr.cc  deep_cfr.h 
    neurd_solver.cc  neurd_solver.h
    ${OPEN_SPIEL_OBJECTS})
target_link_libraries(run_neurd deep_cfr_base)

add_executable(run_rpg rpg.cc run_deep_cfr.cc  deep_cfr.h 
    neurd_solver.cc  neurd_solver.h
    ${OPEN_SPIEL_OBJECTS})
target_link_libraries(run_rpg deep_cfr_base)
