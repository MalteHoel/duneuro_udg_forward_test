add_executable("duneuro_udg_forward_test" duneuro_udg_forward_test.cc)
target_link_dune_default_libraries("duneuro_udg_forward_test")

# check if path to the root directory of simbiosphere was defined
if (NOT DEFINED SIMBIOSPHERE_ROOT)
	message(FATAL_ERROR "path to root directory of simbiosphere not defined as a cmake variable")
endif()

# include and link against simbiosphere
set_property(TARGET "duneuro_udg_forward_test" APPEND PROPERTY INCLUDE_DIRECTORIES ${SIMBIOSPHERE_ROOT}/src/include)
target_link_libraries("duneuro_udg_forward_test" ${SIMBIOSPHERE_ROOT}/build/src/libsimbiosphere.so)
execute_process(COMMAND ${CMAKE_COMMAND} "-E" "create_symlink" "${CMAKE_CURRENT_SOURCE_DIR}/input/udg_configs.ini" "${CMAKE_CURRENT_BINARY_DIR}/udg_configs.ini")
execute_process(COMMAND ${CMAKE_COMMAND} "-E" "create_symlink" "${CMAKE_CURRENT_SOURCE_DIR}/input/conductivities.txt" "${CMAKE_CURRENT_BINARY_DIR}/conductivities.txt")
execute_process(COMMAND ${CMAKE_COMMAND} "-E" "create_symlink" "${CMAKE_CURRENT_SOURCE_DIR}/input/electrodes.txt" "${CMAKE_CURRENT_BINARY_DIR}/electrodes.txt")
execute_process(COMMAND ${CMAKE_COMMAND} "-E" "create_symlink" "${CMAKE_CURRENT_SOURCE_DIR}/input/dipole.txt" "${CMAKE_CURRENT_BINARY_DIR}/dipole.txt")
