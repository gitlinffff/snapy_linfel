# =================
# Examples Problems
# =================

setup_problem(shock)
setup_problem(straka)
setup_problem(bryan)
setup_problem(jupiter_evap_precip_1d)
setup_problem(jupiter_dry_mesh)
setup_problem(run_hydro)
setup_problem(shallow_xy)
setup_problem(shallow_splash)
add_subdirectory(linfel2026)

file(GLOB inputs CONFIGURE_DEPENDS *.yaml *.py)
# create bin folder if does not exist
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
foreach(input ${inputs})
  get_filename_component(inp ${input} NAME)
  execute_process(COMMAND ln -sf ${input} ${CMAKE_BINARY_DIR}/bin/${inp})
endforeach()
