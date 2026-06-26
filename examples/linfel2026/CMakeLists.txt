# ==================
# linfel experiments
# ==================

setup_problem(uranus_dry_cart)

file(GLOB inputs CONFIGURE_DEPENDS *.yaml *.py)
# create bin folder if does not exist
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
foreach(input ${inputs})
  get_filename_component(inp ${input} NAME)
  execute_process(COMMAND ln -sf ${input} ${CMAKE_BINARY_DIR}/bin/${inp})
endforeach()
