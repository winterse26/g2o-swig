if(POLICY CMP0078)
  cmake_policy(SET CMP0078 NEW)
endif()

if(POLICY CMP0086)
  cmake_policy(SET CMP0086 NEW)
endif()

find_package(SWIG REQUIRED)
find_package(Java REQUIRED)
find_package(JNI REQUIRED)

include(UseSWIG)

if(JNI_FOUND)
  message("JNI found: ${JNI_INCLUDE_DIRS}")
endif()

set_property(SOURCE odoslam.i PROPERTY CPLUSPLUS ON)
list(APPEND CMAKE_SWIG_FLAGS "-package;g2o.odoslam")

include_directories(${JNI_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

swig_add_library(odo_slam LANGUAGE java TYPE ${G2O_LIB_TYPE} SOURCES odoslam.i odoslam.cc)
add_dependencies(odo_slam graph_proto_dependency)
set_target_properties(odo_slam PROPERTIES OUTPUT_NAME ${LIB_PREFIX}odoslam)

set_property(TARGET odo_slam PROPERTY SWIG_INCLUDE_DIRECTORIES ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(odo_slam
    core
    solver_eigen
    solver_csparse
    types_slam3d
    types_slam3d_addons
    graph_proto)

if(UNIX)
  target_link_libraries(odo_slam ${CMAKE_DL_LIBS})
endif()

install(TARGETS odo_slam
  RUNTIME DESTINATION ${RUNTIME_DESTINATION}
  LIBRARY DESTINATION ${LIBRARY_DESTINATION}
  ARCHIVE DESTINATION ${ARCHIVE_DESTINATION}
  INCLUDES DESTINATION ${INCLUDES_DESTINATION})
