cmake_minimum_required (VERSION 3.1)
project(dde_bin_loader)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --std=c++11")

find_package (Python COMPONENTS Interpreter Development)

include_directories(${Python_INCLUDE_DIRS})
link_directories(${Python_LIBRARY_DIRS})

file(GLOB SRCS dde_bin_loader.cc)
set(LINK_LIBS ${Python_LIBRARIES})

add_library(dde_bin_loader SHARED ${SRCS})
target_link_libraries(dde_bin_loader ${LINK_LIBS})
if(MSVC)
set_target_properties(dde_bin_loader PROPERTIES SUFFIX ".pyd")
elseif(UNIX)
set_target_properties(dde_bin_loader PROPERTIES PREFIX "")
endif()

install(TARGETS dde_bin_loader DESTINATION ${CMAKE_SOURCE_DIR})
