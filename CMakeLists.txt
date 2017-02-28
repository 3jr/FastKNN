#cmake_minimum_required(VERSION 3.2)
cmake_minimum_required(VERSION 2.8)

project(FastKNN)

find_package(OpenMP REQUIRED)

find_package(Boost REQUIRED COMPONENTS unit_test_framework)

include_directories(src)

# include_directories(${Boost_INCLUDE_DIRS})

set(CMAKE_CXX_FLAGS "-Wno-ignored-attributes -pedantic ${CMAKE_CXX_FLAGS}")

#However, the file(GLOB...) allows for wildcard additions:
file(GLOB SOURCES_COMMON
  "src/*.cpp"
  )
message("source files: " ${SOURCES_COMMON})

file(GLOB SOURCES_TESTS "src/tests/*.cpp")

# file(GLOB_RECURSE SOURCES "src/*.cpp")

set(SOURCES_TESTS ${SOURCES_COMMON} ${SOURCES_TESTS})

add_executable(boost_tests ${SOURCES_TESTS})
target_compile_options(boost_tests PUBLIC -std=c++14 -march=native -mtune=native)
target_compile_options(boost_tests PUBLIC ${HPX_APPLICATION_CFLAGS} ${OpenMP_CXX_FLAGS})
target_link_libraries(boost_tests PUBLIC ${HPX_APPLICATION_LDFLAGS} ${Boost_LIBRARIES} ${OpenMP_CXX_FLAGS})
INSTALL_TARGETS(/bin boost_tests)
