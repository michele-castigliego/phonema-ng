cmake_minimum_required(VERSION 3.10)
project(extract_mels)

set(CMAKE_CXX_STANDARD 17)

find_package(YAML-CPP REQUIRED)
find_package(PkgConfig REQUIRED)

pkg_check_modules(SNDFILE REQUIRED sndfile)
pkg_check_modules(SAMPLERATE REQUIRED samplerate)

add_executable(extract_mels extract_mels.cpp)

target_include_directories(extract_mels PRIVATE ${SNDFILE_INCLUDE_DIRS} ${SAMPLERATE_INCLUDE_DIRS})

target_link_libraries(extract_mels PRIVATE yaml-cpp ${SNDFILE_LIBRARIES} ${SAMPLERATE_LIBRARIES})

# Optional: nlohmann/json and cnpy as header-only
find_package(nlohmann_json REQUIRED)
find_package(cnpy REQUIRED)

target_link_libraries(extract_mels PRIVATE nlohmann_json::nlohmann_json cnpy)

