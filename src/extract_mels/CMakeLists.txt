cmake_minimum_required(VERSION 3.10)
project(extract_mels)

set(CMAKE_CXX_STANDARD 17)

# Dipendenze base
find_package(YAML-CPP REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(SNDFILE REQUIRED sndfile)
pkg_check_modules(SAMPLERATE REQUIRED samplerate)
find_package(nlohmann_json REQUIRED)
find_package(ZLIB REQUIRED)

# Executable
add_executable(extract_mels extract_mels.cpp)

# Include directories
target_include_directories(extract_mels PRIVATE
    ${SNDFILE_INCLUDE_DIRS}
    ${SAMPLERATE_INCLUDE_DIRS}
    /usr/local/include  # dove si trova cnpy.h
)

# Link libraries
target_link_libraries(extract_mels PRIVATE
    yaml-cpp
    ${SNDFILE_LIBRARIES}
    ${SAMPLERATE_LIBRARIES}
    nlohmann_json::nlohmann_json
    /usr/local/lib/libcnpy.a  # oppure .so se preferisci dinamico
    ZLIB::ZLIB
)

