cmake_minimum_required(VERSION 3.10)
project(extract_mels)

set(CMAKE_CXX_STANDARD 17)

# Dipendenze base
find_package(YAML-CPP REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(AVFORMAT REQUIRED libavformat)
pkg_check_modules(AVCODEC REQUIRED libavcodec)
pkg_check_modules(AVUTIL REQUIRED libavutil)
pkg_check_modules(SWRESAMPLE REQUIRED libswresample)
find_package(nlohmann_json REQUIRED)
find_package(ZLIB REQUIRED)

# Executable
add_executable(extract_mels extract_mels.cpp)

# Include directories
target_include_directories(extract_mels PRIVATE
    ${AVFORMAT_INCLUDE_DIRS}
    ${AVCODEC_INCLUDE_DIRS}
    ${AVUTIL_INCLUDE_DIRS}
    ${SWRESAMPLE_INCLUDE_DIRS}
    /usr/local/include  # dove si trova cnpy.h
)

# Link libraries
target_link_libraries(extract_mels PRIVATE
    yaml-cpp
    ${AVFORMAT_LIBRARIES}
    ${AVCODEC_LIBRARIES}
    ${AVUTIL_LIBRARIES}
    ${SWRESAMPLE_LIBRARIES}
    nlohmann_json::nlohmann_json
    /usr/local/lib/libcnpy.a  # oppure .so se preferisci dinamico
    ZLIB::ZLIB
)

