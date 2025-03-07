cmake_minimum_required(VERSION 3.10)
project(Helmsman)

# If you don't actually need these set(...) calls for Windows, you can remove them:
set(OpenCV_DIR "C:/opencv/build/x64/vc16/lib")  # Example only
set(OpenCV_BIN_DIR "C:/opencv/build/x64/vc16/bin")
set(OpenCV_DEBUG_DLL_FILENAME "opencv_world480d.dll")
set(OpenCV_RELEASE_DLL_FILENAME "opencv_world480.dll")

# Path to ONNX Runtime via Homebrew or similar
set(ONNXRUNTIME_DIR "/usr/local/opt/onnxruntime")

find_package(OpenCV REQUIRED)

# --- Configure your project files ---
include_directories(include)  # Include your header files directory

# Recursively collect all source files under 'src' directory
file(GLOB_RECURSE CURR_SOURCES src/*.cpp)

# Create the executable
add_executable(Helmsman ${CURR_SOURCES})

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

target_include_directories(Helmsman PRIVATE "${ONNXRUNTIME_DIR}/include")

target_compile_features(Helmsman PRIVATE cxx_std_17)

target_link_libraries(Helmsman PUBLIC ${OpenCV_LIBS})

# Find the ONNX Runtime library
if (APPLE)
    # For macOS, look for the dylib file
    find_library(ONNXRUNTIME_LIB onnxruntime PATHS "${ONNXRUNTIME_DIR}/lib" NO_DEFAULT_PATH)
    if (NOT ONNXRUNTIME_LIB)
        message(FATAL_ERROR "ONNX Runtime library (libonnxruntime.dylib) not found in ${ONNXRUNTIME_DIR}/lib")
    endif()
elseif (UNIX)
    # For Linux, look for the .so file
    find_library(ONNXRUNTIME_LIB onnxruntime PATHS "${ONNXRUNTIME_DIR}/lib" NO_DEFAULT_PATH)
    if (NOT ONNXRUNTIME_LIB)
        message(FATAL_ERROR "ONNX Runtime library (libonnxruntime.so) not found in ${ONNXRUNTIME_DIR}/lib")
    endif()
endif()

target_link_libraries(Helmsman PRIVATE ${ONNXRUNTIME_LIB})

if (WIN32)
    target_link_libraries(Helmsman "${ONNXRUNTIME_DIR}/lib/onnxruntime.lib")
    add_custom_command(TARGET Helmsman POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${ONNXRUNTIME_DIR}/lib/onnxruntime.dll"
            "$<TARGET_FILE_DIR:Helmsman>"
    )
    add_custom_command(TARGET Helmsman POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${OpenCV_BIN_DIR}/${OpenCV_DEBUG_DLL_FILENAME}"
            "$<TARGET_FILE_DIR:Helmsman>"
    )
    add_custom_command(TARGET Helmsman POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${OpenCV_BIN_DIR}/${OpenCV_RELEASE_DLL_FILENAME}"
            "$<TARGET_FILE_DIR:Helmsman>"
    )
endif()
