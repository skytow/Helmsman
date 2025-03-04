cmake_minimum_required(VERSION 3.10)
project(Helmsman)

# --- Update these paths to match your installation ---
# For example, on Windows you might have:
set(OpenCV_DIR "C:/opencv/build/x64/vc16/lib")           # OpenCV lib root
set(OpenCV_BIN_DIR "C:/opencv/build/x64/vc16/bin")           # OpenCV bin root
set(OpenCV_DEBUG_DLL_FILENAME "opencv_world480d.dll")                # Debug DLL filename
set(OpenCV_RELEASE_DLL_FILENAME "opencv_world480.dll")               # Release DLL filename

# For ONNX Runtime installed via Homebrew on macOS:
set(ONNXRUNTIME_DIR "/usr/local/opt/onnxruntime")

# Find OpenCV package
find_package(OpenCV REQUIRED)

# Include your project's header files directory
include_directories(${CMAKE_SOURCE_DIR}/include)

# Recursively collect all source files in src/
file(GLOB_RECURSE CURR_SOURCES ${CMAKE_SOURCE_DIR}/src/*.cpp)

# Create the executable
add_executable(Helmsman ${CURR_SOURCES})

# Set C++ standard to C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
target_compile_features(Helmsman PRIVATE cxx_std_17)

# Include ONNX Runtime headers (pointing to the onnxruntime subfolder)
target_include_directories(Helmsman PRIVATE "${ONNXRUNTIME_DIR}/include/onnxruntime")

# Link against OpenCV libraries
target_link_libraries(Helmsman ${OpenCV_LIBS})

# Platform-specific settings:
if(WIN32)
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
elseif(APPLE)
    # On macOS, use the .dylib extension
    target_link_libraries(Helmsman "${ONNXRUNTIME_DIR}/lib/libonnxruntime.dylib")
elseif(UNIX)
    target_link_libraries(Helmsman "${ONNXRUNTIME_DIR}/lib/libonnxruntime.so")
endif()