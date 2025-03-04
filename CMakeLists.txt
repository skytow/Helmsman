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

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ------------------------------------------------------------------------------
# 1) Create a STATIC LIBRARY for your ONNX code:
#    (assuming these .cpp files hold AutoBackendOnnx, OnnxModelBase, etc.)
# ------------------------------------------------------------------------------
add_library(onnx_backend STATIC
        src/nn/autobackend.cpp
        src/nn/onnx_model_base.cpp
        src/utils/common.cpp
)
target_include_directories(onnx_backend
        PUBLIC
        "${CMAKE_SOURCE_DIR}/include"
        "${OpenCV_INCLUDE_DIRS}"
        "${ONNXRUNTIME_DIR}/include"    # so that onnxruntime_cxx_api.h is found
)
# If any code inside autobackend.cpp or onnx_model_base.cpp calls onnxruntime,
# you also need to link it here:
target_link_libraries(onnx_backend
        PUBLIC
        ${OpenCV_LIBS}
        # Potentially also link onnxruntime here if that code depends on it:
        # For example, on macOS:
        # "${ONNXRUNTIME_DIR}/lib/libonnxruntime.dylib"
)

# ------------------------------------------------------------------------------
# 2) Create a STATIC LIBRARY for your visualization utilities (viz_utils)
# ------------------------------------------------------------------------------
add_library(viz_utils STATIC
        src/viz_utils.cpp
)
target_include_directories(viz_utils
        PUBLIC
        "${CMAKE_SOURCE_DIR}/include"       # Where your .hpp files are
        "${OpenCV_INCLUDE_DIRS}"           # OpenCV includes
        "${ONNXRUNTIME_DIR}/include"       # In case viz_utils includes onnx headers
)
target_link_libraries(viz_utils
        PUBLIC
        ${OpenCV_LIBS}
        # Possibly link onnxruntime if your viz_utils code calls it
        # e.g. "${ONNXRUNTIME_DIR}/lib/libonnxruntime.dylib"
)

# ------------------------------------------------------------------------------
# 3) main_image executable
# ------------------------------------------------------------------------------
add_executable(main_image
        src/main_image.cpp
)
target_include_directories(main_image
        PRIVATE
        "${CMAKE_SOURCE_DIR}/include"
        "${OpenCV_INCLUDE_DIRS}"
        "${ONNXRUNTIME_DIR}/include"
)
# Link both libraries (viz_utils + onnx_backend) plus OpenCV + onnxruntime
target_link_libraries(main_image
        PRIVATE
        viz_utils
        onnx_backend
        ${OpenCV_LIBS}
)

# Now platform-specific linking for main_image
if(WIN32)
    target_link_libraries(main_image PRIVATE "${ONNXRUNTIME_DIR}/lib/onnxruntime.lib")

    add_custom_command(TARGET main_image POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${ONNXRUNTIME_DIR}/lib/onnxruntime.dll"
            "$<TARGET_FILE_DIR:main_image>"
    )
    add_custom_command(TARGET main_image POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${OpenCV_BIN_DIR}/${OpenCV_DEBUG_DLL_FILENAME}"
            "$<TARGET_FILE_DIR:main_image>"
    )
    add_custom_command(TARGET main_image POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${OpenCV_BIN_DIR}/${OpenCV_RELEASE_DLL_FILENAME}"
            "$<TARGET_FILE_DIR:main_image>"
    )
elseif(APPLE)
    target_link_libraries(main_image PRIVATE "${ONNXRUNTIME_DIR}/lib/libonnxruntime.dylib")
elseif(UNIX)
    target_link_libraries(main_image PRIVATE "${ONNXRUNTIME_DIR}/lib/libonnxruntime.so")
endif()

# ------------------------------------------------------------------------------
# 4) main_video executable
# ------------------------------------------------------------------------------
add_executable(main_video
        src/main_video.cpp
)
target_include_directories(main_video
        PRIVATE
        "${CMAKE_SOURCE_DIR}/include"
        "${OpenCV_INCLUDE_DIRS}"
        "${ONNXRUNTIME_DIR}/include"
)
target_link_libraries(main_video
        PRIVATE
        viz_utils
        onnx_backend
        ${OpenCV_LIBS}
)

# Platform-specific linking for main_video
if(WIN32)
    target_link_libraries(main_video PRIVATE "${ONNXRUNTIME_DIR}/lib/onnxruntime.lib")

    add_custom_command(TARGET main_video POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${ONNXRUNTIME_DIR}/lib/onnxruntime.dll"
            "$<TARGET_FILE_DIR:main_video>"
    )
    add_custom_command(TARGET main_video POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${OpenCV_BIN_DIR}/${OpenCV_DEBUG_DLL_FILENAME}"
            "$<TARGET_FILE_DIR:main_video>"
    )
    add_custom_command(TARGET main_video POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${OpenCV_BIN_DIR}/${OpenCV_RELEASE_DLL_FILENAME}"
            "$<TARGET_FILE_DIR:main_video>"
    )
elseif(APPLE)
    target_link_libraries(main_video PRIVATE "${ONNXRUNTIME_DIR}/lib/libonnxruntime.dylib")
elseif(UNIX)
    target_link_libraries(main_video PRIVATE "${ONNXRUNTIME_DIR}/lib/libonnxruntime.so")
endif()
