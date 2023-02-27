
MJPG_STREAMER_PLUGIN_OPTION(cvfilter_cpp "OpenCV example filter")
MJPG_STREAMER_PLUGIN_COMPILE(cvfilter_cpp filter_cpp.cpp)

add_library(detector SHARED IMPORTED)
set_target_properties(detector PROPERTIES IMPORTED_LOCATION "/usr/lib/libdetector.so")
target_include_directories(cvfilter_cpp PRIVATE modules/ extra/)
target_link_libraries(cvfilter_cpp detector ${OpenCV_LIBS})