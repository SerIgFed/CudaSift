find_package(TBB REQUIRED)

add_executable(cudasift_test mainSift.cpp geomFuncs.cpp)
target_link_libraries(cudasift_test cudasift ${OpenCV_LIBS} TBB::tbb)

install(
  TARGETS cudasift
  EXPORT Find${PROJECT_NAME}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

