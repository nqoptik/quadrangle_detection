cmake_minimum_required(VERSION 3.1)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(card_detection)

find_package(OpenCV REQUIRED)

add_executable(card_detection src/card_detection.cpp)
target_link_libraries(card_detection ${OpenCV_LIBS})
