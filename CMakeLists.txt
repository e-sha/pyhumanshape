cmake_minimum_required(VERSION 2.8)

project(shapemodel)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_HOME_DIRECTORY}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_HOME_DIRECTORY}/bin)

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3 -Wall -march=native")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3 -Wall -march=native -D_G_HAVE_BOOL")

set(ADD_PATH "3dParty/humanshape/shapemodel/")

#find boost & build pyfacesdk
find_package(PythonLibs 2.7 REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS} ${ADD_PATH} ${ADD_PATH}/lib/include ${ADD_PATH}/lib/nr /usr/lib/python2.7/site-packages/numpy/core/include) 

FIND_PACKAGE(Boost REQUIRED )
FIND_PACKAGE(Boost COMPONENTS python REQUIRED)
set(Boost_USE_STATIC_LIBS ON)
include_directories( ${Boost_INCLUDE_DIR})

add_library(shapemodel SHARED src/humanshape.cpp ${ADD_PATH}NRBM.cpp ${ADD_PATH}paramMap.cpp ${ADD_PATH}CTMesh-30DOF.cpp ${ADD_PATH}NMath.cpp ${ADD_PATH}Show.cpp)
TARGET_LINK_LIBRARIES(shapemodel ${Boost_LIBRARIES})
set_target_properties(shapemodel PROPERTIES PREFIX "")
