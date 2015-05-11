set(LIBRARY cryptopp)

include_directories(.)

# todo, subset
file(GLOB HEADERS "*.h")
file(GLOB SOURCE "*.cpp")

add_library(${LIBRARY} ${HEADERS} ${SOURCE})

set(CRYPTOPP_INCLUDE_DIRS "." "../" PARENT_SCOPE)
set(CRYPTOPP_LIBRARIES ${LIBRARY} PARENT_SCOPE)
set(CRYPTOPP_FOUND TRUE PARENT_SCOPE)
