add_library(${PROJECT_NAME}_backend
        vertex.cc
        edge.cc
        problem.cc
        )

target_link_libraries( ${PROJECT_NAME}_backend
        ${THIRD_PARTY_LIBS}
        )
