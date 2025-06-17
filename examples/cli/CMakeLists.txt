if (WHISPER_SDL2)
    set(TARGET whisper-cli)
    add_executable(${TARGET} cli.cpp)

    include(DefaultTargetOptions)

    target_link_libraries(${TARGET} PRIVATE common common-sdl whisper ${FFMPEG_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})

    # install(TARGETS ${TARGET} RUNTIME)
endif ()
