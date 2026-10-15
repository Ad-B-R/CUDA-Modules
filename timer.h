#ifndef _TIMER_H_
#define _TIMER_H_

#include <chrono>
#include <cstdio>

enum PrintColor { NONE, GREEN, DGREEN, CYAN };

typedef struct {
    std::chrono::steady_clock::time_point startTime;
    std::chrono::steady_clock::time_point endTime;
} Timer;

static void startTime(Timer* timer) {
    timer->startTime = std::chrono::steady_clock::now();
}

static void stopTime(Timer* timer) {
    timer->endTime = std::chrono::steady_clock::now();
}

static float elapsedMs(Timer timer) {
    return std::chrono::duration<float, std::milli>(timer.endTime - timer.startTime).count();
}

static void printElapsedTime(Timer timer, const char* s, enum PrintColor color = NONE) {
    float t = elapsedMs(timer);
    switch (color) {
        case GREEN:  printf("\033[1;32m"); break;
        case DGREEN: printf("\033[0;32m"); break;
        case CYAN:   printf("\033[1;36m"); break;
        default: break;
    }
    printf("%s: %f ms\n", s, t);
    if (color != NONE) printf("\033[0m");
}

#endif
