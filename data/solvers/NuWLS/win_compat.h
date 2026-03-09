// Windows compatibility shim for POSIX timing functions
// This file is injected via -include during compilation on Windows.
// It does NOT modify the original NuWLS source code.
#ifndef WIN_COMPAT_H
#define WIN_COMPAT_H

#ifdef _WIN32

// Provide sys/times.h equivalents
#include <windows.h>
#include <time.h>

struct tms {
    clock_t tms_utime;
    clock_t tms_stime;
    clock_t tms_cutime;
    clock_t tms_cstime;
};

static inline clock_t times(struct tms *buf) {
    FILETIME create, exit, kernel, user;
    if (GetProcessTimes(GetCurrentProcess(), &create, &exit, &kernel, &user)) {
        // Convert 100-nanosecond intervals to clock ticks (CLOCKS_PER_SEC)
        ULARGE_INTEGER u, k;
        u.LowPart = user.dwLowDateTime;
        u.HighPart = user.dwHighDateTime;
        k.LowPart = kernel.dwLowDateTime;
        k.HighPart = kernel.dwHighDateTime;
        buf->tms_utime = (clock_t)(u.QuadPart / (10000000ULL / CLOCKS_PER_SEC));
        buf->tms_stime = (clock_t)(k.QuadPart / (10000000ULL / CLOCKS_PER_SEC));
    } else {
        buf->tms_utime = clock();
        buf->tms_stime = 0;
    }
    buf->tms_cutime = 0;
    buf->tms_cstime = 0;
    return clock();
}

// sysconf(_SC_CLK_TCK) replacement
#ifndef _SC_CLK_TCK
#define _SC_CLK_TCK 2
#endif

static inline long sysconf(int name) {
    if (name == _SC_CLK_TCK) return CLOCKS_PER_SEC;
    return -1;
}

// Suppress the real sys/times.h and unistd.h includes
#define _SYS_TIMES_H
#define _UNISTD_H

// Provide getpid if needed
#include <process.h>

#endif // _WIN32
#endif // WIN_COMPAT_H
