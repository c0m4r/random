#include "core/log.hpp"

#include <cstdlib>
#include <ctime>
#include <mutex>
#include <vector>

namespace room2 {
namespace {

LogLevel g_level = LogLevel::Info;
std::mutex g_mutex;
bool g_color = false;

const char* levelName(LogLevel l) {
    switch (l) {
        case LogLevel::Trace: return "TRACE";
        case LogLevel::Debug: return "DEBUG";
        case LogLevel::Info: return "INFO ";
        case LogLevel::Warn: return "WARN ";
        case LogLevel::Error: return "ERROR";
        case LogLevel::Fatal: return "FATAL";
    }
    return "?????";
}

const char* levelColor(LogLevel l) {
    switch (l) {
        case LogLevel::Trace: return "\x1b[90m";
        case LogLevel::Debug: return "\x1b[36m";
        case LogLevel::Info: return "\x1b[32m";
        case LogLevel::Warn: return "\x1b[33m";
        case LogLevel::Error: return "\x1b[31m";
        case LogLevel::Fatal: return "\x1b[1;31m";
    }
    return "";
}

const char* baseName(const char* path) {
    const char* b = path;
    for (const char* p = path; *p; ++p)
        if (*p == '/' || *p == '\\') b = p + 1;
    return b;
}

}  // namespace

void logSetLevel(LogLevel level) { g_level = level; }
LogLevel logGetLevel() { return g_level; }

void logWrite(LogLevel level, const char* file, int line, const std::string& msg) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_color) {
        g_color = std::getenv("NO_COLOR") == nullptr;
    }
    std::FILE* out = (level >= LogLevel::Warn) ? stderr : stdout;
    if (g_color)
        std::fprintf(out, "%s[%s]\x1b[0m %s:%d  %s\n", levelColor(level), levelName(level),
                     baseName(file), line, msg.c_str());
    else
        std::fprintf(out, "[%s] %s:%d  %s\n", levelName(level), baseName(file), line, msg.c_str());
    std::fflush(out);
}

void logFlush() {
    std::fflush(stdout);
    std::fflush(stderr);
}

}  // namespace room2
