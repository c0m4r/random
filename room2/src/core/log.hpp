// room2 - minimal logging.
#pragma once

#include <cstdio>
#include <string>
#include <sstream>

namespace room2 {

enum class LogLevel { Trace = 0, Debug, Info, Warn, Error, Fatal };

void logSetLevel(LogLevel level);
LogLevel logGetLevel();
void logWrite(LogLevel level, const char* file, int line, const std::string& msg);
// Flush any buffered output (used before a crash/abort).
void logFlush();

namespace detail {
inline void logBuild(std::ostringstream&) {}
template <typename T, typename... Rest>
void logBuild(std::ostringstream& os, T&& first, Rest&&... rest) {
    os << std::forward<T>(first);
    logBuild(os, std::forward<Rest>(rest)...);
}
}  // namespace detail

template <typename... Args>
void logMessage(LogLevel level, const char* file, int line, Args&&... args) {
    if (static_cast<int>(level) < static_cast<int>(logGetLevel())) return;
    std::ostringstream os;
    detail::logBuild(os, std::forward<Args>(args)...);
    logWrite(level, file, line, os.str());
}

}  // namespace room2

#define R2_TRACE(...) ::room2::logMessage(::room2::LogLevel::Trace, __FILE__, __LINE__, __VA_ARGS__)
#define R2_DEBUG(...) ::room2::logMessage(::room2::LogLevel::Debug, __FILE__, __LINE__, __VA_ARGS__)
#define R2_INFO(...)  ::room2::logMessage(::room2::LogLevel::Info,  __FILE__, __LINE__, __VA_ARGS__)
#define R2_WARN(...)  ::room2::logMessage(::room2::LogLevel::Warn,  __FILE__, __LINE__, __VA_ARGS__)
#define R2_ERROR(...) ::room2::logMessage(::room2::LogLevel::Error, __FILE__, __LINE__, __VA_ARGS__)
#define R2_FATAL(...) do { ::room2::logMessage(::room2::LogLevel::Fatal, __FILE__, __LINE__, __VA_ARGS__); ::room2::logFlush(); } while (0)
