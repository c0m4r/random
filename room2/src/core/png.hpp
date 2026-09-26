// room2 - minimal dependency-free PNG writer (used for screenshots and texture dumps).
// Writes 8-bit RGBA or 8-bit greyscale images using stored (uncompressed) deflate
// blocks, which every PNG reader accepts.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace room2::png {

// `rgba` must contain width*height*4 bytes, top row first.
bool writeRgba(const std::string& path, uint32_t width, uint32_t height, const uint8_t* rgba);
// `grey` must contain width*height bytes, top row first.
bool writeGrey(const std::string& path, uint32_t width, uint32_t height, const uint8_t* grey);

// Byte-level helpers, exposed for tests.
uint32_t crc32(const uint8_t* data, size_t length, uint32_t seed = 0);
uint32_t adler32(const uint8_t* data, size_t length);
std::vector<uint8_t> zlibStore(const uint8_t* data, size_t length);

}  // namespace room2::png
