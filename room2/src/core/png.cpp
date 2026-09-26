#include "core/png.hpp"

#include <cstdio>
#include <cstring>

#include "core/log.hpp"

namespace room2::png {
namespace {

uint32_t crcTableEntry(uint32_t n) {
    uint32_t c = n;
    for (int k = 0; k < 8; ++k) c = (c & 1) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
    return c;
}

const uint32_t* crcTable() {
    static uint32_t table[256];
    static bool initialised = false;
    if (!initialised) {
        for (uint32_t i = 0; i < 256; ++i) table[i] = crcTableEntry(i);
        initialised = true;
    }
    return table;
}

void pushBigEndian(std::vector<uint8_t>& out, uint32_t value) {
    out.push_back(static_cast<uint8_t>((value >> 24) & 0xFF));
    out.push_back(static_cast<uint8_t>((value >> 16) & 0xFF));
    out.push_back(static_cast<uint8_t>((value >> 8) & 0xFF));
    out.push_back(static_cast<uint8_t>(value & 0xFF));
}

void pushChunk(std::vector<uint8_t>& out, const char type[4], const std::vector<uint8_t>& payload) {
    pushBigEndian(out, static_cast<uint32_t>(payload.size()));
    const size_t crcStart = out.size();
    out.insert(out.end(), type, type + 4);
    out.insert(out.end(), payload.begin(), payload.end());
    const uint32_t crc = crc32(out.data() + crcStart, out.size() - crcStart);
    pushBigEndian(out, crc);
}

std::vector<uint8_t> filterRows(const uint8_t* data, uint32_t width, uint32_t height,
                                uint32_t bytesPerPixel) {
    const size_t stride = static_cast<size_t>(width) * bytesPerPixel;
    std::vector<uint8_t> filtered;
    filtered.reserve((stride + 1) * height);
    for (uint32_t y = 0; y < height; ++y) {
        // Filter type 0 (None) keeps this simple and still compresses acceptably when
        // the deflate stream is stored verbatim.
        filtered.push_back(0);
        filtered.insert(filtered.end(), data + static_cast<size_t>(y) * stride,
                        data + static_cast<size_t>(y) * stride + stride);
    }
    return filtered;
}

bool writePngInternal(const std::string& path, uint32_t width, uint32_t height,
                      const uint8_t* data, uint32_t bytesPerPixel, uint8_t colorType) {
    if (width == 0 || height == 0 || !data) {
        R2_ERROR("png::write: invalid image (", width, "x", height, ")");
        return false;
    }
    std::vector<uint8_t> filtered = filterRows(data, width, height, bytesPerPixel);
    std::vector<uint8_t> compressed = zlibStore(filtered.data(), filtered.size());

    std::vector<uint8_t> out;
    out.reserve(compressed.size() + 128);
    const uint8_t signature[8] = {0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A};
    out.insert(out.end(), signature, signature + 8);

    std::vector<uint8_t> ihdr;
    pushBigEndian(ihdr, width);
    pushBigEndian(ihdr, height);
    ihdr.push_back(8);            // bit depth
    ihdr.push_back(colorType);    // 6 = RGBA, 0 = grey
    ihdr.push_back(0);            // compression
    ihdr.push_back(0);            // filter
    ihdr.push_back(0);            // interlace
    pushChunk(out, "IHDR", ihdr);
    pushChunk(out, "IDAT", compressed);
    pushChunk(out, "IEND", {});

    std::FILE* file = std::fopen(path.c_str(), "wb");
    if (!file) {
        R2_ERROR("png::write: cannot open '", path, "' for writing");
        return false;
    }
    const size_t written = std::fwrite(out.data(), 1, out.size(), file);
    std::fclose(file);
    if (written != out.size()) {
        R2_ERROR("png::write: short write to '", path, "'");
        return false;
    }
    return true;
}

}  // namespace

uint32_t crc32(const uint8_t* data, size_t length, uint32_t seed) {
    const uint32_t* table = crcTable();
    uint32_t c = seed ^ 0xFFFFFFFFu;
    for (size_t i = 0; i < length; ++i) c = table[(c ^ data[i]) & 0xFF] ^ (c >> 8);
    return c ^ 0xFFFFFFFFu;
}

uint32_t adler32(const uint8_t* data, size_t length) {
    uint32_t a = 1, b = 0;
    const uint32_t mod = 65521u;
    for (size_t i = 0; i < length; ++i) {
        a = (a + data[i]) % mod;
        b = (b + a) % mod;
    }
    return (b << 16) | a;
}

std::vector<uint8_t> zlibStore(const uint8_t* data, size_t length) {
    std::vector<uint8_t> out;
    out.reserve(length + length / 65535 * 5 + 16);
    // zlib header: deflate, 32k window, no preset dictionary, fastest compression.
    out.push_back(0x78);
    out.push_back(0x01);

    size_t offset = 0;
    do {
        const size_t chunk = (length - offset) > 65535 ? 65535 : (length - offset);
        const bool last = (offset + chunk) >= length;
        out.push_back(static_cast<uint8_t>(last ? 1 : 0));
        out.push_back(static_cast<uint8_t>(chunk & 0xFF));
        out.push_back(static_cast<uint8_t>((chunk >> 8) & 0xFF));
        const uint16_t inv = static_cast<uint16_t>(~chunk);
        out.push_back(static_cast<uint8_t>(inv & 0xFF));
        out.push_back(static_cast<uint8_t>((inv >> 8) & 0xFF));
        out.insert(out.end(), data + offset, data + offset + chunk);
        offset += chunk;
    } while (offset < length);

    const uint32_t adler = adler32(data, length);
    pushBigEndian(out, adler);
    return out;
}

bool writeRgba(const std::string& path, uint32_t width, uint32_t height, const uint8_t* rgba) {
    return writePngInternal(path, width, height, rgba, 4, 6);
}

bool writeGrey(const std::string& path, uint32_t width, uint32_t height, const uint8_t* grey) {
    return writePngInternal(path, width, height, grey, 1, 0);
}

}  // namespace room2::png
