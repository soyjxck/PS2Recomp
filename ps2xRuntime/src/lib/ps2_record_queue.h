#pragma once

#include <cstdint>
#include <cstring>
#include <vector>

// What one thread sends another in order: records of a kind, a number and
// some bytes, appended to one buffer that the receiver takes whole
// (std::swap with an empty one of its own, which it has finished with). The
// two buffers keep the size they have grown to, so once they have, a record
// costs its copy and nothing else: no allocation by the sender to be freed by
// the receiver, thousands of times a frame.
//
// Neither side is protected here; the owner holds its lock around append()
// and the swap.
struct RecordQueue
{
    struct Record
    {
        uint32_t kind = 0;
        uint64_t value = 0;
        const uint8_t *data = nullptr; // 16-byte aligned, as a buffer of its own was
        uint32_t size = 0;
    };

    std::vector<uint8_t> bytes;
    uint64_t count = 0;

    bool empty() const { return count == 0u; }

    void clear()
    {
        bytes.clear();
        count = 0u;
    }

    void append(uint32_t kind, uint64_t value, const uint8_t *data, uint32_t size)
    {
        // (Inserted, not resized into: the bytes are written once.)
        const Header header{size, kind, value};
        const uint8_t *const head = reinterpret_cast<const uint8_t *>(&header);
        bytes.insert(bytes.end(), head, head + sizeof(header));
        if (size != 0u)
            bytes.insert(bytes.end(), data, data + size);
        bytes.resize((bytes.size() + 15u) & ~static_cast<size_t>(15u));
        ++count;
    }

    // The records in the order they were appended: next() until it is false.
    struct Reader
    {
        const std::vector<uint8_t> &bytes;
        size_t at = 0;

        bool next(Record &record)
        {
            if (at + sizeof(Header) > bytes.size())
                return false;
            Header header;
            std::memcpy(&header, bytes.data() + at, sizeof(header));
            record.kind = header.kind;
            record.value = header.value;
            record.size = header.size;
            record.data = bytes.data() + at + sizeof(Header);
            at += sizeof(Header) + ((static_cast<size_t>(header.size) + 15u) & ~static_cast<size_t>(15u));
            return true;
        }
    };

    Reader read() const { return Reader{bytes}; }

private:
    struct Header
    {
        uint32_t size;
        uint32_t kind;
        uint64_t value;
    };
    static_assert(sizeof(Header) == 16u, "a record's bytes start where its header did, 16 further on");
};
