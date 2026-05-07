#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "fsc/store.h"
#include "fsc/capi-store.h"

#include <cstring>
#include <thread>
#include <chrono>
#include <random>
#include <algorithm>
#include <set>
#include <atomic>

using namespace fsc;

// Helper to generate random strings
std::string random_string(size_t len, std::mt19937& rng) {
    static const char charset[] = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
    std::uniform_int_distribution<> dist(0, sizeof(charset) - 2);
    std::string result;
    result.reserve(len);
    for (size_t i = 0; i < len; ++i) {
        result += charset[dist(rng)];
    }
    return result;
}

TEST_CASE("store stress test", "[store][.]") {
    INFO("Testing store memory management with mixed operations and random data");

    // Create a store
    DataStore store = createStore();

    // Random number generator for test data
    std::random_device rd;
    std::mt19937 rng(rd());

    // Test 1: Create and immediately remove many entries with random keys/values
    SECTION("Immediate add/remove") {
        std::set<std::string> keys;

        for (int i = 0; i < 1000; ++i) {
            // Generate random key and value
            std::string key = "key_" + std::to_string(i) + "_" + random_string(16, rng);
            std::string value = "value_" + std::to_string(i) + "_" + random_string(32, rng);

            // Track keys for later verification (all unique)
            keys.insert(key);

            // Convert to byte arrays
            auto keyData = kj::arrayPtr(
                reinterpret_cast<const byte*>(key.data()),
                key.size()
            );
            auto valueData = kj::heapArray(
                reinterpret_cast<const byte*>(value.data()),
                value.size()
            );

            // Publish and immediately release
            auto entry = store.publish(keyData, kj::mv(valueData));
            // Entry goes out of scope and gets released automatically
        }

        // Force GC to clean up any remaining entries
        store.gc();
    }

    // Test 2: Create entries that persist for some time with random data
    SECTION("Delayed remove") {
        std::vector<StoreEntry> entries;
        std::set<std::string> keys;

        for (int i = 0; i < 100; ++i) {
            // Generate random key and value
            std::string key = "key_delayed_" + std::to_string(i) + "_" + random_string(16, rng);
            std::string value = "value_delayed_" + std::to_string(i) + "_" + random_string(32, rng);

            keys.insert(key);

            auto keyData = kj::arrayPtr(
                reinterpret_cast<const byte*>(key.data()),
                key.size()
            );
            auto valueData = kj::heapArray(
                reinterpret_cast<const byte*>(value.data()),
                value.size()
            );

            entries.push_back(store.publish(keyData, kj::mv(valueData)));
        }

        // Now remove them
        entries.clear();

        // Force GC
        store.gc();
    }

    // Test 3: Mixed operations with random pattern and random data
    SECTION("Mixed operations") {
        std::vector<StoreEntry> activeEntries;
        std::set<std::string> activeKeys;
        std::string value_base = "value_mixed_" + random_string(16, rng);

        for (int i = 0; i < 500; ++i) {
            // Randomly decide: add new entry, keep existing, or remove existing
            int action = i % 3;

            switch (action) {
                case 0: {
                    // Add new entry with random key and value
                    std::string key = "key_mixed_" + std::to_string(i) + "_" + random_string(16, rng);
                    std::string value = value_base + "_" + std::to_string(i) + "_" + random_string(16, rng);

                    auto keyData = kj::arrayPtr(
                        reinterpret_cast<const byte*>(key.data()),
                        key.size()
                    );
                    auto valueData = kj::heapArray(
                        reinterpret_cast<const byte*>(value.data()),
                        value.size()
                    );

                    activeEntries.push_back(store.publish(keyData, kj::mv(valueData)));
                    activeKeys.insert(key);
                    break;
                }
                case 1: {
                    // Query an existing entry (if any exist)
                    if (!activeEntries.empty()) {
                        int idx = i % activeEntries.size();
                        auto entry = activeEntries[idx];
                        // Entry will be released when we overwrite it
                    }
                    break;
                }
                case 2: {
                    // Remove an entry
                    if (!activeEntries.empty()) {
                        activeEntries.pop_back();
                        // Also remove from activeKeys (simplified - just clear set for safety)
                        activeKeys.clear();
                    }
                    break;
                }
            }
        }

        // Clean up remaining entries
        activeEntries.clear();
        activeKeys.clear();
    }

    // Test 4: Concurrent operations with random data
    SECTION("Concurrent operations") {
        std::atomic<int> errors{0};

        auto worker = [&](int id) {
            try {
                std::mt19937 thread_rng(id);  // Seed with thread ID for reproducibility

                for (int i = 0; i < 100; ++i) {
                    std::string key = "key_concurrent_" + std::to_string(id) + "_" + std::to_string(i) + "_" + random_string(16, thread_rng);
                    std::string value = "value_concurrent_" + std::to_string(id) + "_" + std::to_string(i) + "_" + random_string(32, thread_rng);

                    auto keyData = kj::arrayPtr(
                        reinterpret_cast<const byte*>(key.data()),
                        key.size()
                    );
                    auto valueData = kj::heapArray(
                        reinterpret_cast<const byte*>(value.data()),
                        value.size()
                    );

                    auto entry = store.publish(keyData, kj::mv(valueData));
                    // Entry immediately released
                }
            } catch (...) {
                errors.fetch_add(1);
            }
        };

        std::thread t1(worker, 1);
        std::thread t2(worker, 2);
        std::thread t3(worker, 3);

        t1.join();
        t2.join();
        t3.join();

        CHECK(errors.load() == 0);
    }

    // Test 5: Verify query works correctly with random data
    SECTION("Query verification") {
        std::string key = "key_query_" + random_string(16, rng);
        std::string value = "value_query_" + random_string(32, rng);

        auto valueData = kj::heapArray(
            reinterpret_cast<const byte*>(value.data()),
            value.size()
        );
        auto keyData = kj::arrayPtr(
            reinterpret_cast<const byte*>(key.data()),
            key.size()
        );

        // Publish
        auto entry = store.publish(keyData, kj::mv(valueData));

        // Query
        KJ_IF_MAYBE(result, store.query(keyData)) {
            // Verify the data matches
            auto resultPtr = result->asPtr();
            REQUIRE(resultPtr.size() == value.size());
            REQUIRE(memcmp(resultPtr.begin(), value.data(), value.size()) == 0);
        } else {
            REQUIRE(false);  // Query should have returned a result
        }

        // Entry goes out of scope
    }

    // Force final GC
    store.gc();

    // Store goes out of scope - this should clean up everything
}
