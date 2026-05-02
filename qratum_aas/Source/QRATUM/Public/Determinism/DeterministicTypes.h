/// File: Source/QRATUM/Public/Determinism/DeterministicTypes.h
// Copyright QRATUM Platform. All Rights Reserved.
// Deterministic type definitions for replay-safe AI

#pragma once

#include "CoreMinimal.h"

/**
 * QRATUM Determinism Layer
 * 
 * This header defines deterministic primitives that ensure identical behavior
 * across runs, machines, sessions, and replays. These types override Unreal
 * defaults where necessary to maintain invariant execution.
 * 
 * Determinism Requirements:
 * 1. Fixed-seed RNG with explicit state management
 * 2. Ordered containers for iteration stability
 * 3. Stable floating-point operations (avoiding platform-specific SIMD)
 * 4. Explicit tick ordering for multi-agent systems
 * 
 * Usage:
 * - All AAS operations MUST use these types instead of raw STL or UE containers
 * - RNG state must be serialized for replay support
 * - Container iteration order must be deterministic (no hash-based iteration)
 */

namespace QRATUM
{
    /**
     * Fixed-precision value type for deterministic calculations.
     * Uses integer representation to avoid floating-point variance.
     * 
     * Range: [-32768.0, 32767.999969] with ~0.000031 precision
     * This provides sufficient range and precision for game AI heuristics
     * while guaranteeing bit-exact reproduction across platforms.
     */
    struct FFixedPoint32
    {
        int32 RawValue;

        static constexpr int32 FractionalBits = 15;
        static constexpr int32 Scale = 1 << FractionalBits; // 32768

        FFixedPoint32() : RawValue(0) {}
        explicit FFixedPoint32(int32 Raw) : RawValue(Raw) {}
        
        static FFixedPoint32 FromFloat(float Value)
        {
            FFixedPoint32 Result;
            Result.RawValue = static_cast<int32>(Value * Scale);
            return Result;
        }

        static FFixedPoint32 FromInt(int32 Value)
        {
            FFixedPoint32 Result;
            Result.RawValue = Value * Scale;
            return Result;
        }

        float ToFloat() const
        {
            return static_cast<float>(RawValue) / Scale;
        }

        int32 ToInt() const
        {
            return RawValue / Scale;
        }

        FFixedPoint32 operator+(const FFixedPoint32& Other) const
        {
            return FFixedPoint32(RawValue + Other.RawValue);
        }

        FFixedPoint32 operator-(const FFixedPoint32& Other) const
        {
            return FFixedPoint32(RawValue - Other.RawValue);
        }

        FFixedPoint32 operator*(const FFixedPoint32& Other) const
        {
            // Use 64-bit intermediate to prevent overflow
            int64 Result = static_cast<int64>(RawValue) * Other.RawValue;
            return FFixedPoint32(static_cast<int32>(Result >> FractionalBits));
        }

        FFixedPoint32 operator/(const FFixedPoint32& Other) const
        {
            if (Other.RawValue == 0)
            {
                return FFixedPoint32(RawValue > 0 ? INT32_MAX : INT32_MIN);
            }
            int64 Result = (static_cast<int64>(RawValue) << FractionalBits) / Other.RawValue;
            return FFixedPoint32(static_cast<int32>(Result));
        }

        FFixedPoint32 operator-() const
        {
            return FFixedPoint32(-RawValue);
        }

        bool operator<(const FFixedPoint32& Other) const { return RawValue < Other.RawValue; }
        bool operator>(const FFixedPoint32& Other) const { return RawValue > Other.RawValue; }
        bool operator<=(const FFixedPoint32& Other) const { return RawValue <= Other.RawValue; }
        bool operator>=(const FFixedPoint32& Other) const { return RawValue >= Other.RawValue; }
        bool operator==(const FFixedPoint32& Other) const { return RawValue == Other.RawValue; }
        bool operator!=(const FFixedPoint32& Other) const { return RawValue != Other.RawValue; }

        static FFixedPoint32 Zero() { return FFixedPoint32(0); }
        static FFixedPoint32 One() { return FFixedPoint32(Scale); }
        static FFixedPoint32 Max() { return FFixedPoint32(INT32_MAX); }
        static FFixedPoint32 Min() { return FFixedPoint32(INT32_MIN); }
    };

    /**
     * Deterministic pseudo-random number generator.
     * 
     * Uses xorshift64* algorithm which provides:
     * - Full 64-bit state for long periods
     * - High-quality randomness (passes TestU01)
     * - Deterministic: same seed = same sequence
     * - Fast: single multiply per number
     * 
     * State must be saved/restored for replay support.
     */
    class FDeterministicRNG
    {
    public:
        /** Default seed: "QATU" in ASCII (0x51='Q', 0x41='A', 0x54='T', 0x55='U') */
        static constexpr uint64 DEFAULT_RNG_SEED = 0x51415455;

        explicit FDeterministicRNG(uint64 Seed = DEFAULT_RNG_SEED)
            : State(Seed != 0 ? Seed : DEFAULT_RNG_SEED) // Prevent zero state
        {
        }

        /** Reset to a new seed value */
        void Seed(uint64 NewSeed)
        {
            State = NewSeed != 0 ? NewSeed : DEFAULT_RNG_SEED;
        }

        /** Get current state for serialization */
        uint64 GetState() const { return State; }

        /** Restore state from serialization */
        void SetState(uint64 NewState)
        {
            State = NewState != 0 ? NewState : 0x51415455;
        }

        /** Generate next 64-bit unsigned integer */
        uint64 Next()
        {
            // xorshift64* algorithm
            State ^= State >> 12;
            State ^= State << 25;
            State ^= State >> 27;
            return State * 0x2545F4914F6CDD1DULL;
        }

        /** Generate uniform integer in range [0, Max) */
        uint64 NextInRange(uint64 Max)
        {
            if (Max <= 1) return 0;
            
            // Rejection sampling for uniform distribution
            uint64 Threshold = (~Max + 1) % Max; // = (2^64 - Max) % Max
            uint64 Result;
            do
            {
                Result = Next();
            }
            while (Result < Threshold);
            
            return Result % Max;
        }

        /** Generate uniform integer in range [Min, Max) */
        int64 NextInRange(int64 Min, int64 Max)
        {
            if (Min >= Max) return Min;
            return Min + static_cast<int64>(NextInRange(static_cast<uint64>(Max - Min)));
        }

        /** Generate uniform float in range [0, 1) */
        float NextFloat()
        {
            // Use upper 24 bits for mantissa (float has 23-bit mantissa + implicit 1)
            return static_cast<float>(Next() >> 40) / static_cast<float>(1ULL << 24);
        }

        /** Generate uniform float in range [Min, Max) */
        float NextFloatInRange(float Min, float Max)
        {
            return Min + NextFloat() * (Max - Min);
        }

        /** Generate uniform FFixedPoint32 in range [0, 1) */
        FFixedPoint32 NextFixed()
        {
            // Use 15 bits for fractional part
            int32 Raw = static_cast<int32>(Next() >> 49) & (FFixedPoint32::Scale - 1);
            return FFixedPoint32(Raw);
        }

        /** Shuffle array in place using Fisher-Yates algorithm */
        template<typename T>
        void Shuffle(TArray<T>& Array)
        {
            const int32 Num = Array.Num();
            for (int32 i = Num - 1; i > 0; --i)
            {
                int32 j = static_cast<int32>(NextInRange(static_cast<uint64>(i + 1)));
                Array.Swap(i, j);
            }
        }

    private:
        uint64 State;
    };

    /**
     * Deterministic hash function for stable container ordering.
     * Uses FNV-1a which is simple, fast, and deterministic.
     */
    inline uint64 DeterministicHash(const void* Data, size_t Size)
    {
        const uint64 FNV_OFFSET_BASIS = 14695981039346656037ULL;
        const uint64 FNV_PRIME = 1099511628211ULL;

        uint64 Hash = FNV_OFFSET_BASIS;
        const uint8* Bytes = static_cast<const uint8*>(Data);

        for (size_t i = 0; i < Size; ++i)
        {
            Hash ^= Bytes[i];
            Hash *= FNV_PRIME;
        }

        return Hash;
    }

    /** Hash combine for multiple values */
    inline uint64 HashCombine(uint64 A, uint64 B)
    {
        return A ^ (B + 0x9E3779B97F4A7C15ULL + (A << 6) + (A >> 2));
    }
}
