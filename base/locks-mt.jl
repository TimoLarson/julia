# This file is a part of Julia. License is MIT: https://julialang.org/license

import .Base: unsafe_convert, lock, trylock, unlock, islocked, wait, notify, AbstractLock

export SpinLock, RecursiveSpinLock

# Important Note: these low-level primitives defined here
#   are typically not for general usage

##########################################
# Atomic Locks
##########################################

# Test-and-test-and-set spin locks are quickest up to about 30ish
# contending threads. If you have more contention than that, perhaps
# a lock is the wrong way to synchronize.
"""
    SpinLock()

Create a non-reentrant lock.
Recursive use will result in a deadlock.
Each [`lock`](@ref) must be matched with an [`unlock`](@ref).

Test-and-test-and-set spin locks are quickest up to about 30ish
contending threads. If you have more contention than that, perhaps
a lock is the wrong way to synchronize.
"""
struct SpinLock <: AbstractLock
    handle::Atomic{Int}
    SpinLock() = new(Atomic{Int}(0))
end

function lock(l::SpinLock)
    while true
        if l.handle[] == 0
            p = atomic_xchg!(l.handle, 1)
            if p == 0
                return
            end
        end
        ccall(:jl_cpu_pause, Cvoid, ())
        # Temporary solution before we have gc transition support in codegen.
        ccall(:jl_gc_safepoint, Cvoid, ())
    end
end

function trylock(l::SpinLock)
    if l.handle[] == 0
        return atomic_xchg!(l.handle, 1) == 0
    end
    return false
end

function unlock(l::SpinLock)
    l.handle[] = 0
    ccall(:jl_cpu_wake, Cvoid, ())
    return
end

function islocked(l::SpinLock)
    return l.handle[] != 0
end

"""
    RecursiveSpinLock()

Creates a reentrant lock.
The same thread can acquire the lock as many times as required.
Each [`lock`](@ref) must be matched with an [`unlock`](@ref).

See also [`SpinLock`](@ref) for a slightly faster version.
"""
struct RecursiveSpinLock <: AbstractLock
    ownertid::Atomic{Int16}
    handle::Atomic{Int}
    RecursiveSpinLock() = new(Atomic{Int16}(0), Atomic{Int}(0))
end

function lock(l::RecursiveSpinLock)
    if l.ownertid[] == threadid()
        l.handle[] += 1
        return
    end
    while true
        if l.handle[] == 0
            if atomic_cas!(l.handle, 0, 1) == 0
                l.ownertid[] = threadid()
                return
            end
        end
        ccall(:jl_cpu_pause, Cvoid, ())
        # Temporary solution before we have gc transition support in codegen.
        ccall(:jl_gc_safepoint, Cvoid, ())
    end
end

function trylock(l::RecursiveSpinLock)
    if l.ownertid[] == threadid()
        l.handle[] += 1
        return true
    end
    if l.handle[] == 0
        if atomic_cas!(l.handle, 0, 1) == 0
            l.ownertid[] = threadid()
            return true
        end
        return false
    end
    return false
end

function unlock(l::RecursiveSpinLock)
    @assert(l.ownertid[] == threadid(), "unlock from wrong thread")
    @assert(l.handle[] != 0, "unlock count must match lock count")
    if l.handle[] == 1
        l.ownertid[] = 0
        l.handle[] = 0
        ccall(:jl_cpu_wake, Cvoid, ())
    else
        l.handle[] -= 1
    end
    return
end

function islocked(l::RecursiveSpinLock)
    return l.handle[] != 0
end
