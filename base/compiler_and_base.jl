# This file is a part of Julia. License is MIT: https://julialang.org/license

#Core.include(Core, "boot.jl")
Core.include(Main, "compiler/compiler.jl")
Core.println("\n== compiler included ==\n")
Core.include(Main, "BaseTrimmed.jl")

using .Base

# Ensure this file is also tracked
pushfirst!(Base._included_files, (@__MODULE__, joinpath(@__DIR__, "Base.jl")))
pushfirst!(Base._included_files, (@__MODULE__, joinpath(@__DIR__, "sysimg.jl")))

# set up depot & load paths to be able to find stdlib packages
@eval Base creating_sysimg = true

# ADDED
#Base.reinit_stdio()

Base.init_depot_path()
Base.init_load_path()

#Base.__init__()

Core.println("\n== in sysimg.jl ==\n")
Core.println("\n== Base: ", Base, " ==\n")
#Core.println("\n== is_primary_base_module: ", is_primary_base_module, " ==\n")
Core.println("\n== Base.is_primary_base_module: ", Base.is_primary_base_module, " ==\n")

#=
# ADDED FOR DEBUGGING
#=

zed1 = "z1"
if Base.is_primary_base_module
    zed2 = "z2"
# load some stdlib packages but don't put their names in Main
let
    # Stdlibs manually sorted in top down order
    stdlibs = [
            # No deps
            #=
            :Base64,
            :CRC32c,
            :SHA,
            :FileWatching,
            :Unicode,
            :Mmap,
            :Serialization,
            :Libdl,
            :Printf,
            :Markdown,
            :LibGit2,
            :Logging,
            :Sockets,
            :Profile,
            :Dates,
            :DelimitedFiles,
            :Random,
            :UUIDs,
            :Future,
            :LinearAlgebra,
            :SparseArrays,
            :SuiteSparse,
            :Distributed,
            :SharedArrays,
            :Pkg,
            :Test,
            :REPL,
            :Statistics,
            =#
        ]

    maxlen = isempty(stdlibs) ? 0 : maximum(textwidth.(string.(stdlibs)))

    print_time = (mod, t) -> (print(rpad(string(mod) * "  ", maxlen + 3, "─")); Base.time_print(t * 10^9); println())
    print_time(Base, (Base.end_base_include - Base.start_base_include) * 10^(-9))

    Base._track_dependencies[] = true
    Base.tot_time_stdlib[] = @elapsed for stdlib in stdlibs
        println("DEBUG: ", stdlib)
        tt = @elapsed Base.require(Base, stdlib)
        print_time(stdlib, tt)
    end
    for dep in Base._require_dependencies
        dep[3] == 0.0 && continue
        push!(Base._included_files, dep[1:2])
    end
    empty!(Base._require_dependencies)
    Base._track_dependencies[] = false

    print_time("Stdlibs total", Base.tot_time_stdlib[])
end
end

# ADDED FOR DEBUGGING
=# # if false

# Clear global state
empty!(Core.ARGS)
empty!(Base.ARGS)
empty!(LOAD_PATH)
@eval Base creating_sysimg = false
Base.init_load_path() # want to be able to find external packages in userimg.jl

# ADDED FOR DEBUGGING
#=

let
tot_time_userimg = @elapsed (Base.isfile("userimg.jl") && Base.include(Main, "userimg.jl"))


tot_time_base = (Base.end_base_include - Base.start_base_include) * 10.0^(-9)
tot_time = tot_time_base + Base.tot_time_stdlib[] + tot_time_userimg

println("Sysimage built. Summary:")
print("Total ─────── "); Base.time_print(tot_time               * 10^9); print(" \n");
print("Base: ─────── "); Base.time_print(tot_time_base          * 10^9); print(" "); show(IOContext(stdout, :compact=>true), (tot_time_base          / tot_time) * 100); println("%")
print("Stdlibs: ──── "); Base.time_print(Base.tot_time_stdlib[] * 10^9); print(" "); show(IOContext(stdout, :compact=>true), (Base.tot_time_stdlib[] / tot_time) * 100); println("%")
if isfile("userimg.jl")
print("Userimg: ──── "); Base.time_print(tot_time_userimg       * 10^9); print(" "); show(IOContext(stdout, :compact=>true), (tot_time_userimg       / tot_time) * 100); println("%")
end
end

# ADDED FOR DEBUGGING
=# # if false

empty!(LOAD_PATH)
empty!(DEPOT_PATH)
=#
