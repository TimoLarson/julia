This branch exists to explore native compilation of packages to shared libraries.

Design:

Augment jl_module_t with the path and a handle to a corresponding native shared library.

Augment jl_code_instance_t with a "natived" flag.
Set the natived flag when a code instance is compiled into a native shared library.

Augment the serializer to:
* Output jl_module_t with the libpath (path to module shared library.)
  [Currently not saving the libpath.]
* Output jl_code_instance with function names retained.
* Collect code and compile a native shared library.
  This can be stored next to the package's ji file.

Augment the deserializer to:
* When loading jl_module to load the libpath, dlopen the module shared library, if any,
  and save its handle in libhandle.
  [Not loading the libpath yet. Opening the shared library elsewhere.]
* When loading jl_code_instance if function names are present look them up in the module's
  shared library and link them into the code instance.

Symbol (function) lookup and linking could happen on
demand rather than during deserialization.

