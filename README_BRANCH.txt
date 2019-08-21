This branch exists to explore native compilation of packages
to shared libraries.

Design:

Augment jl_module_t with the path and a handle to a
corresponding native shared library.

Augment jl_code_instance_t with a "compiled" flag
Set the compiled flag when a code instance is compiled

Augment the serializer to:
* Output a jl_comp_module tag which records the path to
  the package's shared library.
* Output a jl_comp_code_instance tag and serialize
  compiled code instances with symbol names retained.
* Collect code and compile a native shared library.
  This can be stored next to the package's ji file.

Augment the deserializer to:
* Handle jl_comp_module by loading the shared library
  and attaching a handle to it to the module.
* Handle jl_comp_code_instance tag by restoring the
  symbol names, looking them up in the shared library,
  and linking them into the code instance.

Symbol (function) lookup and linking could happen on
demand rather than during deserialization.

