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

When deserializing:
* For a code instance with the natived flag is set load function names and push code
  the coe instance onto a list to be resolved later.
* After the code instance is fully connected open the shared library and link the
  functions from the shared library into the code instance.
  [This could instead be done in a lazy manner on first use of a function.]

