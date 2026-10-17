/* Runtime DLL resolution for WanGP's prebuilt Triton HIP helper.
 * Copyright (c) 2026 DeepBeepMeep. MIT license, see TRITON_LICENSE.txt.
 * Included by the build recipe; no user-specific path is compiled in.
 */
static void *wangp_load_hip_library(void) {
  PyObject *driver = PyImport_ImportModule("triton.backends.amd.driver");
  if (!driver)
    return NULL;
  PyObject *path = PyObject_CallMethod(driver, "_get_path_to_hip_runtime_dylib", NULL);
  Py_DECREF(driver);
  if (!path)
    return NULL;
  wchar_t *filename = PyUnicode_AsWideCharString(path, NULL);
  Py_DECREF(path);
  if (!filename)
    return NULL;

  // Match Triton's reuse of PyTorch's HIP runtime, including versioned DLL names.
  const wchar_t *basename = wcsrchr(filename, L'\\');
  const wchar_t *slash = wcsrchr(filename, L'/');
  if (!basename || (slash && slash > basename))
    basename = slash;
  basename = basename ? basename + 1 : filename;
  HMODULE library = NULL;
  if (!GetModuleHandleExW(0, basename, &library))
    library = LoadLibraryExW(filename, NULL, LOAD_LIBRARY_SEARCH_DEFAULT_DIRS | LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR);
  PyMem_Free(filename);
  if (!library)
    PyErr_SetFromWindowsErr(0);
  return (void *)library;
}
