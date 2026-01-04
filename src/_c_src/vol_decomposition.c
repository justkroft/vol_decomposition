#define PY_SSIZE_T_CLEAN
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_11_API_VERSION
#include <numpy/arrayobject.h>

// Compute realized variance from returns
static PyObject* compute_realised_variance(PyObject* self, PyObject* args) {
    PyArrayObject *returns, *day_indices, *output;
    int n_days;

    if (!PyArg_ParseTuple(
        args, "O!O!i",
        &PyArray_Type, &returns,
        &PyArray_Type, &day_indices,
        &n_days
    )) {
        return NULL;
    }

    // create output array
    const npy_intp dims[1] = {n_days};
    output = (PyArrayObject*) PyArray_ZEROS(1, dims, NPY_DOUBLE, 0);
    if (output == NULL) {
        return NULL;
    }
    // get pointers
    double* ret_arr = (double*) PyArray_DATA(returns);
    long* day_arr = (long*) PyArray_DATA(day_indices);
    double* out_arr = (double*) PyArray_DATA(output);

    const npy_intp n = PyArray_SIZE(returns);

    // actual computation
    for (npy_intp i = 0; i < n; i++) {
        long day_idx = day_arr[i];
        out_arr[day_idx] += ret_arr[i] * ret_arr[i];
    }

    return (PyObject*) output;
}

// Method definitions
static PyMethodDef VolDecompMethods[] = {
    {"compute_realised_variance", compute_realised_variance, METH_VARARGS, 
     "Compute realised variance"},
    {NULL, NULL, 0, NULL}
};

// Module definition
static struct PyModuleDef voldecompmodule = {
    PyModuleDef_HEAD_INIT,
    "_vol_decomposition_c",
    "Module for volatility decomposition operations",
    -1,
    VolDecompMethods
};

// Module initialization
PyMODINIT_FUNC PyInit__vol_decomposition_c(void) {
    import_array();  // Required for NumPy C API
    return PyModule_Create(&voldecompmodule);
}
