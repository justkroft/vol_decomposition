#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>

// Function that multiplies each element of a numpy array by a scalar
static PyObject* multiply_array(PyObject* self, PyObject* args) {
    PyArrayObject *input_array, *output_array;
    double scalar;
    
    // Parse arguments: numpy array and a double scalar
    if (!PyArg_ParseTuple(args, "O!d", &PyArray_Type, &input_array, &scalar)) {
        return NULL;
    }
    
    // Check if array is contiguous and of type double
    if (!PyArray_ISCONTIGUOUS(input_array)) {
        PyErr_SetString(PyExc_ValueError, "Array must be contiguous");
        return NULL;
    }
    
    // Create output array with same shape as input
    npy_intp *dims = PyArray_DIMS(input_array);
    int ndim = PyArray_NDIM(input_array);
    output_array = (PyArrayObject*)PyArray_SimpleNew(ndim, dims, NPY_DOUBLE);
    
    if (output_array == NULL) {
        return NULL;
    }
    
    // Get pointers to data
    double *in_data = (double*)PyArray_DATA(input_array);
    double *out_data = (double*)PyArray_DATA(output_array);
    npy_intp size = PyArray_SIZE(input_array);
    
    // Perform multiplication
    for (npy_intp i = 0; i < size; i++) {
        out_data[i] = in_data[i] * scalar;
    }
    
    return (PyObject*)output_array;
}

// Method definitions
static PyMethodDef VolDecompMethods[] = {
    {"multiply_array", multiply_array, METH_VARARGS, 
     "Multiply a numpy array by a scalar value"},
    {NULL, NULL, 0, NULL}
};

// Module definition
static struct PyModuleDef voldecompmodule = {
    PyModuleDef_HEAD_INIT,
    "vol_decomposition",
    "Module for volatility decomposition operations",
    -1,
    VolDecompMethods
};

// Module initialization
PyMODINIT_FUNC PyInit_vol_decomposition(void) {
    import_array();  // Required for NumPy C API
    return PyModule_Create(&voldecompmodule);
}

// #define PY_SSIZE_T_CLEAN
// #include <Python.h>

// #define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
// #include <numpy/arrayobject.h>
// #include <math.h>

// compute realised variance from returns
// static PyObject* compute_rv(PyObject* self, PyObject* args) {
//     PyArrayObject *returns, *day_indices, *output;
//     int n_days;

//     if (!PyArg_ParseTuple(
//             args, "0!0!i",
//             &PyArray_Type, &returns,
//             &PyArray_Type, &day_indices,
//             &n_days
//         )) {
//         return NULL;
//     }
// }