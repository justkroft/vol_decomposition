#define PY_SSIZE_T_CLEAN
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_11_API_VERSION
#include <numpy/arrayobject.h>
#include <math.h>


static double mu_func(double p) {
    double num = pow(2.0, p / 2.0) * tgamma((p + 1.0) / 2.0);
    double den = tgamma(0.5);   // equals sqrt(pi)
    return num / den;
}


// realized variance
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

    // do actual computation
    for (npy_intp i = 0; i < n; i++) {
        long day_idx = day_arr[i];
        out_arr[day_idx] += ret_arr[i] * ret_arr[i];
    }

    return (PyObject*) output;
}


// bi-power variance
static PyObject* compute_bipower_variance(PyObject* self, PyObject* args) {
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

    const npy_intp dims[1] = {n_days};
    output = (PyArrayObject*) PyArray_ZEROS(1, dims, NPY_DOUBLE, 0);
    if (output == NULL) {
        return NULL;
    }

    double* ret_arr = (double*) PyArray_DATA(returns);
    long* day_arr = (long*) PyArray_DATA(day_indices);
    double* out_arr = (double*) PyArray_DATA(output);

    const npy_intp n = PyArray_SIZE(returns);

    for (npy_intp i = 1; i < n; i++) {
        if (day_arr[i] == day_arr[i-1]) {
            long day_idx = day_arr[i];
            out_arr[day_idx] += fabs(ret_arr[i-1]) * fabs(ret_arr[i]);
        }
    }

    // compute mu_1^(-2)
    double mu_1 = mu_func(1.0);
    double mu_1_inv_sq = 1.0 / (mu_1 * mu_1);

    // scale by mu_1^(-2)
    for (int i = 0; i < n_days; i++) {
        out_arr[i] *= mu_1_inv_sq;
    }

    return (PyObject*) output;
}


// tri-power variance
static PyObject* compute_tripower_quarticity(PyObject* self, PyObject* args) {
    PyArrayObject *returns, *day_indices, *output;
    int n_days;
    double delta;

    if (!PyArg_ParseTuple(
        args, "O!O!id",
        &PyArray_Type, &returns,
        &PyArray_Type, &day_indices,
        &n_days,
        &delta
    )) {
        return NULL;
    }

    const npy_intp dims[1] = {n_days};
    output = (PyArrayObject*) PyArray_ZEROS(1, dims, NPY_DOUBLE, 0);
    if (output == NULL) {
        return NULL;
    }

    double* ret_arr = (double*) PyArray_DATA(returns);
    long *day_arr = (long*) PyArray_DATA(day_indices);
    double *out_arr = (double*) PyArray_DATA(output);

    const npy_intp n = PyArray_SIZE(returns);
    const double exponent = 4.0 / 3.0;

    for (npy_intp i = 2; i < n; i++) {
        if (day_arr[i] == day_arr[i-1] && day_arr[i-1] == day_arr[i-2]) {
            long day_idx = day_arr[i];
            double val = pow(fabs(ret_arr[i-2]), exponent)
                       * pow(fabs(ret_arr[i-1]), exponent)
                       * pow(fabs(ret_arr[i]), exponent);
            out_arr[day_idx] += val; 
        }
    }
    
    // compute mu_43^3 and scale
    double mu_43 = mu_func(4.0 / 3.0);

    double scale = delta * mu_43 * mu_43 * mu_43;
    for (int i = 0; i < n_days; i++) {
        out_arr[i] /= scale;
    }

    return (PyObject*) output;
}


// Method definitions
static PyMethodDef VolDecompMethods[] = {
    {
        "compute_realised_variance",
        compute_realised_variance,
        METH_VARARGS, 
        "Compute realised variance"
    },
    {
        "compute_bipower_variance",
        compute_bipower_variance,
        METH_VARARGS, 
        "Compute bi-power variance"
    },
    {
        "compute_tripower_quarticity",
        compute_tripower_quarticity,
        METH_VARARGS,
        "Compute tri-power quarticity"
    },
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
