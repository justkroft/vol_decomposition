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
    PyObject *returns_obj = NULL, *day_indices_obj = NULL;
    PyArrayObject *returns = NULL, *day_indices = NULL, *output = NULL;
    npy_intp n_days;

    if (!PyArg_ParseTuple(
        args, "O!O!n",
        &PyArray_Type, &returns_obj,
        &PyArray_Type, &day_indices_obj,
        &n_days
    )) {
        return NULL;
    }

    if (n_days <= 0) {
        PyErr_SetString(PyExc_ValueError, "n_days must be positive");
        return NULL;
    }

    // Convert inputs to well-defined NumPy arrays
    returns = (PyArrayObject*) PyArray_FROM_OTF(
        returns_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY
    );
    if (returns == NULL) {
        return NULL;
    }

    day_indices = (PyArrayObject*) PyArray_FROM_OTF(
        day_indices_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY
    );
    if (day_indices == NULL) {
        Py_DECREF(returns);
        return NULL;
    }

    // Validate dimensions
    if (PyArray_NDIM(returns) != 1 || PyArray_NDIM(day_indices) != 1) {
        PyErr_SetString(PyExc_ValueError, "returns and day_indices must be 1D arrays");
        Py_DECREF(returns);
        Py_DECREF(day_indices);
        return NULL;
    }

    if (PyArray_SIZE(returns) != PyArray_SIZE(day_indices)) {
        PyErr_SetString(PyExc_ValueError, "returns and day_indices must have the same length");
        Py_DECREF(returns);
        Py_DECREF(day_indices);
        return NULL;
    }

    // Create output array
    npy_intp dims[1] = {n_days};
    output = (PyArrayObject*) PyArray_ZEROS(1, dims, NPY_DOUBLE, 0);
    if (output == NULL) {
        Py_DECREF(returns);
        Py_DECREF(day_indices);
        return NULL;
    }

    // get pointers
    double* ret_arr = (double*) PyArray_DATA(returns);
    int64_t* day_arr = (int64_t*) PyArray_DATA(day_indices);
    double* out_arr = (double*) PyArray_DATA(output);

    const npy_intp n = PyArray_SIZE(returns);

    // do actual computation
    for (npy_intp i = 0; i < n; i++) {
        int64_t day_idx = day_arr[i];

        if (day_idx < 0) {
            PyErr_SetString(PyExc_IndexError, "day_indices contains out-of-bounds value (< 0)");
            Py_DECREF(returns);
            Py_DECREF(day_indices);
            Py_DECREF(output);
            return NULL;
        }

        double r = ret_arr[i];
        out_arr[day_idx] += r * r;
    }

    Py_DECREF(returns);
    Py_DECREF(day_indices);
    return (PyObject*) output;
}


// bi-power variance
static PyObject* compute_bipower_variance(PyObject* self, PyObject* args) {
    PyObject *returns_obj = NULL, *day_indices_obj = NULL;
    PyArrayObject *returns = NULL, *day_indices = NULL, *output = NULL;
    npy_intp n_days;

    if (!PyArg_ParseTuple(
        args, "O!O!n",
        &PyArray_Type, &returns_obj,
        &PyArray_Type, &day_indices_obj,
        &n_days
    )) {
        return NULL;
    }

    if (n_days <= 0) {
        PyErr_SetString(PyExc_ValueError, "n_days must be positive");
        return NULL;
    }

    returns = (PyArrayObject*) PyArray_FROM_OTF(
        returns_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY
    );
    if (returns == NULL) {
        return NULL;
    }

    day_indices = (PyArrayObject*) PyArray_FROM_OTF(
        day_indices_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY
    );
    if (day_indices == NULL) {
        Py_DECREF(returns);
        return NULL;
    }

    if (PyArray_NDIM(returns) != 1 || PyArray_NDIM(day_indices) != 1) {
        PyErr_SetString(PyExc_ValueError, "returns and day_indices must be 1D arrays");
        Py_DECREF(returns);
        Py_DECREF(day_indices);
        return NULL;
    }

    if (PyArray_SIZE(returns) != PyArray_SIZE(day_indices)) {
        PyErr_SetString(PyExc_ValueError, "returns and day_indices must have the same length");
        Py_DECREF(returns);
        Py_DECREF(day_indices);
        return NULL;
    }

    const npy_intp dims[1] = {n_days};
    output = (PyArrayObject*) PyArray_ZEROS(1, dims, NPY_DOUBLE, 0);
    if (output == NULL) {
        Py_DECREF(returns);
        Py_DECREF(day_indices);
        return NULL;
    }

    double* ret_arr = (double*) PyArray_DATA(returns);
    int64_t* day_arr = (int64_t*) PyArray_DATA(day_indices);
    double* out_arr = (double*) PyArray_DATA(output);

    const npy_intp n = PyArray_SIZE(returns);

    for (npy_intp i = 1; i < n; i++) {
        if (day_arr[i] == day_arr[i-1]) {
            int64_t day_idx = day_arr[i];

            if (day_idx < 0) {
                PyErr_SetString(
                    PyExc_IndexError,
                    "day_indices contains out-of-bounds value (< 0)"
                );
                Py_DECREF(returns);
                Py_DECREF(day_indices);
                Py_DECREF(output);
                return NULL;
            }

            out_arr[day_idx] += fabs(ret_arr[i-1]) * fabs(ret_arr[i]);
        }
    }

    // compute mu_1^(-2)
    const double mu_1 = mu_func(1.0);
    const double mu_1_inv_sq = 1.0 / (mu_1 * mu_1);

    // scale by mu_1^(-2)
    for (int i = 0; i < n_days; i++) {
        out_arr[i] *= mu_1_inv_sq;
    }

    Py_DECREF(returns);
    Py_DECREF(day_indices);
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

    if (n_days <= 0) {
        PyErr_SetString(PyExc_ValueError, "n_days must be positive");
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
    const double mu_43 = mu_func(4.0 / 3.0);

    const double scale = delta * mu_43 * mu_43 * mu_43;
    for (int i = 0; i < n_days; i++) {
        out_arr[i] /= scale;
    }

    return (PyObject*) output;
}


// z-statistic
static PyObject* compute_z_stats(PyObject* self, PyObject* args) {
    PyArrayObject *realised_variance, *bipower_variance, *tripower_quarticity, *output;
    double delta;

    if (!PyArg_ParseTuple(
        args, "O!O!O!d",
        &PyArray_Type, &realised_variance,
        &PyArray_Type, &bipower_variance,
        &PyArray_Type, &tripower_quarticity,
        &delta
    )) {
        return NULL;
    }

    const npy_intp n = PyArray_SIZE(realised_variance);
    npy_intp dims[1] = {n};
    output = (PyArrayObject*) PyArray_ZEROS(1, dims, NPY_DOUBLE, 0);
    if (output == NULL) {
        return NULL;
    }

    double* rv_arr = (double*) PyArray_DATA(realised_variance);
    double* bpv_arr = (double*) PyArray_DATA(bipower_variance);
    double* tpq_arr = (double*) PyArray_DATA(tripower_quarticity);
    double* out_arr = (double*) PyArray_DATA(output);

    const double mu_1 = mu_func(1);
    const double mu_1_inv_4 = 1.0 / (mu_1 * mu_1 * mu_1 * mu_1);
    const double mu_1_inv_2 = 1.0 / (mu_1 * mu_1);
    const double const_term = sqrt(mu_1_inv_4 + 2.0 * mu_1_inv_2 - 5.0);
    const double sqrt_delta = sqrt(delta);

    for (npy_intp i = 0; i < n; i++) {
        double max_func;
        if (bpv_arr[i] != 0.0) {
            double ratio = tpq_arr[i] / (bpv_arr[i] * bpv_arr[i]);
            max_func = fmax(1.0, ratio);
        } 
        else {
            max_func = 1.0;
        }

        if (rv_arr[i] > 0.0) {
            out_arr[i] = (rv_arr[i] - bpv_arr[i])
                       / (rv_arr[i] * const_term * max_func * sqrt_delta);
        }
        else {
            out_arr[i] = 0.0;
        }
    }

    return (PyObject*) output;
}


// apply jumpy filter; compute continous and jumpy component
// static PyObject* apply_jump_filter(PyObject* self, PyObject* args) {
//     PyArrayObject *realised_variance, *bipower_variance, *z_stats;
//     PyArrayObject *cont_out, *jump_out;
//     double sig_threshold;
//     int truncate_zero;
// }


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
    {
        "compute_z_stats",
        compute_z_stats,
        METH_VARARGS,
        "Compute the z-statistics for jump"
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
