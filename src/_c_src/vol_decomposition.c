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


typedef struct {
    PyArrayObject* returns;
    PyArrayObject* day_indices;
    PyArrayObject* output;
    npy_intp n;
    npy_intp n_days;
} VarInputs;


static int prepare_var_inputs(PyObject* args, VarInputs* in) {
    PyObject* returns_obj = NULL;
    PyObject* day_indices_obj = NULL;

    in->returns = NULL;
    in->day_indices = NULL;
    in->output = NULL;

    if (!PyArg_ParseTuple(
        args, "O!O!n",
        &PyArray_Type, &returns_obj,
        &PyArray_Type, &day_indices_obj,
        &in->n_days
    )) {
        return 0;
    }

    if (in->n_days <= 0) {
        PyErr_SetString(PyExc_ValueError, "n_days must be positive");
        return 0;
    }

    // convert inputs to well-defined NumPy arrays
    in->returns = (PyArrayObject*) PyArray_FROM_OTF(
        returns_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY
    );
    if (!in->returns) {
        return 0;
    }

    in->day_indices = (PyArrayObject*) PyArray_FROM_OTF(
        day_indices_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY
    );
    if (!in->day_indices) {
        Py_DECREF(in->returns);
        in->returns = NULL;
        return 0;
    }

    // validate dimensions
    if (PyArray_NDIM(in->returns) != 1 ||
        PyArray_NDIM(in->day_indices) != 1)
    {
        PyErr_SetString(
            PyExc_ValueError,
            "returns and day_indices must be 1-D arrays"
        );
        Py_DECREF(in->returns);
        Py_DECREF(in->day_indices);
        in->returns = in->day_indices = NULL;
        return 0;
    }

    in->n = PyArray_SIZE(in->returns);
    if (in->n != PyArray_SIZE(in->day_indices)) {
        PyErr_SetString(
            PyExc_ValueError,
            "returns and day_indices must have the same length"
        );
        Py_DECREF(in->returns);
        Py_DECREF(in->day_indices);
        in->returns = in->day_indices = NULL;
        return 0;
    }

    // create output array
    const npy_intp dims[1] = { in->n_days };
    in->output = (PyArrayObject*) PyArray_ZEROS(1, dims, NPY_DOUBLE, 0);
    if (!in->output) {
        Py_DECREF(in->returns);
        Py_DECREF(in->day_indices);
        in->returns = in->day_indices = NULL;
        return 0;
    }

    return 1;
}


// helper function to clean up
static void free_var_inputs(VarInputs* in) {
    Py_XDECREF(in->returns);
    Py_XDECREF(in->day_indices);
    Py_XDECREF(in->output);
}


// realized variance
static PyObject* compute_realised_variance(PyObject* self, PyObject* args) {
    VarInputs in;
    if (!prepare_var_inputs(args, &in)) {
        return NULL;
    }

    // get pointers
    double* ret_arr = (double*) PyArray_DATA(in.returns);
    int64_t* day_arr = (int64_t*) PyArray_DATA(in.day_indices);
    double* out_arr = (double*) PyArray_DATA(in.output);

    // do actual computation
    for (npy_intp i = 0; i < in.n; i++) {
        int64_t day_idx = day_arr[i];

        if (day_idx < 0) {
            PyErr_SetString(PyExc_IndexError, "day_indices contains out-of-bounds value (< 0)");
            free_var_inputs(&in);
            return NULL;
        }

        double r = ret_arr[i];
        out_arr[day_idx] += r * r;
    }

    Py_DECREF(in.returns);
    Py_DECREF(in.day_indices);
    return (PyObject*) in.output;
}


// bi-power variance
static PyObject* compute_bipower_variance(PyObject* self, PyObject* args) {
    VarInputs in;
    if (!prepare_var_inputs(args, &in)) {
        return NULL;
    }

    double* ret_arr = (double*) PyArray_DATA(in.returns);
    int64_t* day_arr = (int64_t*) PyArray_DATA(in.day_indices);
    double* out_arr = (double*) PyArray_DATA(in.output);

    for (npy_intp i = 1; i < in.n; i++) {
        if (day_arr[i] == day_arr[i-1]) {
            int64_t day_idx = day_arr[i];

            if (day_idx < 0) {
                PyErr_SetString(
                    PyExc_IndexError,
                    "day_indices contains out-of-bounds value (< 0)"
                );
                free_var_inputs(&in);
                return NULL;
            }

            out_arr[day_idx] += fabs(ret_arr[i-1]) * fabs(ret_arr[i]);
        }
    }

    // compute mu_1^(-2)
    const double mu_1 = mu_func(1.0);
    const double mu_1_inv_sq = 1.0 / (mu_1 * mu_1);

    // scale by mu_1^(-2)
    for (int i = 0; i < in.n_days; i++) {
        out_arr[i] *= mu_1_inv_sq;
    }

    Py_DECREF(in.returns);
    Py_DECREF(in.day_indices);
    return (PyObject*) in.output;
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
