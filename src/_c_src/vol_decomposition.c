#define PY_SSIZE_T_CLEAN
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_11_API_VERSION
#include <numpy/arrayobject.h>
#include <numpy/npy_math.h>
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


static int array_has_nan_or_inf(PyArrayObject* arr, const char* name) {
    const npy_intp n = PyArray_SIZE(arr);
    double* data = (double*) PyArray_DATA(arr);

    for (npy_intp i = 0; i < n; i++) {
        if (!npy_isfinite(data[i])) {
            PyErr_Format(
                PyExc_ValueError,
                "%s contains NaN or Inf at index %ld",
                name, (long)i
            );
            return 1;
        }
    }
    return 0;
}


// Helper function to validate day_indices are within bounds
static int invalid_day_indices(
    PyArrayObject* day_indices,
    npy_intp n_days
) {
    npy_intp n = PyArray_SIZE(day_indices);
    int64_t* data = (int64_t*) PyArray_DATA(day_indices);

    for (npy_intp i = 0; i < n; i++) {
        if (data[i] < 0) {
            PyErr_Format(
                PyExc_IndexError,
                "day_indices[%ld] = %ld is negative",
                (long)i, (long)data[i]
            );
            return 1;
        }
        if (data[i] >= n_days) {
            PyErr_Format(
                PyExc_IndexError,
                "day_indices[%ld] = %ld is out of bounds (n_days = %ld)",
                (long)i, (long)data[i], (long)n_days
            );
            return 1;
        }
    }
    return 0;
}


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
    if (invalid_day_indices(in->day_indices, in->n_days)) {
        Py_DECREF(in->returns);
        Py_DECREF(in->day_indices);
        in->returns = in->day_indices = NULL;
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

    // validate array is not empty
    if (in->n == 0) {
        PyErr_SetString(
            PyExc_ValueError,
            "returns and day_indices must not be empty"
        );
        Py_DECREF(in->returns);
        Py_DECREF(in->day_indices);
        in->returns = in->day_indices = NULL;
        return 0;
    }

    // check for nan/inf
    if (array_has_nan_or_inf(in->returns, "returns")) {
        Py_DECREF(in->returns);
        Py_DECREF(in->day_indices);
        in->returns = in->day_indices = NULL;
        return 0;
    }

    if (array_has_nan_or_inf(in->day_indices, "day_indices")) {
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


static int prepare_var_inputs_with_delta(PyObject* args, VarInputs* in, double* delta) {
    PyObject* returns_obj = NULL;
    PyObject* day_indices_obj = NULL;

    // Parse all arguments first
    if (!PyArg_ParseTuple(
        args, "O!O!nd",
        &PyArray_Type, &returns_obj,
        &PyArray_Type, &day_indices_obj,
        &in->n_days,
        delta
    )) {
        return 0;
    }

    // validate delta
    if (*delta <= 0.0) {
        PyErr_SetString(PyExc_ValueError, "delta must be positive");
        return 0;
    }

    // base args and call base prepare function
    PyObject* base_args = Py_BuildValue("OOn", returns_obj, day_indices_obj, in->n_days);
    if (!base_args) {
        return 0;
    }
    int result = prepare_var_inputs(base_args, in);
    Py_DECREF(base_args);

    return result;
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

    // can't access struct members without GIL
    // hence store local variable
    const npy_intp n = in.n;

    // do actual computation
    Py_BEGIN_ALLOW_THREADS
    for (npy_intp i = 0; i < n; i++) {
        int64_t day_idx = day_arr[i];
        double r = ret_arr[i];
        out_arr[day_idx] += r * r;
    }
    Py_END_ALLOW_THREADS

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

    if (in.n < 2) {
        PyErr_SetString(
            PyExc_ValueError,
            "bipower variance requires at least 2 observations"
        );
        free_var_inputs(&in);
        return NULL;
    }

    double* ret_arr = (double*) PyArray_DATA(in.returns);
    int64_t* day_arr = (int64_t*) PyArray_DATA(in.day_indices);
    double* out_arr = (double*) PyArray_DATA(in.output);

    const npy_intp n = in.n;
    const npy_intp n_days = in.n_days;

    // compute mu_1^(-2)
    const double mu_1 = mu_func(1.0);
    const double mu_1_inv_sq = 1.0 / (mu_1 * mu_1);

    Py_BEGIN_ALLOW_THREADS
    for (npy_intp i = 1; i < n; i++) {
        if (day_arr[i] == day_arr[i-1]) {
            int64_t day_idx = day_arr[i];
            out_arr[day_idx] += fabs(ret_arr[i-1]) * fabs(ret_arr[i]);
        }
    }

    // scale by mu_1^(-2)
    for (npy_intp i = 0; i < n_days; i++) {
        out_arr[i] *= mu_1_inv_sq;
    }
    Py_END_ALLOW_THREADS

    Py_DECREF(in.returns);
    Py_DECREF(in.day_indices);
    return (PyObject*) in.output;
}


// tri-power variance
static PyObject* compute_tripower_quarticity(PyObject* self, PyObject* args) {
    VarInputs in;;
    double delta;

    if (!prepare_var_inputs_with_delta(args, &in, &delta)) {
        return NULL;
    }

    if (in.n < 3) {
        PyErr_SetString(
            PyExc_ValueError,
            "tripower quarticity requires at least 3 observations"
        );
        free_var_inputs(&in);
        return NULL;
    }

    double* ret_arr = (double*) PyArray_DATA(in.returns);
    int64_t* day_arr = (int64_t*) PyArray_DATA(in.day_indices);
    double* out_arr = (double*) PyArray_DATA(in.output);

    const npy_intp n = in.n;
    const npy_intp n_days = in.n_days;
    const double exponent = 4.0 / 3.0;

    // compute mu_43^3 and scale
    const double mu_43 = mu_func(4.0 / 3.0);
    const double scale = delta * mu_43 * mu_43 * mu_43;

    Py_BEGIN_ALLOW_THREADS
    for (npy_intp i = 2; i < n; i++) {
        if (day_arr[i] == day_arr[i-1] && day_arr[i-1] == day_arr[i-2]) {
            int64_t day_idx = day_arr[i];
            double val = pow(fabs(ret_arr[i-2]), exponent)
                       * pow(fabs(ret_arr[i-1]), exponent)
                       * pow(fabs(ret_arr[i]), exponent);
            out_arr[day_idx] += val;
        }
    }

    for (npy_intp i = 0; i < n_days; i++) {
        out_arr[i] /= scale;
    }
    Py_END_ALLOW_THREADS

    Py_DECREF(in.returns);
    Py_DECREF(in.day_indices);
    return (PyObject*) in.output;
}


// z-statistic
static PyObject* compute_z_stats(PyObject* self, PyObject* args) {
    PyObject *realised_variance_obj = NULL;
    PyObject *bipower_variance_obj = NULL;
    PyObject *tripower_quarticity_obj = NULL;
    PyArrayObject *realised_variance = NULL;
    PyArrayObject *bipower_variance = NULL;
    PyArrayObject *tripower_quarticity = NULL;
    PyArrayObject *output = NULL;
    double delta;

    if (!PyArg_ParseTuple(
        args, "O!O!O!d",
        &PyArray_Type, &realised_variance_obj,
        &PyArray_Type, &bipower_variance_obj,
        &PyArray_Type, &tripower_quarticity_obj,
        &delta
    )) {
        return NULL;
    }

    if (delta <= 0.0) {
        PyErr_SetString(PyExc_ValueError, "delta must be positive");
        return NULL;
    }

    if (!npy_isfinite(delta)) {
        PyErr_SetString(PyExc_ValueError, "delta must be finite");
        return NULL;
    }

    realised_variance = (PyArrayObject*) PyArray_FROM_OTF(
        realised_variance_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY
    );
    if (realised_variance == NULL) {
        return NULL;
    }

    bipower_variance = (PyArrayObject*) PyArray_FROM_OTF(
        bipower_variance_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY
    );
    if (bipower_variance == NULL) {
        Py_DECREF(realised_variance);
        return NULL;
    }

    tripower_quarticity = (PyArrayObject*) PyArray_FROM_OTF(
        tripower_quarticity_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY
    );
    if (tripower_quarticity == NULL) {
        Py_DECREF(realised_variance);
        Py_DECREF(bipower_variance);
        return NULL;
    }

    if (
        PyArray_NDIM(realised_variance) != 1
        || PyArray_NDIM(bipower_variance) != 1
        || PyArray_NDIM(tripower_quarticity) != 1
    ) {
        PyErr_SetString(
            PyExc_ValueError,
            "realised_variance, bipower_variance, and tripower_quarticity must be 1-D arrays"
        );
        Py_DECREF(realised_variance);
        Py_DECREF(bipower_variance);
        Py_DECREF(tripower_quarticity);
        return NULL;
    }

    const npy_intp n = PyArray_SIZE(realised_variance);

    if (n == 0) {
        PyErr_SetString(PyExc_ValueError, "input arrays must not be empty");
        Py_DECREF(realised_variance);
        Py_DECREF(bipower_variance);
        Py_DECREF(tripower_quarticity);
        return NULL;
    }

    if (
        PyArray_SIZE(bipower_variance) != n
        || PyArray_SIZE(tripower_quarticity) != n
    ) {
        PyErr_SetString(
            PyExc_ValueError,
            "all input arrays must have the same length"
        );
        Py_DECREF(realised_variance);
        Py_DECREF(bipower_variance);
        Py_DECREF(tripower_quarticity);
        return NULL;
    }

    if (array_has_nan_or_inf(realised_variance, "realised_variance")) {
        Py_DECREF(realised_variance);
        Py_DECREF(bipower_variance);
        Py_DECREF(tripower_quarticity);
        return NULL;
    }
    if (array_has_nan_or_inf(bipower_variance, "bipower_variance")) {
        Py_DECREF(realised_variance);
        Py_DECREF(bipower_variance);
        Py_DECREF(tripower_quarticity);
        return NULL;
    }
    if (array_has_nan_or_inf(tripower_quarticity, "tripower_quarticity")) {
        Py_DECREF(realised_variance);
        Py_DECREF(bipower_variance);
        Py_DECREF(tripower_quarticity);
        return NULL;
    }

    npy_intp dims[1] = {n};
    output = (PyArrayObject*) PyArray_ZEROS(1, dims, NPY_DOUBLE, 0);
    if (output == NULL) {
        Py_DECREF(realised_variance);
        Py_DECREF(bipower_variance);
        Py_DECREF(tripower_quarticity);
        return NULL;
    }

    double* rv_arr = (double*) PyArray_DATA(realised_variance);
    double* bpv_arr = (double*) PyArray_DATA(bipower_variance);
    double* tpq_arr = (double*) PyArray_DATA(tripower_quarticity);
    double* out_arr = (double*) PyArray_DATA(output);

    const double const_term = (NPY_PI * NPY_PI) / 4 + NPY_PI - 5;

    Py_BEGIN_ALLOW_THREADS
    for (npy_intp i = 0; i < n; i++) {
        double max_func;
        double ratio;
        if (bpv_arr[i] != 0.0) {
            ratio = tpq_arr[i] / (bpv_arr[i] * bpv_arr[i]);
            max_func = fmax(1.0, ratio);
        }
        else {
            max_func = 1.0;
        }

        if (rv_arr[i] > 0.0) {
            out_arr[i] = ((rv_arr[i] - bpv_arr[i])
                       / (rv_arr[i] * sqrt(const_term * max_func)))
                       / sqrt(delta);
        }
        else {
            out_arr[i] = 0.0;
        }
    }
    Py_END_ALLOW_THREADS

    Py_DECREF(realised_variance);
    Py_DECREF(bipower_variance);
    Py_DECREF(tripower_quarticity);
    return (PyObject*) output;
}


// apply jumpy filter; compute continous and jump component
static PyObject* apply_jump_filter(PyObject* self, PyObject* args) {
    PyObject *realised_variance_obj = NULL;
    PyObject *bipower_variance_obj = NULL;
    PyObject *z_stats_obj = NULL;
    PyArrayObject *realised_variance = NULL;
    PyArrayObject *bipower_variance = NULL;
    PyArrayObject *z_stats = NULL;
    PyArrayObject *cont_out = NULL, *jump_out = NULL;
    double sig_threshold;
    npy_intp truncate_zero;

    if (!PyArg_ParseTuple(
        args, "O!O!O!dn",
        &PyArray_Type, &realised_variance_obj,
        &PyArray_Type, &bipower_variance_obj,
        &PyArray_Type, &z_stats_obj,
        &sig_threshold,
        &truncate_zero
    )) {
        return NULL;
    }

    if (sig_threshold <= 0.0) {
        PyErr_SetString(PyExc_ValueError, "sig_threshold must be greater than 0");
        return NULL;
    }

    if (!npy_isfinite(sig_threshold)) {
        PyErr_SetString(PyExc_ValueError, "sig_threshold must be finite");
        return NULL;
    }

    realised_variance = (PyArrayObject*) PyArray_FROM_OTF(
        realised_variance_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY
    );
    if (realised_variance == NULL) {
        return NULL;
    }

    bipower_variance = (PyArrayObject*) PyArray_FROM_OTF(
        bipower_variance_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY
    );
    if (bipower_variance == NULL) {
        Py_DECREF(realised_variance);
        return NULL;
    }

    z_stats = (PyArrayObject*) PyArray_FROM_OTF(
        z_stats_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY
    );
    if (z_stats == NULL) {
        Py_DECREF(realised_variance);
        Py_DECREF(bipower_variance);
        return NULL;
    }

    if (
        PyArray_NDIM(realised_variance) != 1
        || PyArray_NDIM(bipower_variance) != 1
        || PyArray_NDIM(z_stats) != 1
    ) {
        PyErr_SetString(
            PyExc_ValueError,
            "realised_variance, bipower_variance, and z_stats must be 1-D arrays"
        );
        Py_DECREF(realised_variance);
        Py_DECREF(bipower_variance);
        Py_DECREF(z_stats);
        return NULL;
    }

    const npy_intp n = PyArray_SIZE(realised_variance);

    if (n == 0) {
        PyErr_SetString(PyExc_ValueError, "input arrays must not be empty");
        Py_DECREF(realised_variance);
        Py_DECREF(bipower_variance);
        Py_DECREF(z_stats);
        return NULL;
    }

    if (
        PyArray_SIZE(bipower_variance) != n
        || PyArray_SIZE(z_stats) != n
    ) {
        PyErr_SetString(
            PyExc_ValueError,
            "all input arrays must have the same length"
        );
        Py_DECREF(realised_variance);
        Py_DECREF(bipower_variance);
        Py_DECREF(z_stats);
        return NULL;
    }

    if (array_has_nan_or_inf(realised_variance, "realised_variance")) {
        Py_DECREF(realised_variance);
        Py_DECREF(bipower_variance);
        Py_DECREF(z_stats);
        return NULL;
    }
    if (array_has_nan_or_inf(bipower_variance, "bipower_variance")) {
        Py_DECREF(realised_variance);
        Py_DECREF(bipower_variance);
        Py_DECREF(z_stats);
        return NULL;
    }
    if (array_has_nan_or_inf(z_stats, "z_stats")) {
        Py_DECREF(realised_variance);
        Py_DECREF(bipower_variance);
        Py_DECREF(z_stats);
        return NULL;
    }

    npy_intp dims[1] = {n};
    cont_out = (PyArrayObject*) PyArray_ZEROS(1, dims, NPY_DOUBLE, 0);
    jump_out = (PyArrayObject*) PyArray_ZEROS(1, dims, NPY_DOUBLE, 0);

    if (cont_out == NULL || jump_out == NULL) {
        Py_DECREF(realised_variance);
        Py_DECREF(bipower_variance);
        Py_DECREF(z_stats);

        // Clean up output arrays (might be NULL, might not be)
        // Use Py_XDECREF which safely handles NULL pointers
        Py_XDECREF(cont_out);
        Py_XDECREF(jump_out);
        return NULL;
    }

    double* rv_arr = (double*) PyArray_DATA(realised_variance);
    double* bpv_arr = (double*) PyArray_DATA(bipower_variance);
    double* z_arr = (double*) PyArray_DATA(z_stats);
    double* cont_arr = (double*) PyArray_DATA(cont_out);
    double* jump_arr = (double*) PyArray_DATA(jump_out);

    Py_BEGIN_ALLOW_THREADS
    for (npy_intp i = 0; i < n; i++) {
        double raw_jump = rv_arr[i] - bpv_arr[i];

        if (truncate_zero && raw_jump < 0.0) {
            raw_jump = 0.0;
        }

        if (z_arr[i] > sig_threshold) {
            jump_arr[i] = raw_jump;
        }
        else {
            jump_arr[i] = 0.0;
        }

        cont_arr[i] = rv_arr[i] - jump_arr[i];
    }
    Py_END_ALLOW_THREADS

    PyObject* result = Py_BuildValue("(OO)", cont_out, jump_out);
    Py_DECREF(realised_variance);
    Py_DECREF(bipower_variance);
    Py_DECREF(z_stats);
    Py_DECREF(cont_out);  // Py_BuildValue increases refcount
    Py_DECREF(jump_out);
    return result;
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
    {
        "compute_z_stats",
        compute_z_stats,
        METH_VARARGS,
        "Compute the z-statistics for jump"
    },
    {
        "apply_jump_filter",
        apply_jump_filter,
        METH_VARARGS,
        "Apply the filter fo jump significance"
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
