#include <Python.h>
#include <fstream>

static void insert_key(PyObject* dict, const char* key, const float* value, Py_ssize_t len)
{
    auto pykey = PyUnicode_FromString(key);
    auto pytuple = PyTuple_New(len);
    for (Py_ssize_t i = 0; i < len; ++i)
    {
        PyTuple_SetItem(pytuple, i, PyFloat_FromDouble(value[i]));
    }
    PyDict_SetItem(dict, pykey, pytuple);
}

static PyObject* load_bin_wrap(PyObject* self, PyObject* args)
{
    const char* path = nullptr;
    PyArg_ParseTuple(args, "s", &path);
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs.is_open())
    {
        Py_RETURN_NONE;
    }
    float identity[75];
    float rotation[4];
    float translation[3];
    float expression[46];
    float landmark[75 * 2];
    ifs.read((char*)&identity[0], 75 * sizeof(float));
    ifs.read((char*)&rotation[0], 4 * sizeof(float));
    ifs.read((char*)&translation[0], 3 * sizeof(float));
    ifs.read((char*)&expression[0], 46 * sizeof(float));
    ifs.read((char*)&landmark[0], 75 * 2 * sizeof(float));
    ifs.close();
    auto tbl = PyDict_New();
    insert_key(tbl, "identity", identity, 75);
    insert_key(tbl, "rotation", rotation, 4);
    insert_key(tbl, "translation", translation, 3);
    insert_key(tbl, "expression", expression, 46);
    insert_key(tbl, "landmark", landmark, 75 * 2);
    return tbl;
}

static PyMethodDef methodList[] =
{
    {
        "load_bin",
        load_bin_wrap,
        METH_VARARGS,
        ""
    },
    { NULL, NULL, 0, NULL }
};

#if PY_MAJOR_VERSION >= 3
static PyModuleDef moduleDef =
{
    PyModuleDef_HEAD_INIT,
    "dde_bin_loader", "",
    -1,
    methodList
};

PyMODINIT_FUNC PyInit_dde_bin_loader(void) {
    return PyModule_Create(&moduleDef);
}
#else
PyMODINIT_FUNC initdde_bin_loader(void) {
    (void)Py_InitModule("dde_bin_loader", methodList);
}
#endif
