/* BOOST */
#define BOOST_PYTHON_STATIC_LIB
#include <boost/python.hpp>
using namespace boost::python;

/* NUMPY */
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/ndarrayobject.h>

#include <iostream>
#include <vector>

#include "CMatrix.h"
#include "Show.h"

#include "NMath.h"
#include "NRBM.h"
#include "CTMesh.h"
#include "onlyDefines.h"
#include "paramMap.h"

#define SMOOTHMODEL true

template<class T>
inline T* make_shift(T *in_ptr, size_t in_shift)
{
    return (T*)((char*)in_ptr + in_shift);
}

struct PitchedPtr
{
    size_t num_rows;
    size_t num_cols;
    size_t pitch;
    double *ptr;

    double *get_row(size_t in_row)
    {
        return make_shift<double>(ptr, in_row * pitch);
    }

    double *get_elem(size_t in_row, size_t in_column)
    {
        return get_row(in_row) + in_column;
    }
};

void process(PitchedPtr &in_pose_params, PitchedPtr &in_shape_params,
        PitchedPtr &in_evectors, std::string &in_filename,
        PitchedPtr &out_points, PitchedPtr &out_joints)
{
    size_t num_motion_params = in_pose_params.num_cols;
    CVector<double> mParams(num_motion_params);
    for (size_t idx = 0; idx < num_motion_params; ++idx)
        mParams(idx) = *in_pose_params.get_elem(0, idx); 

    size_t num_shape_params = in_shape_params.num_cols;
    CVector<double> shapeParams(num_shape_params);
    for (size_t idx = 0; idx < num_shape_params; ++idx)
        shapeParams[idx] = *in_shape_params.get_elem(0, idx);

    CVector<float> shapeParamsFloat(num_shape_params);
    for (size_t idx = 0; idx < num_shape_params; ++idx)
        shapeParamsFloat(idx) = float(shapeParams(idx));

    // Read object model
    CMatrix<float> mRBM(4, 4);
    NRBM::RVT2RBM(&mParams, mRBM);
    CMesh initMesh;

    initMesh.readModel(in_filename.c_str(), SMOOTHMODEL);
    initMesh.updateJntPos();

    initMesh.centerModel();

    // use eingevectors passed as an argument
    initMesh.readShapeSpaceEigens(in_evectors.ptr, num_shape_params,
            initMesh.GetPointSize());

    // reshape the model
    initMesh.shapeChangesToMesh(shapeParamsFloat);

    // update joints
    initMesh.updateJntPos();

    CVector<CMatrix<float> > M(initMesh.joints() + 1);
    CVector<float> TW(initMesh.joints() + 6);
    
    for (int j = 6; j < mParams.size(); ++j)
        TW(j) = (float)mParams(j);

    initMesh.angleToMatrix(mRBM, TW, M);

    // rotate joints
    initMesh.rigidMotion(M, TW, true, true);

    // Fill in resulting joint array
    size_t nJoints = initMesh.joints();
    for (size_t i = 0; i < nJoints; ++i)
    {
        CJoint joint = initMesh.joint(i+1);
        double *row = out_joints.get_row(i);
        row[0] = i + 1;
        row[1] = joint.getDirection()[0];
        row[2] = joint.getDirection()[1];
        row[3] = joint.getDirection()[2];
        row[4] = joint.getPoint()[0];
        row[5] = joint.getPoint()[1];
        row[6] = joint.getPoint()[2];
        row[7] = double(joint.mParent);
    }

    // Fill in resulting points array
    size_t nPoints = initMesh.GetPointSize();
    for (size_t i = 0; i < nPoints; ++i)
    {
        float x, y, z;
        initMesh.GetPoint(i, x, y, z);
        *out_points.get_elem(i, 0) = x;
        *out_points.get_elem(i, 1) = y;
        *out_points.get_elem(i, 2) = z;
    }
}

void parseDoubleParam(const object &in_param, PitchedPtr &out_matrix)
{
    PyArrayObject *param_obj = (PyArrayObject*)in_param.ptr();
    PyArray_Descr *param_descr = PyArray_DESCR(param_obj);
    if (param_descr->type != 'd')
        throw("Wrong ndarray type, 64bit floating point");
    int dims = PyArray_NDIM(param_obj);
    if (dims != 2)
        throw("Wrong number of dimensions");
    size_t row_stride = PyArray_STRIDE(param_obj, 0);
    size_t col_stride = PyArray_STRIDE(param_obj, 1);
    if (row_stride < col_stride)
        throw("Wrong ndarray format. It should have C-contiguous order.");
    out_matrix.num_rows = PyArray_DIMS(param_obj)[0];
    out_matrix.num_cols = PyArray_DIMS(param_obj)[1];
    out_matrix.ptr = static_cast<double*>(PyArray_DATA(param_obj));
    out_matrix.pitch = row_stride;
}

void constructDoubleMatrix(size_t in_num_rows, size_t in_num_cols,
       PitchedPtr &io_matrix, object &out_p_matrix)
{
    Py_intptr_t shape[2] = {in_num_rows, in_num_cols};
    PyObject *pyObj = PyArray_SimpleNew(2, shape, NPY_DOUBLE);
    // construct boost object
    out_p_matrix = object(handle<>(pyObj));

    // interpret pyObject as array
    PyArrayObject *obj = (PyArrayObject*)pyObj;

    io_matrix.num_rows = in_num_rows;
    io_matrix.num_cols = in_num_cols;
    io_matrix.pitch = PyArray_STRIDE(obj, 0);
    io_matrix.ptr = static_cast<double*>(PyArray_DATA(obj));
}

object shapepose(const object &in_pose_params, const object &in_shape_params,
       const object &in_evectors, std::string in_model_filename)
{
    PitchedPtr pose_params;
    PitchedPtr shape_params;
    PitchedPtr evectors;

    PitchedPtr point_array;
    PitchedPtr joint_array;

    // makes copy of the array. In process function copy is performed too. It should be fixed
    parseDoubleParam(in_pose_params, pose_params);
    parseDoubleParam(in_shape_params, shape_params);
    parseDoubleParam(in_evectors, evectors);

    if (*in_model_filename.rbegin() != '/')
        in_model_filename.append("/");
    in_model_filename.append("model.dat");

    object point_param;
    object joint_param;
    constructDoubleMatrix(6449, 3, point_array, point_param);
    constructDoubleMatrix(25, 8, joint_array, joint_param);

    process(pose_params, shape_params, evectors, in_model_filename, point_array, joint_array);

    boost::python::tuple res = boost::python::make_tuple(point_param, joint_param);
    return res;
}

BOOST_PYTHON_MODULE(shapemodel)
{
    import_array();

    def("shapepose", shapepose);
}
