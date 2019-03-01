// See docs in ../ops/image_ops.cc.

#define EIGEN_USE_THREADS

#include <vector>
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

using shape_inference::InferenceContext;

REGISTER_OP("ExtractPatches")
    .Input("input: float")
    .Input("size: int32")
    .Input("offsets: float")
    .Output("patches: float")
    .SetShapeFn([](InferenceContext* c) {
        return Status::OK();
    });

class ExtractPatchesOp : public OpKernel {
 public:
  explicit ExtractPatchesOp(OpKernelConstruction* context) : OpKernel(context) {

  }

  // Expect input tensor of rank 4 with dimensions (batch_size, height, width,
  // depth).
  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    const TensorShape input_shape = input.shape();
    const int32 num_dims = input_shape.dims();
    OP_REQUIRES(
        context, num_dims == 4,
        errors::InvalidArgument(
            "input must be 4-dimensional (batch_size, height, width, depth)",
            input_shape.DebugString()));

    const int32 batch_size = input_shape.dim_size(0);
    const int32 input_height = input_shape.dim_size(1);
    const int32 input_width = input_shape.dim_size(2);
    const int32 depth = input_shape.dim_size(3);

    const Tensor& window_size = context->input(1);
    OP_REQUIRES(context, (window_size.shape().dims() == 1) &&
                             window_size.shape().dim_size(0) == 2,
                errors::InvalidArgument(
                    "patch shape must be a vector of size 2 (height, width)",
                    window_size.shape().DebugString()));

    const int32 patch_height = window_size.tensor<int, 1>()(0);
    const int32 patch_width = window_size.tensor<int, 1>()(1);
    
    const Tensor& offsets = context->input(2);
    OP_REQUIRES(context, offsets.shape().dims() == 3,
                errors::InvalidArgument("input must be a tensor [batch_size, num_patches, 2]",
                                        offsets.shape().DebugString()));
    OP_REQUIRES(context, offsets.shape().dim_size(0) == batch_size,
                errors::InvalidArgument("first dimension should be batch",
                                        offsets.shape().DebugString()));
    OP_REQUIRES(
        context, offsets.shape().dim_size(2) == 2,
        errors::InvalidArgument("third dimension should be of size 2 (y,x)",
                                offsets.shape().DebugString()));

    auto num_patches = offsets.shape().dim_size(1);
    TensorShape output_shape({batch_size, num_patches, patch_height, patch_width, depth});

    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));
    if (output->NumElements() == 0) {
      // Nothing else to do.
      return;
    }

    typename TTypes<float, 5>::Tensor output_patches = output->tensor<float, 5>();
    typename TTypes<float, 4>::ConstTensor input_images = input.tensor<float, 4>();

    for (int i = 0; i < batch_size; ++i) {
      for (int n = 0; n < num_patches; ++n) {
        const float offset_y = offsets.tensor<float, 3>()(i, n, 0);
        const float offset_x = offsets.tensor<float, 3>()(i, n, 1);

        for(int source_x=offset_x-patch_width/2, target_x=0;
                target_x < patch_width;
                ++source_x, ++target_x) {
          for(int source_y=offset_y-patch_height/2, target_y=0;
                target_y < patch_height;
                ++source_y, ++target_y) {
            if (source_x > 0 && source_x < input_width && source_y > 0 && source_y < input_height) {
              for (int c = 0; c < depth; ++c) {
                output_patches(i, n, target_y, target_x, c) = input_images(i, source_y, source_x, c);
              }
            }
          }
        }
      }
    }

  }

};

REGISTER_KERNEL_BUILDER(Name("ExtractPatches").Device(DEVICE_CPU),
                        ExtractPatchesOp);

}  // end namespace tensorflow
