# cuda
export PATH=/usr/local/cuda/bin:$PATH
export LIBRARY_PATH=/usr/local/cuda/lib64:$LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export CUDA_ROOT=/usr/local/cuda/
export CUDA_HOME=/usr/local/cuda/
export CUDA_BIN_PATH=/usr/local/cuda/bin
#
#
#
export CUDNN_LIB_DIR=/data/zhijie/vision_transformer/cuda/lib64
export CUDNN_INCLUDE_DIR=/data/zhijie/vision_transformer/cuda/include
#
# # cudnn
export LD_LIBRARY_PATH=/data/zhijie/vision_transformer/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64/:$LD_LIBRARY_PATH
export LIBRARY_PATH=$LIBRARY_PATH:/data/zhijie/vision_transformer/cuda/lib64
export CPATH=$CPATH:/data/zhijie/vision_transformer/cuda/include
