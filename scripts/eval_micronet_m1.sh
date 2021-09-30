export DATA_PATH=$1/imagenet
export OUTPUT_PATH=$2/micronet-m1-eval
export WEIGHT_PATH=$3

CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --arch MicroNet -d $DATA_PATH -c $OUTPUT_PATH -j 48 --input-size 224 -b 512 -e --weight $WEIGHT_PATH \
                                                         MODEL.MICRONETS.BLOCK DYMicroBlock \
                                                         MODEL.MICRONETS.NET_CONFIG msnx_dy6_exp6_6M_221 \
                                                         MODEL.MICRONETS.STEM_CH 6 \
                                                         MODEL.MICRONETS.STEM_GROUPS 3,2 \
                                                         MODEL.MICRONETS.STEM_DILATION 1 \
                                                         MODEL.MICRONETS.STEM_MODE spatialsepsf \
                                                         MODEL.MICRONETS.OUT_CH 960 \
                                                         MODEL.MICRONETS.DEPTHSEP True \
                                                         MODEL.MICRONETS.POINTWISE group \
                                                         MODEL.MICRONETS.DROPOUT 0.05 \
                                                         MODEL.ACTIVATION.MODULE DYShiftMax \
                                                         MODEL.ACTIVATION.ACT_MAX 2.0 \
                                                         MODEL.ACTIVATION.LINEARSE_BIAS False \
                                                         MODEL.ACTIVATION.INIT_A_BLOCK3 1.0,0.0 \
                                                         MODEL.ACTIVATION.INIT_A 1.0,1.0 \
                                                         MODEL.ACTIVATION.INIT_B 0.0,0.0 \
                                                         MODEL.ACTIVATION.REDUCTION 8 \
                                                         MODEL.MICRONETS.SHUFFLE True \
