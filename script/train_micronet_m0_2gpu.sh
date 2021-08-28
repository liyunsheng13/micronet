export DATA_PATH=$1/imagenet
export OUTPUT_PATH=$2/micronet-m0-2gpu

CUDA_VISIBLE_DEVICES=0,1 python main.py --arch MicroNet -d $DATA_PATH --epochs 600 --lr-decay cos --lr 0.1 --wd 3e-5 \
                                                         -c $OUTPUT_PATH -j 48 --input-size 224 --label-smoothing 0.0 -b 256 \
                                                         MODEL.MICRONETS.BLOCK DYMicroBlock \
                                                         MODEL.MICRONETS.NET_CONFIG msnx_dy6_exp4_4M_221 \
                                                         MODEL.MICRONETS.STEM_CH 4 \
                                                         MODEL.MICRONETS.STEM_GROUPS 2,2 \
                                                         MODEL.MICRONETS.STEM_DILATION 1 \
                                                         MODEL.MICRONETS.STEM_MODE spatialsepsf \
                                                         MODEL.MICRONETS.OUT_CH 640 \
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
