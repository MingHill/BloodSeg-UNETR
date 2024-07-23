2024-06-20 21:01:07,584 - INFO - {'BATCH_SIZE': 4096,
 'COMMENT': 'Run 09',
 'DATASOURCE': '/home/hmgillis/datasets/third-party/pytorch-datasets',
 'LEARNING_RATE': 0.001,
 'LOG_DIR': 'logs/cifar/09/',
 'NUM_EPOCHS': 100,
 'RANDOM_SEED': 42,
 'SAVE_MODELS': 1,
 'SCHEDULER': 0,
 'WEIGHT_DECAY': 0.0}
2024-06-20 21:01:14,579 - INFO - Train -> 40000, 10 (samples, batches)
2024-06-20 21:01:14,579 - INFO - Valid -> 10000, 3 (samples, batches)
2024-06-20 21:01:14,579 - INFO - Test  -> 10000,  3  (samples, batches)
2024-06-20 21:01:14,683 - INFO - ViTMAEForPreTraining(
  (vit): ViTMAEModel(
    (embeddings): ViTMAEEmbeddings(
      (patch_embeddings): ViTMAEPatchEmbeddings(
        (projection): Conv2d(3, 128, kernel_size=(4, 4), stride=(4, 4))
      )
    )
    (encoder): ViTMAEEncoder(
      (layer): ModuleList(
        (0-5): 6 x ViTMAELayer(
          (attention): ViTMAESdpaAttention(
            (attention): ViTMAESdpaSelfAttention(
              (query): Linear(in_features=128, out_features=128, bias=True)
              (key): Linear(in_features=128, out_features=128, bias=True)
              (value): Linear(in_features=128, out_features=128, bias=True)
              (dropout): Dropout(p=0.0, inplace=False)
            )
            (output): ViTMAESelfOutput(
              (dense): Linear(in_features=128, out_features=128, bias=True)
              (dropout): Dropout(p=0.0, inplace=False)
            )
          )
          (intermediate): ViTMAEIntermediate(
            (dense): Linear(in_features=128, out_features=256, bias=True)
            (intermediate_act_fn): GELUActivation()
          )
          (output): ViTMAEOutput(
            (dense): Linear(in_features=256, out_features=128, bias=True)
            (dropout): Dropout(p=0.0, inplace=False)
          )
          (layernorm_before): LayerNorm((128,), eps=1e-06, elementwise_affine=True)
          (layernorm_after): LayerNorm((128,), eps=1e-06, elementwise_affine=True)
        )
      )
    )
    (layernorm): LayerNorm((128,), eps=1e-06, elementwise_affine=True)
  )
  (decoder): ViTMAEDecoder(
    (decoder_embed): Linear(in_features=128, out_features=64, bias=True)
    (decoder_layers): ModuleList(
      (0-1): 2 x ViTMAELayer(
        (attention): ViTMAESdpaAttention(
          (attention): ViTMAESdpaSelfAttention(
            (query): Linear(in_features=64, out_features=64, bias=True)
            (key): Linear(in_features=64, out_features=64, bias=True)
            (value): Linear(in_features=64, out_features=64, bias=True)
            (dropout): Dropout(p=0.0, inplace=False)
          )
          (output): ViTMAESelfOutput(
            (dense): Linear(in_features=64, out_features=64, bias=True)
            (dropout): Dropout(p=0.0, inplace=False)
          )
        )
        (intermediate): ViTMAEIntermediate(
          (dense): Linear(in_features=64, out_features=128, bias=True)
          (intermediate_act_fn): GELUActivation()
        )
        (output): ViTMAEOutput(
          (dense): Linear(in_features=128, out_features=64, bias=True)
          (dropout): Dropout(p=0.0, inplace=False)
        )
        (layernorm_before): LayerNorm((64,), eps=1e-06, elementwise_affine=True)
        (layernorm_after): LayerNorm((64,), eps=1e-06, elementwise_affine=True)
      )
    )
    (decoder_norm): LayerNorm((64,), eps=1e-06, elementwise_affine=True)
    (decoder_pred): Linear(in_features=64, out_features=48, bias=True)
  )
)
2024-06-20 21:01:14,685 - INFO - ViTMAEConfig {
  "attention_probs_dropout_prob": 0.0,
  "decoder_hidden_size": 64,
  "decoder_intermediate_size": 128,
  "decoder_num_attention_heads": 4,
  "decoder_num_hidden_layers": 2,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.0,
  "hidden_size": 128,
  "image_size": 48,
  "initializer_range": 0.02,
  "intermediate_size": 256,
  "layer_norm_eps": 1e-06,
  "mask_ratio": 0.75,
  "model_type": "vit_mae",
  "norm_pix_loss": true,
  "num_attention_heads": 4,
  "num_channels": 3,
  "num_hidden_layers": 6,
  "patch_size": 4,
  "qkv_bias": true,
  "transformers_version": "4.41.2"
}

2024-06-20 21:01:14,685 - INFO - Parameters: 907888
2024-06-20 21:01:14,685 - INFO - Trainable parameters: 880048

2024-06-20 21:01:14,686 - INFO - Scheduler: None
2024-06-20 21:01:18,147 - INFO - Epoch 1/100 - Train Loss: 0.9296, Valid Loss: 0.9307, 
2024-06-20 21:01:21,407 - INFO - Epoch 2/100 - Train Loss: 0.9242, Valid Loss: 0.9253, 
2024-06-20 21:01:24,671 - INFO - Epoch 3/100 - Train Loss: 0.9171, Valid Loss: 0.9180, 
2024-06-20 21:01:27,936 - INFO - Epoch 4/100 - Train Loss: 0.8950, Valid Loss: 0.8945, 
2024-06-20 21:01:31,203 - INFO - Epoch 5/100 - Train Loss: 0.8614, Valid Loss: 0.8614, 
2024-06-20 21:01:34,467 - INFO - Epoch 6/100 - Train Loss: 0.7866, Valid Loss: 0.7859, 
2024-06-20 21:01:37,736 - INFO - Epoch 7/100 - Train Loss: 0.7590, Valid Loss: 0.7578, 
2024-06-20 21:01:41,005 - INFO - Epoch 8/100 - Train Loss: 0.7602, Valid Loss: 0.7570, 
2024-06-20 21:01:44,274 - INFO - Epoch 9/100 - Train Loss: 0.7266, Valid Loss: 0.7225, 
2024-06-20 21:01:47,544 - INFO - Epoch 10/100 - Train Loss: 0.7152, Valid Loss: 0.7127, 
2024-06-20 21:01:50,805 - INFO - Epoch 11/100 - Train Loss: 0.7115, Valid Loss: 0.7085, 
2024-06-20 21:01:54,072 - INFO - Epoch 12/100 - Train Loss: 0.7112, Valid Loss: 0.7079, 
2024-06-20 21:01:57,341 - INFO - Epoch 13/100 - Train Loss: 0.7289, Valid Loss: 0.7260, 
2024-06-20 21:02:00,610 - INFO - Epoch 14/100 - Train Loss: 0.7092, Valid Loss: 0.7062, 
2024-06-20 21:02:03,880 - INFO - Epoch 15/100 - Train Loss: 0.7056, Valid Loss: 0.7030, 
2024-06-20 21:02:07,150 - INFO - Epoch 16/100 - Train Loss: 0.7036, Valid Loss: 0.7006, 
2024-06-20 21:02:10,416 - INFO - Epoch 17/100 - Train Loss: 0.7025, Valid Loss: 0.6995, 
2024-06-20 21:02:13,684 - INFO - Epoch 18/100 - Train Loss: 0.7025, Valid Loss: 0.6989, 
2024-06-20 21:02:16,955 - INFO - Epoch 19/100 - Train Loss: 0.7024, Valid Loss: 0.6992, 
2024-06-20 21:02:20,225 - INFO - Epoch 20/100 - Train Loss: 0.7035, Valid Loss: 0.7004, 
2024-06-20 21:02:23,495 - INFO - Epoch 21/100 - Train Loss: 0.6941, Valid Loss: 0.6909, 
2024-06-20 21:02:26,759 - INFO - Epoch 22/100 - Train Loss: 0.6885, Valid Loss: 0.6843, 
2024-06-20 21:02:30,028 - INFO - Epoch 23/100 - Train Loss: 0.6769, Valid Loss: 0.6732, 
2024-06-20 21:02:33,299 - INFO - Epoch 24/100 - Train Loss: 0.6791, Valid Loss: 0.6754, 
2024-06-20 21:02:36,570 - INFO - Epoch 25/100 - Train Loss: 0.6811, Valid Loss: 0.6771, 
2024-06-20 21:02:39,841 - INFO - Epoch 26/100 - Train Loss: 0.6712, Valid Loss: 0.6672, 
2024-06-20 21:02:43,103 - INFO - Epoch 27/100 - Train Loss: 0.6752, Valid Loss: 0.6711, 
2024-06-20 21:02:46,365 - INFO - Epoch 28/100 - Train Loss: 0.6727, Valid Loss: 0.6682, 
2024-06-20 21:02:49,635 - INFO - Epoch 29/100 - Train Loss: 0.6592, Valid Loss: 0.6553, 
2024-06-20 21:02:52,907 - INFO - Epoch 30/100 - Train Loss: 0.6523, Valid Loss: 0.6484, 
2024-06-20 21:02:56,180 - INFO - Epoch 31/100 - Train Loss: 0.6648, Valid Loss: 0.6608, 
2024-06-20 21:02:59,452 - INFO - Epoch 32/100 - Train Loss: 0.6477, Valid Loss: 0.6433, 
2024-06-20 21:03:02,718 - INFO - Epoch 33/100 - Train Loss: 0.6400, Valid Loss: 0.6357, 
2024-06-20 21:03:05,988 - INFO - Epoch 34/100 - Train Loss: 0.6410, Valid Loss: 0.6369, 
2024-06-20 21:03:09,260 - INFO - Epoch 35/100 - Train Loss: 0.6348, Valid Loss: 0.6303, 
2024-06-20 21:03:12,531 - INFO - Epoch 36/100 - Train Loss: 0.6416, Valid Loss: 0.6372, 
2024-06-20 21:03:15,804 - INFO - Epoch 37/100 - Train Loss: 0.6385, Valid Loss: 0.6341, 
2024-06-20 21:03:19,067 - INFO - Epoch 38/100 - Train Loss: 0.6318, Valid Loss: 0.6274, 
2024-06-20 21:03:22,335 - INFO - Epoch 39/100 - Train Loss: 0.6358, Valid Loss: 0.6308, 
2024-06-20 21:03:25,607 - INFO - Epoch 40/100 - Train Loss: 0.6299, Valid Loss: 0.6252, 
2024-06-20 21:03:28,880 - INFO - Epoch 41/100 - Train Loss: 0.6439, Valid Loss: 0.6387, 
2024-06-20 21:03:32,153 - INFO - Epoch 42/100 - Train Loss: 0.6547, Valid Loss: 0.6498, 
2024-06-20 21:03:35,426 - INFO - Epoch 43/100 - Train Loss: 0.6504, Valid Loss: 0.6464, 
2024-06-20 21:03:38,693 - INFO - Epoch 44/100 - Train Loss: 0.6297, Valid Loss: 0.6248, 
2024-06-20 21:03:41,963 - INFO - Epoch 45/100 - Train Loss: 0.6277, Valid Loss: 0.6234, 
2024-06-20 21:03:45,236 - INFO - Epoch 46/100 - Train Loss: 0.6248, Valid Loss: 0.6207, 
2024-06-20 21:03:48,509 - INFO - Epoch 47/100 - Train Loss: 0.6210, Valid Loss: 0.6166, 
2024-06-20 21:03:51,782 - INFO - Epoch 48/100 - Train Loss: 0.6179, Valid Loss: 0.6139, 
2024-06-20 21:03:55,046 - INFO - Epoch 49/100 - Train Loss: 0.6172, Valid Loss: 0.6127, 
2024-06-20 21:03:58,313 - INFO - Epoch 50/100 - Train Loss: 0.6196, Valid Loss: 0.6149, 
2024-06-20 21:04:01,586 - INFO - Epoch 51/100 - Train Loss: 0.6165, Valid Loss: 0.6119, 
2024-06-20 21:04:04,859 - INFO - Epoch 52/100 - Train Loss: 0.6218, Valid Loss: 0.6177, 
2024-06-20 21:04:08,132 - INFO - Epoch 53/100 - Train Loss: 0.6003, Valid Loss: 0.5957, 
2024-06-20 21:04:11,406 - INFO - Epoch 54/100 - Train Loss: 0.5807, Valid Loss: 0.5764, 
2024-06-20 21:04:14,673 - INFO - Epoch 55/100 - Train Loss: 0.6060, Valid Loss: 0.6019, 
2024-06-20 21:04:17,944 - INFO - Epoch 56/100 - Train Loss: 0.6018, Valid Loss: 0.5980, 
2024-06-20 21:04:21,217 - INFO - Epoch 57/100 - Train Loss: 0.5747, Valid Loss: 0.5708, 
2024-06-20 21:04:24,490 - INFO - Epoch 58/100 - Train Loss: 0.5940, Valid Loss: 0.5898, 
2024-06-20 21:04:27,763 - INFO - Epoch 59/100 - Train Loss: 0.6002, Valid Loss: 0.5959, 
2024-06-20 21:04:31,027 - INFO - Epoch 60/100 - Train Loss: 0.5639, Valid Loss: 0.5600, 
2024-06-20 21:04:34,296 - INFO - Epoch 61/100 - Train Loss: 0.5666, Valid Loss: 0.5628, 
2024-06-20 21:04:37,568 - INFO - Epoch 62/100 - Train Loss: 0.5546, Valid Loss: 0.5510, 
2024-06-20 21:04:40,842 - INFO - Epoch 63/100 - Train Loss: 0.5485, Valid Loss: 0.5449, 
2024-06-20 21:04:44,116 - INFO - Epoch 64/100 - Train Loss: 0.5451, Valid Loss: 0.5410, 
2024-06-20 21:04:47,389 - INFO - Epoch 65/100 - Train Loss: 0.5595, Valid Loss: 0.5555, 
2024-06-20 21:04:50,656 - INFO - Epoch 66/100 - Train Loss: 0.5395, Valid Loss: 0.5354, 
2024-06-20 21:04:53,929 - INFO - Epoch 67/100 - Train Loss: 0.5443, Valid Loss: 0.5399, 
2024-06-20 21:04:57,202 - INFO - Epoch 68/100 - Train Loss: 0.5488, Valid Loss: 0.5441, 
2024-06-20 21:05:00,475 - INFO - Epoch 69/100 - Train Loss: 0.5395, Valid Loss: 0.5361, 
2024-06-20 21:05:03,749 - INFO - Epoch 70/100 - Train Loss: 0.5310, Valid Loss: 0.5274, 
2024-06-20 21:05:07,013 - INFO - Epoch 71/100 - Train Loss: 0.5581, Valid Loss: 0.5532, 
2024-06-20 21:05:10,282 - INFO - Epoch 72/100 - Train Loss: 0.5344, Valid Loss: 0.5300, 
2024-06-20 21:05:13,555 - INFO - Epoch 73/100 - Train Loss: 0.5294, Valid Loss: 0.5255, 
2024-06-20 21:05:16,829 - INFO - Epoch 74/100 - Train Loss: 0.5432, Valid Loss: 0.5385, 
2024-06-20 21:05:20,103 - INFO - Epoch 75/100 - Train Loss: 0.5305, Valid Loss: 0.5261, 
2024-06-20 21:05:23,377 - INFO - Epoch 76/100 - Train Loss: 0.5258, Valid Loss: 0.5220, 
2024-06-20 21:05:26,644 - INFO - Epoch 77/100 - Train Loss: 0.5333, Valid Loss: 0.5290, 
2024-06-20 21:05:29,916 - INFO - Epoch 78/100 - Train Loss: 0.5268, Valid Loss: 0.5229, 
2024-06-20 21:05:33,190 - INFO - Epoch 79/100 - Train Loss: 0.5259, Valid Loss: 0.5219, 
2024-06-20 21:05:36,465 - INFO - Epoch 80/100 - Train Loss: 0.5316, Valid Loss: 0.5274, 
2024-06-20 21:05:39,739 - INFO - Epoch 81/100 - Train Loss: 0.5260, Valid Loss: 0.5223, 
2024-06-20 21:05:43,003 - INFO - Epoch 82/100 - Train Loss: 0.5300, Valid Loss: 0.5262, 
2024-06-20 21:05:46,273 - INFO - Epoch 83/100 - Train Loss: 0.5253, Valid Loss: 0.5210, 
2024-06-20 21:05:49,547 - INFO - Epoch 84/100 - Train Loss: 0.5233, Valid Loss: 0.5192, 
2024-06-20 21:05:52,821 - INFO - Epoch 85/100 - Train Loss: 0.5159, Valid Loss: 0.5119, 
2024-06-20 21:05:56,096 - INFO - Epoch 86/100 - Train Loss: 0.5092, Valid Loss: 0.5048, 
2024-06-20 21:05:59,369 - INFO - Epoch 87/100 - Train Loss: 0.5116, Valid Loss: 0.5075, 
2024-06-20 21:06:02,637 - INFO - Epoch 88/100 - Train Loss: 0.5022, Valid Loss: 0.4983, 
2024-06-20 21:06:05,910 - INFO - Epoch 89/100 - Train Loss: 0.4942, Valid Loss: 0.4901, 
2024-06-20 21:06:09,183 - INFO - Epoch 90/100 - Train Loss: 0.4906, Valid Loss: 0.4867, 
2024-06-20 21:06:12,457 - INFO - Epoch 91/100 - Train Loss: 0.4946, Valid Loss: 0.4904, 
2024-06-20 21:06:15,731 - INFO - Epoch 92/100 - Train Loss: 0.4794, Valid Loss: 0.4754, 
2024-06-20 21:06:18,996 - INFO - Epoch 93/100 - Train Loss: 0.4895, Valid Loss: 0.4848, 
2024-06-20 21:06:22,265 - INFO - Epoch 94/100 - Train Loss: 0.4761, Valid Loss: 0.4719, 
2024-06-20 21:06:25,539 - INFO - Epoch 95/100 - Train Loss: 0.4769, Valid Loss: 0.4733, 
2024-06-20 21:06:28,814 - INFO - Epoch 96/100 - Train Loss: 0.4666, Valid Loss: 0.4634, 
2024-06-20 21:06:32,087 - INFO - Epoch 97/100 - Train Loss: 0.4690, Valid Loss: 0.4658, 
2024-06-20 21:06:35,361 - INFO - Epoch 98/100 - Train Loss: 0.4712, Valid Loss: 0.4678, 
2024-06-20 21:06:38,628 - INFO - Epoch 99/100 - Train Loss: 0.4693, Valid Loss: 0.4657, 
2024-06-20 21:06:41,901 - INFO - Epoch 100/100 - Train Loss: 0.4669, Valid Loss: 0.4631, 
2024-06-20 21:06:42,185 - INFO - Test Loss: 0.5491
2024-06-20 21:06:42,185 - INFO - Elapsed time -> 5.5769 minutes
