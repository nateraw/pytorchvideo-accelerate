# pytorchvideo-accelerate

Distributed training of video action recognition models with pytorchvideo and Hugging Face accelerate

Currently just working with the Kinetics 700 dataset, but will add a dataset or two for fine-tuning as well.

Uses `accelerate` for distributed training, `wandb` for logging, `torchmetrics` for metrics, `pytorchvideo` for models/datasets/transforms.

## Usage

1. Clone the repo, `cd` into it. Then `pip install -r requirements.txt`.

2. Then, `accelerate config` to set up your training configuration.

3. Finally, launch the script:

Edit any args below as needed...

```
accelerate launch run.py \
--output_dir outputs \
--batch_size 8 \
--num_workers 8 \
--gradient_accumulation_steps 4 \
--checkpointing_steps epoch \
--mixed_precision fp16 \
--with_tracking \
--num_frames 32 \
--sampling_rate 2 \
--is_slowfast \
--pin_memory
```

