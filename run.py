import itertools
import os
from argparse import Namespace

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from pytorchvideo.data import Kinetics, make_clip_sampler
from pytorchvideo.models import create_res_basic_head
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Div255,
    Normalize,
    RandomShortSideScale,
    RemoveKey,
    ShortSideScale,
    UniformTemporalSubsample,
)
from torch.utils.data import DataLoader, Dataset, DistributedSampler, RandomSampler
from torchmetrics import Accuracy
from torchvision.transforms import CenterCrop, Compose, RandomCrop, RandomHorizontalFlip
from tqdm.auto import tqdm


class LimitDataset(Dataset):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset
        self.dataset_iter = itertools.chain.from_iterable(itertools.repeat(iter(dataset), 2))

    def __getitem__(self, index):
        return next(self.dataset_iter)

    def __len__(self):
        return self.dataset.num_videos


class PackPathway(torch.nn.Module):
    """
    Transform for converting a video clip into a list of 2 clips with
    different temporal granualirity as needed by the SlowFast video
    model.
    For more details, refere to the paper,
    Christoph Feichtenhofer, Haoqi Fan, Jitendra Malik, and Kaiming He.
    "SlowFast networks for video recognition."
    https://arxiv.org/pdf/1812.03982.pdf
    Args:
        alpha (int): Number of frames to sub-sample from the given clip
        to create the second clip.
    """

    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha

    def forward(self, frames: torch.Tensor):
        fast_pathway = frames
        # Perform temporal sampling from the fast pathway.
        slow_pathway = torch.index_select(
            frames,
            1,
            torch.linspace(0, frames.shape[1] - 1, frames.shape[1] // self.alpha).long(),
        )
        frame_list = [slow_pathway, fast_pathway]
        return frame_list


def make_transform(
    min_short_side_scale=256,
    max_short_side_scale=320,
    mean=[0.45, 0.45, 0.45],
    std=[0.225, 0.225, 0.225],
    crop_size=256,
    num_frames=32,
    slowfast_alpha=4,
    is_slowfast=True,
    horizontal_flip_p=0.5,
    training=False,
):

    transform = [
        UniformTemporalSubsample(num_frames),
        Div255(),
        Normalize(mean, std),
    ]

    if training:
        transform += [
            RandomShortSideScale(min_short_side_scale, max_short_side_scale),
            RandomCrop(crop_size),
            RandomHorizontalFlip(p=horizontal_flip_p),
        ]
    else:
        transform += [
            ShortSideScale(min_short_side_scale),
            CenterCrop(crop_size),
        ]

    if is_slowfast:
        transform.append(PackPathway(slowfast_alpha))

    return Compose([ApplyTransformToKey(key="video", transform=Compose(transform)), RemoveKey("audio")])


def make_slowfast_finetuner(num_labels, pretrained=True, freeze_backbone=False):
    """Init Pretrained Model, freeze its backbone, and replace its classification head"""
    model = torch.hub.load("facebookresearch/pytorchvideo", "slowfast_r50", pretrained=pretrained)
    model.blocks[:-1].requires_grad_(not freeze_backbone)
    model.blocks[-1] = create_res_basic_head(in_features=2304, out_features=num_labels, pool=None)
    return model


def make_slowr50_finetuner(num_labels, pretrained=True, freeze_backbone=False):
    """Init Pretrained Model, freeze its backbone, and replace its classification head"""
    model = torch.hub.load("facebookresearch/pytorchvideo", "slow_r50", pretrained=pretrained)
    model.blocks[:-1].requires_grad_(not freeze_backbone)
    model.blocks[-1] = create_res_basic_head(in_features=2048, out_features=num_labels)
    return model


def training_function(args):

    if hasattr(args.checkpointing_steps, "isdigit"):
        if args.checkpointing_steps == "epoch":
            checkpointing_steps = args.checkpointing_steps
        elif args.checkpointing_steps.isdigit():
            checkpointing_steps = int(args.checkpointing_steps)
        else:
            raise ValueError(
                f"Argument `checkpointing_steps` must be either a number or `epoch`. `{args.checkpointing_steps}` passed."
            )
    else:
        checkpointing_steps = None

    accelerator = Accelerator(
        cpu=args.cpu, mixed_precision=args.mixed_precision, log_with="all", logging_dir=args.logging_dir
    )
    set_seed(42)

    clip_duration = (args.sampling_rate * args.num_frames) / args.frames_per_second

    # Define transforms
    train_transform = make_transform(
        num_frames=args.num_frames, training=True, is_slowfast=args.is_slowfast, slowfast_alpha=args.slowfast_alpha
    )
    val_transform = make_transform(
        num_frames=args.num_frames, training=False, is_slowfast=args.is_slowfast, slowfast_alpha=args.slowfast_alpha
    )

    sampler = RandomSampler if accelerator.state.distributed_type == "NO" else DistributedSampler
    train_dataset = LimitDataset(
        Kinetics(
            os.path.join(args.data_dir, "train"),
            clip_sampler=make_clip_sampler("random", clip_duration),
            decode_audio=False,
            transform=train_transform,
            video_sampler=sampler,
        )
    )
    val_dataset = LimitDataset(
        Kinetics(
            os.path.join(args.data_dir, "val"),
            clip_sampler=make_clip_sampler("uniform", clip_duration),
            decode_audio=False,
            transform=val_transform,
            video_sampler=sampler,
        )
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=args.pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=args.pin_memory,
    )

    num_labels = len({path[1] for path in train_dataset.dataset._labeled_videos._paths_and_labels})

    if args.is_slowfast:
        model = make_slowfast_finetuner(num_labels, pretrained=args.pretrained, freeze_backbone=args.freeze_backbone)
    else:
        model = make_slowr50_finetuner(num_labels, pretrained=args.pretrained, freeze_backbone=args.freeze_backbone)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, (len(train_loader) * args.num_epochs) // args.gradient_accumulation_steps, last_epoch=-1
    )
    model, train_loader, val_loader, optimizer, scheduler = accelerator.prepare(
        model, train_loader, val_loader, optimizer, scheduler
    )
    accelerator.register_for_checkpointing(scheduler)  # TODO - Check if this is redundant

    global_step = 0
    starting_epoch = 0
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            accelerator.print(f"Resumed from checkpoint: {args.resume_from_checkpoint}")
            accelerator.load_state(args.resume_from_checkpoint)
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last

        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]
        accelerator.print("\nTRAINING DIFFERENCE", training_difference)

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
        else:
            resume_step = int(training_difference.replace("step_", ""))
            starting_epoch = resume_step // len(train_loader)
            resume_step -= starting_epoch * len(train_loader)

    # We need to initalize the trackers we use. Overall configurations can also be stored
    if args.with_tracking:
        if accelerator.is_main_process:
            run = str(args.logging_dir).replace(".", "").replace("/", "").replace("\\", "")
            accelerator.print(f"Initializing tracker for run {run}")
            accelerator.init_trackers(run, vars(args))

    progress_bar = tqdm(range(args.num_epochs * len(train_loader)), disable=not accelerator.is_main_process)
    for epoch in range(starting_epoch, args.num_epochs):
        progress_bar.set_description_str("Epoch: %s" % epoch)
        val_acc = Accuracy().to(accelerator.device)

        if args.with_tracking:
            total_loss = 0

        # Training
        model.train()
        for step, batch in enumerate(train_loader):

            # Skip steps when resuming (likely incredibly slow in our use case)
            if args.resume_from_checkpoint and epoch == starting_epoch:
                if resume_step is not None and step < resume_step:
                    global_step += 1
                    continue

            inputs = batch["video"]
            labels = batch["label"]
            outputs = model(batch["video"])
            loss = torch.nn.functional.cross_entropy(outputs, labels)

            loss = loss / args.gradient_accumulation_steps
            accelerator.backward(loss)
            if step % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            progress_bar.update(1)
            global_step += 1

            # Log Train Metrics
            if args.with_tracking:
                step_loss = loss.detach().float()
                total_loss += step_loss

                curr_lr = optimizer.param_groups[0]["lr"]

                if (step + 1) % args.log_every == 0:
                    accelerator.log({"train_loss_step": step_loss.item(), "lr": curr_lr}, step=global_step)

            if isinstance(checkpointing_steps, int):
                output_dir = f"step_{global_step}"
                if global_step % checkpointing_steps == 0:
                    if args.output_dir is not None:
                        output_dir = os.path.join(args.output_dir, output_dir)
                    accelerator.save_state(output_dir)
                    accelerator.print(f"Saving checkpoint to ")

            if step == args.limit_train_batches:
                break

        # Evaluation
        progress_bar.set_description_str("Val Epoch: %s" % epoch)
        model.eval()
        for step, batch in enumerate(val_loader):
            inputs = batch["video"]
            labels = batch["label"]
            with torch.no_grad():
                outputs = model(inputs)
                loss = torch.nn.functional.cross_entropy(outputs, labels)

            predictions = outputs.argmax(dim=-1)
            acc = val_acc(accelerator.gather(predictions), accelerator.gather(labels))

            if step == args.limit_val_batches:
                break

        val_acc_epoch = val_acc.compute().item()
        val_acc.reset()

        # Log Eval/Epoch level metrics
        if args.with_tracking:
            accelerator.log(
                {
                    "accuracy": val_acc_epoch,
                    "train_loss_epoch": total_loss.item() / len(train_loader),
                    "epoch": epoch,
                },
                step=epoch,
            )
        if checkpointing_steps == "epoch":
            output_dir = f"epoch_{epoch}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
            accelerator.save_state(output_dir)

    if args.with_tracking:
        accelerator.end_training()

    accelerator.save_state(output_dir)


def main(
    cpu: bool = False,
    mixed_precision: str = "no",
    checkpointing_steps: str = None,
    resume_from_checkpoint: str = None,
    with_tracking: bool = False,
    logging_dir: str = "pytorchvideo_accelerate_runs",
    output_dir: str = ".",
    log_every: int = 10,
    data_dir: str = "/home/jupyter/data",
    num_frames: int = 8,
    sampling_rate: int = 8,
    frames_per_second: int = 30,
    num_epochs: int = 4,
    pretrained: bool = False,
    lr: float = 0.1,
    momentum: float = 0.9,
    weight_decay: float = 1e-4,
    gradient_accumulation_steps: int = 4,
    num_workers: int = 8,
    batch_size: int = 8,
    limit_train_batches: int = -1,
    limit_val_batches: int = -1,
    is_slowfast: bool = False,
    slowfast_alpha: int = 4,
    freeze_backbone: bool = False,
    pin_memory: bool = False,
    seed: int = 42,
):
    """Run training of 3D Resnet or SlowFast Resnet for Action Recognition on the Kinetics dataset.

    Launch this script with accelerate. Any arg in this function can be a flag thanks to Python Fire.

    accelerate config
    accelerate launch run.py <flags>

    Args:
        cpu (bool, optional): _description_. Defaults to False.
        mixed_precision (str, optional): _description_. Defaults to "no".
        checkpointing_steps (str, optional): Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.. Defaults to None.
        resume_from_checkpoint (str, optional): If the training should continue from a checkpoint folder.. Defaults to None.
        with_tracking (bool, optional): Whether to load in all available experiment trackers from the environment and use them for logging.. Defaults to False.
        logging_dir (str, optional): Location on where to store experiment tracking logs. Defaults to "pytorchvideo_accelerate_runs".
        output_dir (str, optional): Optional save directory where all checkpoint folders will be stored. Default is the current working directory.. Defaults to ".".
        log_every (int, optional): Log results to logger every X training steps. Defaults to 10.
        data_dir (str, optional): _description_. Defaults to "/home/jupyter/data".
        num_frames (int, optional): _description_. Defaults to 8.
        sampling_rate (int, optional): _description_. Defaults to 8.
        frames_per_second (int, optional): _description_. Defaults to 30.
        num_epochs (int, optional): _description_. Defaults to 4.
        pretrained (bool, optional): _description_. Defaults to False.
        lr (float, optional): _description_. Defaults to 0.1.
        momentum (float, optional): _description_. Defaults to 0.9.
        weight_decay (float, optional): _description_. Defaults to 1e-4.
        gradient_accumulation_steps (int, optional): _description_. Defaults to 4.
        num_workers (int, optional): _description_. Defaults to 8.
        batch_size (int, optional): _description_. Defaults to 8.
        limit_train_batches (int, optional): _description_. Defaults to -1.
        limit_val_batches (int, optional): _description_. Defaults to -1.
        is_slowfast (bool, optional): _description_. Defaults to False.
        slowfast_alpha (int, optional): _description_. Defaults to 4.
        freeze_backbone (bool, optional): _description_. Defaults to False.
        pin_memory (bool, optional): _description_. Defaults to False.
    """
    args = Namespace(
        cpu=cpu,
        mixed_precision=mixed_precision,
        checkpointing_steps=checkpointing_steps,
        resume_from_checkpoint=resume_from_checkpoint,
        with_tracking=with_tracking,
        output_dir=output_dir,
        log_every=log_every,
        logging_dir=logging_dir,
        data_dir=data_dir,
        num_frames=num_frames,
        sampling_rate=sampling_rate,
        frames_per_second=frames_per_second,
        num_epochs=num_epochs,
        pretrained=pretrained,
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_workers=num_workers,
        batch_size=batch_size,
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
        is_slowfast=is_slowfast,
        slowfast_alpha=slowfast_alpha,
        freeze_backbone=freeze_backbone,
        pin_memory=pin_memory,
        seed=seed,
    )
    training_function(args)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
