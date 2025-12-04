# All code is original unless otherwise noted.
import sys
import argparse
import torch.optim as optim
from model import *
from data import *
from utils import *
from betty.engine import Engine
from betty.problems import ImplicitProblem
from betty.configs import Config, EngineConfig
import wandb
from transformers import AdamW
import numpy as np
import torch
import torch.utils.checkpoint as cp
from functools import partial
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm


cp.checkpoint = partial(cp.checkpoint, use_reentrant=False)


parser = argparse.ArgumentParser(description="DreamPRM-1.5")
# data file path and model path
parser.add_argument('--train_json_file', type=str, default="./data/train.json")
# parser.add_argument('--train_json_file', type=str, default="./data/train_small_cleaned.json")
# parser.add_argument('--train_json_file', type=str, default="./data/training_data_prm_combined.json")
# parser.add_argument('--meta_json_file', type=str, default="./data/meta.json")
parser.add_argument('--meta_json_file', type=str, default="./data/meta_MMMU_Pro.json")
parser.add_argument('--weights_path', type=str, default="./weights")
parser.add_argument("--reward_model", type=str, default="OpenGVLab/InternVL3-1B")
# bi-level optimization configuration
parser.add_argument("--iteration_num", type=int, default=100000)
parser.add_argument("--save_every_iterations", type=int, default=10)
parser.add_argument("--unroll_steps", type=int, default=1)
parser.add_argument("--gradiant_accumulation", type=int, default=1)
parser.add_argument("--gradiant_clipping", type=float, default=1.0)
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--precision", type=str, default="bf16")
parser.add_argument("--strategy", type=str, default="default")
parser.add_argument("--rollback", action="store_true")
parser.add_argument("--baseline", action="store_true")
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--local_rank", type=int, default=0)
# lower-level optimization hyperparameters
parser.add_argument("--lr", type=float, default=5e-5)
parser.add_argument("--momentum", type=float, default=0.9)
parser.add_argument("--weight_decay", type=float, default=0.01)
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--scheduler_step_size", type=int, default=1000)
parser.add_argument("--scheduler_gamma", type=float, default=0.95)
# higher-level optimization hyperparameters
parser.add_argument("--meta_lr", type=float, default=5e-3)
parser.add_argument("--meta_momentum", type=float, default=0.9)
parser.add_argument("--meta_weight_decay", type=float, default=1e-3)
parser.add_argument("--meta_batch_size", type=int, default=1)
parser.add_argument("--meta_scheduler_step_size", type=int, default=1000)
parser.add_argument("--meta_scheduler_gamma", type=float, default=0.95)
# other important parameters
parser.add_argument("--retrain", type=float, default=False)
parser.add_argument("--activation_function", type=str, default="LeakyReLU", help="LeakyReLU|ReLU|No|Clip")
parser.add_argument("--aggregation_function", type=str, default="mean", help="mean|max|min|log_mean")
parser.add_argument("--loss_target", type=str, default="both", help="+|both")
parser.add_argument("--initialization", type=float, default=1.0)
parser.add_argument("--max_patch_num", type=int, default=6)
parser.add_argument("--scheduler_type", type=str, default="cosine_schedule_with_warmup", help="cosine_schedule_with_warmup|step_lr")
# parser.add_argument("--dampening", type=float, default=0.0)
# parser.add_argument("--nesterov", type=bool, default=False)
# parser.add_argument("--num_meta", type=int, default=1000)
# parser.add_argument("--imbalanced_factor", type=int, default=None)
# parser.add_argument("--corruption_type", type=str, default=None)
# parser.add_argument("--corruption_ratio", type=float, default=0.0)
# parser.add_argument("--max_epoch", type=int, default=120)
# parser.add_argument("--meta_interval", type=int, default=1)
# parser.add_argument("--paint_interval", type=int, default=20)

args = parser.parse_args()
print(args)
set_seed(args.seed)
domain_list = create_dataset_mapping(args.train_json_file)
print(domain_list)

sampler = None
resume_idxes = None
resume_labels = None

(
    train_dataloader,
    meta_dataloader,
) = build_dataloader(
    train_json_file = args.train_json_file,
    meta_json_file = args.meta_json_file,
    train_batch_size= args.batch_size,
    meta_batch_size= args.batch_size,
    max_patch_num = args.max_patch_num,
)
wandb.init(project="DreamPRM-1.5")

device = torch.device(args.device)
criterion = nn.CrossEntropyLoss()
criterion_meta = nn.MSELoss()
lower_weighted_loss = []
lower_loss = []
upper_loss = []
best_loss = 1000
best_acc = 0
MODEL_PATH = args.reward_model
tokenizer = AutoTokenizer.from_pretrained("OpenGVLab/InternVL3-1B", trust_remote_code=True, use_fast=False)

class Lower(ImplicitProblem):
    def forward(self, pixel_values, input_ids, attention_mask, image_flags):
        # torch.cuda.empty_cache()
        logits = self.module(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            image_flags=image_flags,
        ).logits
        return logits

    def training_step(self, batch):
        prompt,  pixel_values, id = batch
        prompt = prompt[0]
        pixel_values =  pixel_values[0]
        id = id  # TODO: support batch inference
        input_ids, attention_mask, image_flags, pixel_values = input_processing(self.module, tokenizer, prompt, pixel_values)
        output = self.forward(pixel_values, input_ids, attention_mask, image_flags) # (batch, seq_len, vocab_size)
        target = generate_target(input_ids, tokenizer, self.module.template) # (batch, seq_len)
        loss = compute_supervised_loss(output, target)
        # print_target_token_logits(output, target)
        print(f"Current step loss: {loss.item()}")
        if args.baseline or args.retrain:
            print(loss.item())
            if len(lower_loss) == 1000:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                mean_inner_loss = np.mean(lower_loss)
                wandb.log({"inner_loss": mean_inner_loss,})
                lower_loss.clear()
            return loss
        # weighted_loss = self.upper(id, loss)
        lower_loss.append(loss.item())
        # lower_weighted_loss.append(weighted_loss.item())
        if len(lower_loss) == len(train_dataloader):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            mean_inner_loss = np.mean(lower_loss)
            # mean_inner_weighted_loss = np.mean(lower_weighted_loss)
            wandb.log({"inner_loss": mean_inner_loss})
            lower_loss.clear()
            sys.exit()
            # lower_weighted_loss.clear()
        # torch.cuda.empty_cache()

        return loss

    def configure_train_data_loader(self):
        return train_dataloader

    def configure_module(self):
        model = AutoModel.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            torch_dtype=torch.bfloat16,
            use_flash_attn=False,
        ).cuda()
        return model

    def configure_optimizer(self):
        optimizer = AdamW(
            self.module.parameters(),
            lr=args.lr,
            weight_decay = args.weight_decay
        )
        return optimizer

    def configure_scheduler(self):
        if args.baseline or args.retrain:
            scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=0.05 * args.iteration_num,
                num_training_steps=args.iteration_num
            )
        else:
            if args.scheduler_type == "cosine_schedule_with_warmup":
                scheduler = get_cosine_schedule_with_warmup(
                    self.optimizer,
                    num_warmup_steps=0.05 * args.iteration_num,
                    num_training_steps=args.iteration_num
                )
            elif args.scheduler_type == "step_lr":
                scheduler = optim.lr_scheduler.StepLR(
                    self.optimizer, step_size=args.scheduler_step_size, gamma=args.scheduler_gamma
                )
        return scheduler


class ReweightingEngine(Engine):
    # @torch.no_grad()
    # def validation(self):
    #     # save checkpoints
    #     torch.cuda.empty_cache()
    #     torch.cuda.reset_peak_memory_stats()
    #     self.lower.module.save_pretrained(args.weights_path)
    #     return torch.tensor(0.0, device=args.device)

    @torch.no_grad()
    def validation(self):
        print(f"Starting validation...")
        torch.cuda.empty_cache()
        test_dataloader = build_test_dataloader(test_json_file="./data/test_MMMU_8cots.json", return_subset=True)

        correct = 0
        total = 0
        global best_acc

        # Get the model from the lower problem
        model = self.lower.module

        # Go through the testing data set for validation
        for inputs in tqdm(test_dataloader):
            # input_test_data_format:
            # {"question": question, "image_path": image_path, "candidates":[...], "true_false":[True, False, ...]}
            true_false, best_index = select_best_answer(
                model, tokenizer, inputs, args.aggregation_function
            )
            correct += int(true_false)
            total += 1

        acc = correct / total * 100

        if best_acc < acc:
            print(f"NEW BEST ACC: {acc}")
            best_acc = acc
            self.lower.module.save_pretrained(args.weights_path)

        # Log to wandb
        wandb.log({"val_acc": acc, "best_acc": best_acc})

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        return torch.tensor(0.0, device=args.device)

lower_config = Config(type="darts", precision=args.precision, unroll_steps=args.unroll_steps, gradient_accumulation=args.gradiant_accumulation, gradient_clipping=args.gradiant_clipping)
engine_config = EngineConfig(
    train_iters=args.iteration_num,
    valid_step=args.save_every_iterations,
    strategy=args.strategy,
    roll_back=args.rollback,
    logger_type="wandb",
)
lower = Lower(name="lower", config=lower_config)

# if args.baseline or args.retrain:
problems = [lower]
u2l, l2u = {}, {}
dependencies = {"l2u": l2u, "u2l": u2l}

engine = ReweightingEngine(
    config=engine_config, problems=problems, dependencies=dependencies
)
engine.run()
