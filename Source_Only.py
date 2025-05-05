import argparse
import os
import random
from time import *
from datetime import datetime # Import datetime for logging timestamp

import numpy as np
import torch
from torch.autograd import Variable
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader

import warnings
import my_loss
import network
# from fine_tune import Dataset_build # Likely not needed for Source_Only if get_loader works
from get_loader import get_sloader, get_tloader
from plot_confusion_matrix import plot_confusion_matrix

warnings.filterwarnings("ignore")


def testall(test_dataloader, trained_model, args):
    # Ensure the directory for saving the plot exists
    os.makedirs(os.path.dirname(args.result_plot_path), exist_ok=True) # Use a separate arg for plot path

    model = network.CNN1(args.class_num)
    model.load_state_dict(torch.load(trained_model))
    model.cuda()
    model.train(False)
    with torch.no_grad():
        num_classes = args.class_num
        # 启动网络
        corr_cnt = 0
        total_iter = 0
        conf = np.zeros([num_classes, num_classes])
        confnorm = np.zeros([num_classes, num_classes])
        log_str_list = [] # To store log lines

        log_str_list.append(f"\n--- Test Results for Model: {trained_model} ---")

        for data in test_dataloader:
            # 设置超参数
            input_data, label = data # Renamed input to input_data to avoid shadowing built-in
            input_data = Variable(input_data.cuda())

            # 计算类别分类正确的个数
            # Make sure the model output matches the expected number of return values if using CNN1
            # CNN1 returns: feature1, feature2, feature3, output
            # If only output is needed for classification:
            try:
                 _, _, _, output = model(input_data) # Assuming CNN1 structure
            except ValueError:
                 # Handle cases where the model might only return output (e.g., if modified)
                 output = model(input_data)


            pred = output.data.max(1, keepdim=True)[1]
            label = label.cpu().numpy() # Convert label tensor to numpy array
            for p, l in zip(pred.cpu().numpy(), label): # Iterate over numpy arrays
                p_item = p[0] # Extract scalar from prediction array
                if (p_item == l):
                    corr_cnt += 1
                conf[l, p_item] = conf[l, p_item] + 1
                total_iter += 1

        overall_acc_percent = 100 * corr_cnt / total_iter
        overall_acc_str = f"{args.name} AccuracyOverall {overall_acc_percent:.4f}% ({corr_cnt}/{total_iter})"
        print(overall_acc_str)
        log_str_list.append(overall_acc_str)

        # 绘制总的混淆矩阵
        if total_iter > 0 : # Avoid division by zero if test set is empty
             # Normalize confusion matrix
             for i in range(num_classes):
                  row_sum = np.sum(conf[i, :])
                  if row_sum > 0:
                       confnorm[i, :] = conf[i, :] / row_sum
                  else:
                       confnorm[i, :] = 0 # Handle case where a class has no samples

             # Check if args.classes is defined and has the correct length
             if hasattr(args, 'classes') and len(args.classes) == num_classes:
                 plot_labels = args.classes
             else:
                 plot_labels = [str(i) for i in range(num_classes)] # Default labels if not provided
                 print(f"Warning: args.classes not defined or length mismatch. Using default labels: {plot_labels}")

             plot_title = args.result_plot_path # Use the dedicated plot path argument
             plot_confusion_matrix(confnorm, mod_labels=plot_labels, title=plot_title)
             print(f"Confusion matrix saved to {plot_title}.png")


             # 输出各个调制方式的识别率
             log_str_list.append("\nPer-Class Accuracy:")
             for i in range(num_classes):
                  class_acc_percent = 100 * confnorm[i, i]
                  class_sum = int(np.sum(conf[i, :]))
                  class_correct = int(conf[i, i])
                  class_acc_str = f"{plot_labels[i]} {class_acc_percent:.4f}% ({class_correct}/{class_sum})"
                  print(class_acc_str)
                  log_str_list.append(class_acc_str)
        else:
             no_test_data_str = "No data found in test_dataloader. Skipping confusion matrix and per-class accuracy."
             print(no_test_data_str)
             log_str_list.append(no_test_data_str)

        log_str_list.append("--- End Test Results ---")

        # Append test results to the *main* train log file
        try:
            with open(args.trainlog, 'a', encoding='utf-8') as f:
                for line in log_str_list:
                    f.write(line + "\n")
        except Exception as e:
            print(f"Error writing test results to log file {args.trainlog}: {e}")


def matadd(mat=None):
    res = 0
    if mat is not None:
        for q in range(len(mat)):
            res += mat[q][q]
    return res


def image_classification(loader, model):
    """Evaluates model performance on the target dataset during training."""
    start_test = True
    all_output = None
    all_label = None
    model.eval() # Set model to evaluation mode
    with torch.no_grad():
        # Check if 'target' loader exists and has data
        if "target" not in loader or len(loader['target']) == 0:
            print("Warning: Target loader is empty or not found in image_classification.")
            return 0.0, torch.zeros(model.classifier[-1].out_features), 0.0 # Return default values

        iter_test = iter(loader["target"])
        for i in range(len(loader['target'])):
            try:
                data = next(iter_test)
            except StopIteration:
                print(f"Warning: StopIteration reached early at index {i} in image_classification.")
                break # Exit loop if iterator is exhausted

            inputs, labels = data
            inputs = inputs.cuda()

            # Adapt model call based on its structure
            try:
                 _, _, _, outputs = model(inputs) # Assuming CNN1 structure
            except ValueError:
                 # Handle cases where the model might only return output
                 outputs = model(inputs)

            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                # Ensure tensors are not None before concatenating
                if all_output is not None and all_label is not None:
                    all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                    all_label = torch.cat((all_label, labels.float()), 0)
                else: # Should not happen if start_test logic is correct, but as a safeguard
                     print("Warning: all_output or all_label is None during concatenation.")
                     all_output = outputs.float().cpu()
                     all_label = labels.float()


    # Calculations after the loop
    if all_output is None or all_label is None or all_label.size(0) == 0:
         print("Warning: No data processed in image_classification. Returning zero accuracy.")
         # Need to know the number of classes for hist_tar shape
         try:
             num_classes = model.classifier[-1].out_features
         except:
             num_classes = args.class_num # Fallback
         return 0.0, torch.zeros(num_classes), 0.0

    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0]) * 100

    # Calculate entropy safely
    softmax_output = torch.nn.Softmax(dim=1)(all_output)
    mean_ent = torch.mean(my_loss.Entropy(softmax_output)).cpu().data.item()

    # Calculate histogram
    hist_tar = softmax_output.sum(dim=0)
    hist_tar = hist_tar / hist_tar.sum()

    model.train(True) # Set model back to train mode
    return accuracy, hist_tar, mean_ent


def train(args):
    # 准备数据，设置训练集和测试集的bs（训练数量）
    train_bs = args.batch_size

    dsets = {}
    # --- Load Source Data ---
    try:
        with open(args.trainlog, 'a', encoding='utf-8') as f:
            f.write("\n**************** 开始加载源域训练数据 ****************")
        print("Loading source data...")
        dsets["source"] = get_sloader(args.s_dset_path, args.trainlog)
        if len(dsets["source"]) == 0:
            raise ValueError("Source dataset is empty.")
        print(f"Source data loaded: {len(dsets['source'])} samples.")
        with open(args.trainlog, 'a', encoding='utf-8') as f:
             f.write(f"\n源域数据加载成功: {len(dsets['source'])} 样本.")
    except Exception as e:
        print(f"Error loading source dataset from {args.s_dset_path}: {e}")
        with open(args.trainlog, 'a', encoding='utf-8') as f:
            f.write(f"\n错误：加载源域数据失败: {e}")
        return # Stop execution if source data fails to load

    # --- Load Target Data (for evaluation during training) ---
    try:
        with open(args.trainlog, 'a', encoding='utf-8') as f:
            f.write("\n**************** 开始加载目标域测试数据 ****************")
        print("Loading target data for evaluation...")
        # Use get_tloader, assuming it's designed for test/target data loading
        dsets["target"] = get_tloader(args.t_dset_path, args.trainlog)
        if len(dsets["target"]) == 0:
             print("Warning: Target dataset (for evaluation) is empty.")
             with open(args.trainlog, 'a', encoding='utf-8') as f:
                  f.write("\n警告：目标域测试数据为空.")
        else:
             print(f"Target data loaded: {len(dsets['target'])} samples.")
             with open(args.trainlog, 'a', encoding='utf-8') as f:
                  f.write(f"\n目标域测试数据加载成功: {len(dsets['target'])} 样本.")
    except Exception as e:
        print(f"Warning: Error loading target dataset from {args.t_dset_path}: {e}. Evaluation during training might fail.")
        with open(args.trainlog, 'a', encoding='utf-8') as f:
            f.write(f"\n警告：加载目标域测试数据失败: {e}")
        dsets["target"] = None # Set to None if loading fails

    # --- Create DataLoaders ---
    dset_loaders = {}
    dset_loaders["source"] = DataLoader(dsets["source"], batch_size=train_bs, shuffle=True, num_workers=args.worker,
                                        drop_last=True, pin_memory=True) # Added pin_memory

    # Only create target loader if target dataset loaded successfully
    if dsets["target"] is not None and len(dsets["target"]) > 0:
        dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, shuffle=False, num_workers=args.worker, # shuffle=False for evaluation
                                            drop_last=False, pin_memory=True)
    else:
        dset_loaders["target"] = None # Explicitly set to None

    # --- Initialize Model and Optimizer ---
    print("Initializing model...")
    base_network = network.CNN1(args.class_num)
    base_network = base_network.cuda()
    print("Model initialized.")

    optimizer = torch.optim.Adam(list(base_network.parameters()), lr=args.lr)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=2, T_mult=2) # Adjust T_0, T_mult as needed

    # --- Training Loop ---
    class_weight = None # In Source_Only, class_weight is calculated but not used in loss
    best_acc = 0.0 # Initialize best_acc
    best_ent = float('inf') # Initialize best_ent

    # Calculate total_epochs more carefully if max_iterations is small relative to dataset size
    # If using iterations, total_epochs isn't strictly necessary for the loop itself
    # total_epochs = args.max_iterations // args.test_interval # This might be misleading

    with open(args.trainlog, 'a', encoding='utf-8') as f:
        f.write("\n**************** 开始训练 ****************")
    print("Starting training...")
    begin_time = time()

    source_iter = iter(dset_loaders["source"]) # Initialize iterator

    for i in range(args.max_iterations + 1):
        base_network.train(True) # Ensure model is in training mode

        # --- Get Source Data Batch ---
        try:
            inputs_source, labels_source = next(source_iter)
        except StopIteration:
            # Epoch finished, reset iterator
            print(f"Epoch finished at iteration {i}. Resetting source iterator.")
            source_iter = iter(dset_loaders["source"])
            inputs_source, labels_source = next(source_iter)

        inputs_source, labels_source = inputs_source.cuda(), labels_source.cuda()

        # --- Forward Pass ---
        # Adapt based on model structure (CNN1 returns multiple values)
        try:
             _, _, _, outputs_source = base_network(inputs_source) # Assuming CNN1
        except ValueError:
             outputs_source = base_network(inputs_source) # Fallback

        # --- Calculate Loss ---
        # Source-Only uses standard CrossEntropyLoss
        criterion = torch.nn.CrossEntropyLoss()
        total_loss = criterion(outputs_source, labels_source)

        # --- Backward Pass and Optimization ---
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # --- Learning Rate Scheduling ---
        # Scheduler step depends on the type. CosineAnnealingWarmRestarts steps per iteration/batch.
        scheduler.step()

        # --- Evaluation and Logging ---
        if (i % args.test_interval == 0 and i > 0) or (i == args.max_iterations):
            print(f"\n--- Iteration {i}/{args.max_iterations} ---")
            log_str_iter = [f"\n--- Iteration {i}/{args.max_iterations} ---"]

            current_lr = optimizer.param_groups[0]['lr'] # Get current learning rate
            iter_loss_str = f"Iter Loss: {total_loss.item():.4f}, LR: {current_lr:.6f}"
            print(iter_loss_str)
            log_str_iter.append(iter_loss_str)


            # Evaluate on the target set if available
            if dset_loaders["target"] is not None:
                print("Evaluating on target set...")
                temp_acc, class_weight_eval, mean_ent = image_classification(dset_loaders, base_network) # class_weight_eval is calculated but not used further in Source_Only
                class_weight_list = [round(cw, 4) for cw in class_weight_eval.cpu().numpy().tolist()] if class_weight_eval is not None else []

                eval_str = f"Target Eval Acc: {temp_acc:.4f}%, Mean Entropy: {mean_ent:.4f}"
                print(eval_str)
                # print(f"Target Class Weights (eval): {class_weight_list}") # Optional: print weights
                log_str_iter.append(eval_str)
                # log_str_iter.append(f"Target Class Weights (eval): {class_weight_list}")

                # Save best model based on target accuracy
                if temp_acc > best_acc:
                     best_acc = temp_acc
                     best_ent = mean_ent # Store corresponding entropy
                     best_model_state = base_network.state_dict()
                     save_path = os.path.join(args.weightpath, args.name + ".pt") # Use os.path.join
                     # Ensure weight directory exists
                     os.makedirs(args.weightpath, exist_ok=True)
                     try:
                          torch.save(best_model_state, save_path)
                          save_msg = f"Best accuracy improved to {best_acc:.4f}%. Saving model to {save_path}"
                          print(save_msg)
                          log_str_iter.append(save_msg)
                     except Exception as e:
                          save_err_msg = f"Error saving model: {e}"
                          print(save_err_msg)
                          log_str_iter.append(save_err_msg)
            else:
                print("No target data loaded, skipping evaluation.")
                log_str_iter.append("No target data loaded, skipping evaluation.")
                # If no target data, maybe save based on iteration count or loss?
                # For Source_Only, saving the last model might be sufficient if no eval set.
                if i == args.max_iterations:
                     last_model_state = base_network.state_dict()
                     save_path = os.path.join(args.weightpath, args.name + "_last.pt")
                     os.makedirs(args.weightpath, exist_ok=True)
                     try:
                          torch.save(last_model_state, save_path)
                          save_msg = f"Reached max iterations. Saving last model to {save_path}"
                          print(save_msg)
                          log_str_iter.append(save_msg)
                     except Exception as e:
                          save_err_msg = f"Error saving last model: {e}"
                          print(save_err_msg)
                          log_str_iter.append(save_err_msg)


            # Write iteration log
            try:
                with open(args.trainlog, 'a', encoding='utf-8') as f:
                    for line in log_str_iter:
                        f.write(line + "\n")
            except Exception as e:
                print(f"Error writing iteration log to {args.trainlog}: {e}")


    # --- End of Training ---
    end_time = time()
    run_time = end_time - begin_time
    final_log_str = [
        f"\n**************** 训练结束 ****************",
        f"最佳目标域准确率 (Best Target Acc): {best_acc:.4f}%",
        f"对应的平均熵 (Mean Entropy at Best Acc): {best_ent:.4f}",
        f"训练总耗时 (Total Training Time): {run_time:.2f}s ({run_time/60:.2f}m)"
    ]
    print("\n" + "\n".join(final_log_str))
    try:
        with open(args.trainlog, 'a', encoding='utf-8') as f:
             for line in final_log_str:
                  f.write(line + "\n")
    except Exception as e:
        print(f"Error writing final log to {args.trainlog}: {e}")

    # --- Final Test ---
    # Optionally run testall on the best model saved
    best_model_path = os.path.join(args.weightpath, args.name + ".pt")
    if os.path.exists(best_model_path) and dset_loaders["target"] is not None:
         print("\nRunning final test on the best saved model...")
         # Define result plot path for testall
         args.result_plot_path = os.path.join(args.trainlog_path, args.name + "_ConfusionMatrix") # Save plot near log
         testall(dset_loaders["target"], best_model_path, args)
    elif dset_loaders["target"] is None:
         print("\nSkipping final test because target data was not loaded.")
    else:
         print(f"\nSkipping final test because best model file not found: {best_model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Source Only Training for AMC')
    # --- Paths ---
    parser.add_argument('--s_dset_path', type=str, required=True, help="Path to the source domain HDF5 dataset")
    parser.add_argument('--t_dset_path', type=str, required=True, help="Path to the target domain HDF5 dataset (used for evaluation)")
    parser.add_argument('--weightpath', type=str, default="./weights/", help="Directory to save model weights")
    parser.add_argument('--trainlog_path', type=str, default="./logs/", help="Directory to save training logs")
    # --- Run Identification ---
    parser.add_argument('--name', type=str, default="source_only_run", help="Name for this training run (used for weight and log files)")
    # --- Model & Data ---
    parser.add_argument('--class_num', type=int, required=True, help="Number of modulation classes in the source dataset")
    parser.add_argument('--classes', type=str, nargs='+', default=["BPSK", "8PSK", "PAM4", "PAM8", "16QAM", "64QAM"], help='List of class names (for plotting)')
    # --- Training Hyperparameters ---
    parser.add_argument('--seed', type=int, default=2021, help="Random seed")
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="GPU device ID to use (e.g., '0', '0,1')")
    parser.add_argument('--batch_size', type=int, default=500, help="Batch size")
    parser.add_argument('--max_iterations', type=int, default=2000, help="Total number of training iterations (batches)")
    parser.add_argument('--worker', type=int, default=1, help="Number of dataloader workers")
    parser.add_argument('--test_interval', type=int, default=100, help="Evaluate on target set every N iterations") # Increased default
    parser.add_argument('--lr', type=float, default=2e-3, help="Learning rate")
    # --- Source Only Specific (placeholders, not used in this script's logic) ---
    # parser.add_argument('--mu', type=int, default=0, help="Placeholder (Not used in Source_Only)")
    # parser.add_argument('--ent_weight', type=float, default=0, help="Placeholder (Not used in Source_Only)")
    # parser.add_argument('--cot_weight', type=float, default=0, help="Placeholder (Not used in Source_Only)")
    # parser.add_argument('--weight_aug', type=bool, default=False, help="Placeholder (Not used in Source_Only)") # Changed default to False
    # parser.add_argument('--weight_cls', type=bool, default=False, help="Placeholder (Not used in Source_Only)") # Changed default to False
    # parser.add_argument('--alpha', type=float, default=1, help="Placeholder (Not used in Source_Only)")

    args = parser.parse_args()

    # --- Environment Setup ---
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    print(f"Using GPU: {args.gpu_id}")

    # --- Seed Setup ---
    print(f"Setting random seed: {args.seed}")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    # Ensure reproducibility (optional, can slow down training)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    # --- Verify/Create Directories ---
    os.makedirs(args.weightpath, exist_ok=True)
    os.makedirs(args.trainlog_path, exist_ok=True)
    print(f"Weights will be saved in: {args.weightpath}")
    print(f"Logs will be saved in: {args.trainlog_path}")

    # --- Construct Log File Path ---
    args.trainlog = os.path.join(args.trainlog_path, args.name + ".txt")
    print(f"Log file path: {args.trainlog}")

    # --- Validate Class List Length ---
    if len(args.classes) != args.class_num:
         print(f"Warning: Length of --classes ({len(args.classes)}) does not match --class_num ({args.class_num}). Using default labels for plotting.")
         # Optionally, adjust args.classes or raise an error
         # args.classes = [str(i) for i in range(args.class_num)]

    # --- Initial Log Header ---
    try:
        with open(args.trainlog, 'w', encoding='utf-8') as f:
            f.write(f"Source Only Training Run: {args.name}\n")
            f.write("===================================\n")
            f.write(f"Timestamp: {datetime.now()}\n")
            f.write(f"Source Dataset: {args.s_dset_path}\n")
            f.write(f"Target/Test Dataset: {args.t_dset_path}\n")
            f.write(f"Weight Path: {args.weightpath}\n")
            f.write(f"Log Path: {args.trainlog}\n")
            f.write(f"Class Num: {args.class_num}\n")
            f.write(f"Classes: {args.classes}\n")
            f.write(f"Batch Size: {args.batch_size}\n")
            f.write(f"Max Iterations: {args.max_iterations}\n")
            f.write(f"Test Interval: {args.test_interval}\n")
            f.write(f"Learning Rate: {args.lr}\n")
            f.write(f"Seed: {args.seed}\n")
            f.write(f"GPU ID: {args.gpu_id}\n")
            f.write("===================================\n")
    except Exception as e:
        print(f"Error writing initial log header to {args.trainlog}: {e}")
        # Decide if you want to exit if the log can't be written
        # sys.exit(1)

    # --- Start Training ---
    print(f"\nStarting training run: {args.name}")
    train(args)
    print(f"\nFinished training run: {args.name}")