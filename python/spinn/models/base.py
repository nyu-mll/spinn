import os
import json
import math
import random
import time

import gflags
import numpy as np

from spinn.data.arithmetic import load_sign_data
from spinn.data.arithmetic import load_simple_data
from spinn.data.dual_arithmetic import load_eq_data
from spinn.data.dual_arithmetic import load_relational_data
from spinn.data.boolean import load_boolean_data
from spinn.data.listops import load_listops_data
from spinn.data.sst import load_sst_data, load_sst_binary_data
from spinn.data.snli import load_snli_data
from spinn.util.data import SimpleProgressBar
from spinn.util.blocks import ModelTrainer, the_gpu, to_gpu, l2_cost, flatten
from spinn.util.misc import Accumulator, EvalReporter, time_per_token
from spinn.util.misc import recursively_set_device
from spinn.util.metrics import MetricsWriter
from spinn.util.logging import train_format, train_extra_format, train_stats, train_accumulate
from spinn.util.logging import eval_format, eval_extra_format
from spinn.util.loss import auxiliary_loss
import spinn.util.evalb as evalb

import spinn.gen_spinn
import spinn.rae_spinn
import spinn.rl_spinn
import spinn.fat_stack
import spinn.plain_rnn
import spinn.cbow

# PyTorch
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim


FLAGS = gflags.FLAGS


def sequential_only():
    return FLAGS.model_type == "RNN" or FLAGS.model_type == "CBOW"


def get_batch(batch):
    X_batch, transitions_batch, y_batch, num_transitions_batch, structure_transitions, example_ids = batch

    # Truncate each batch to max length within the batch.
    X_batch_is_left_padded = sequential_only()
    transitions_batch_is_left_padded = True
    max_length = np.max(num_transitions_batch)
    seq_length = X_batch.shape[1]

    # Truncate batch.
    X_batch = truncate(X_batch, seq_length, max_length, X_batch_is_left_padded)
    transitions_batch = truncate(transitions_batch, seq_length, max_length, transitions_batch_is_left_padded)
    structure_transitions = truncate(structure_transitions, seq_length, max_length, transitions_batch_is_left_padded)

    return X_batch, transitions_batch, y_batch, num_transitions_batch, structure_transitions, example_ids


def truncate(data, seq_length, max_length, left_padded):
    if left_padded:
        data = data[:, seq_length - max_length:]
    else:
        data = data[:, :max_length]
    return data


def evaluate(model, eval_set, logger, step, vocabulary=None):
    filename, dataset = eval_set

    reporter = EvalReporter()

    eval_str = eval_format(model)
    eval_extra_str = eval_extra_format(model)

    # Evaluate
    class_correct = 0
    class_total = 0
    total_batches = len(dataset)
    progress_bar = SimpleProgressBar(msg="Run Eval", bar_length=60, enabled=FLAGS.show_progress_bar)
    progress_bar.step(0, total=total_batches)
    total_tokens = 0
    invalid = 0
    start = time.time()

    model.eval()

    transition_preds = []
    transition_targets = []
    transition_examples = []

    for i, batch in enumerate(dataset):
        eval_X_batch, eval_transitions_batch, eval_y_batch, eval_num_transitions_batch, structure_transitions, eval_ids = get_batch(batch)

        # Run model.
        output = model(eval_X_batch, eval_transitions_batch, eval_y_batch,
            use_internal_parser=FLAGS.use_internal_parser,
            validate_transitions=FLAGS.validate_transitions)

        # Normalize output.
        logits = F.log_softmax(output)

        # Calculate class accuracy.
        target = torch.from_numpy(eval_y_batch).long()
        pred = logits.data.max(1)[1].cpu() # get the index of the max log-probability
        class_correct += pred.eq(target).sum()
        class_total += target.size(0)

        # Optionally calculate transition loss/acc.
        transition_loss = model.transition_loss if hasattr(model, 'transition_loss') else None

        # Update Aggregate Accuracies
        total_tokens += sum([(nt+1)/2 for nt in eval_num_transitions_batch.reshape(-1)])

        # Track number of examples with completely valid transitions.
        if hasattr(model, 'spinn') and hasattr(model.spinn, 'invalid'):
            num_sentences = 2 if model.use_sentence_pair else 1
            invalid += model.spinn.invalid

        # Accumulate stats for transition accuracy.
        if transition_loss is not None:
            transition_preds.append([m["t_preds"] for m in model.spinn.memories if m.get('t_preds', None) is not None])
            transition_targets.append([m["t_given"] for m in model.spinn.memories if m.get('t_given', None) is not None])

        if FLAGS.evalb or FLAGS.num_samples > 0:
            transitions_per_example = model.spinn.get_transitions_per_example()
            if model.use_sentence_pair:
                eval_transitions_batch = np.concatenate([
                    eval_transitions_batch[:,:,0], eval_transitions_batch[:,:,1]], axis=0)

        if FLAGS.num_samples > 0 and len(transition_examples) < FLAGS.num_samples and i % (step % 11 + 1) == 0:
            r = random.randint(0, len(transitions_per_example) - 1)
            transition_examples.append((transitions_per_example[r], eval_transitions_batch[r]))

        if FLAGS.write_eval_report:
            reporter_args = [pred, target, eval_ids, output.data.cpu().numpy()]
            if hasattr(model, 'transition_loss'):
                transitions_per_example = model.spinn.get_transitions_per_example(
                    style="preds" if FLAGS.eval_report_use_preds else "given")
                if model.use_sentence_pair:
                    batch_size = pred.size(0)
                    sent1_transitions = transitions_per_example[:batch_size]
                    sent2_transitions = transitions_per_example[batch_size:]
                    reporter_args.append(sent1_transitions)
                    reporter_args.append(sent2_transitions)
                else:
                    reporter_args.append(transitions_per_example)
            reporter.save_batch(*reporter_args)

        # Print Progress
        progress_bar.step(i+1, total=total_batches)
    progress_bar.finish()

    end = time.time()
    total_time = end - start

    # Get time per token.
    time_metric = time_per_token([total_tokens], [total_time])

    # Get class accuracy.
    eval_class_acc = class_correct / float(class_total)

    # Get transition accuracy if applicable.
    if len(transition_preds) > 0:
        all_preds = np.array(flatten(transition_preds))
        all_truth = np.array(flatten(transition_targets))
        eval_trans_acc = (all_preds == all_truth).sum() / float(all_truth.shape[0])
    else:
        eval_trans_acc = 0.0

    has_invalid = hasattr(model, 'spinn') and hasattr(model.spinn, 'invalid')

    stats_args = dict(
        step=step,
        class_acc=eval_class_acc,
        transition_acc=eval_trans_acc,
        filename=filename,
        time=time_metric,
        inv=invalid/float(total_batches) if has_invalid else 0.0,
        )

    logger.Log(eval_str.format(**stats_args))
    logger.Log(eval_extra_str.format(**stats_args))

    if len(transition_examples) > 0:
        transition_str = "Samples:"
        for t_idx in range(len(transition_examples)):
            gold = transition_examples[t_idx][1]
            pred = transition_examples[t_idx][0]
            _, crossing = evalb.crossing(gold, pred)
            transition_str += "\n{}. crossing={}".format(t_idx, crossing)
            transition_str += "\n     g{}".format("".join(map(str, gold)))
            transition_str += "\n     p{}".format("".join(map(str, pred)))
        logger.Log(transition_str)

    if FLAGS.write_eval_report:
        eval_report_path = os.path.join(FLAGS.log_path, FLAGS.experiment_name + ".report")
        reporter.write_report(eval_report_path)

    return eval_class_acc, eval_trans_acc


def get_data_manager(data_type):
    # Select data format.
    if data_type == "bl":
        data_manager = load_boolean_data
    elif data_type == "sst":
        data_manager = load_sst_data
    elif data_type == "sst-binary":
        data_manager = load_sst_binary_data
    elif data_type == "snli":
        data_manager = load_snli_data
    elif data_type == "arithmetic":
        data_manager = load_simple_data
    elif data_type == "listops":
        data_manager = load_listops_data
    elif data_type == "sign":
        data_manager = load_sign_data
    elif data_type == "eq":
        data_manager = load_eq_data
    elif data_type == "relational":
        data_manager = load_relational_data
    else:
        raise NotImplementedError

    return data_manager


def get_checkpoint_path(ckpt_path, experiment_name, suffix=".ckpt", best=False):
    # Set checkpoint path.
    if ckpt_path.endswith(".ckpt") or ckpt_path.endswith(".ckpt_best"):
        checkpoint_path = ckpt_path
    else:
        checkpoint_path = os.path.join(ckpt_path, experiment_name + suffix)
    if best:
        checkpoint_path += "_best"
    return checkpoint_path


def get_flags():
    # Debug settings.
    gflags.DEFINE_bool("debug", False, "Set to True to disable debug_mode and type_checking.")
    gflags.DEFINE_bool("show_progress_bar", True, "Turn this off when running experiments on HPC.")
    gflags.DEFINE_string("branch_name", "", "")
    gflags.DEFINE_integer("deque_length", None, "Max trailing examples to use for statistics.")
    gflags.DEFINE_string("sha", "", "")
    gflags.DEFINE_string("experiment_name", "", "")

    # Data types.
    gflags.DEFINE_enum("data_type", "bl", ["bl", "sst", "sst-binary", "snli", "multinli", "arithmetic", "listops", "sign", "eq", "relational"],
        "Which data handler and classifier to use.")

    # Where to store checkpoints
    gflags.DEFINE_string("log_path", "./logs", "A directory in which to write logs.")
    gflags.DEFINE_string("ckpt_path", None, "Where to save/load checkpoints. Can be either "
        "a filename or a directory. In the latter case, the experiment name serves as the "
        "base for the filename.")
    gflags.DEFINE_string("metrics_path", None, "A directory in which to write metrics.")
    gflags.DEFINE_integer("ckpt_step", 1000, "Steps to run before considering saving checkpoint.")
    gflags.DEFINE_boolean("load_best", False, "If True, attempt to load 'best' checkpoint.")

    # Data settings.
    gflags.DEFINE_string("training_data_path", None, "")
    gflags.DEFINE_string("eval_data_path", None, "Can contain multiple file paths, separated "
        "using ':' tokens. The first file should be the dev set, and is used for determining "
        "when to save the early stopping 'best' checkpoints.")
    gflags.DEFINE_integer("seq_length", 200, "")
    gflags.DEFINE_integer("eval_seq_length", None, "")
    gflags.DEFINE_boolean("smart_batching", True, "Organize batches using sequence length.")
    gflags.DEFINE_boolean("use_peano", True, "A mind-blowing sorting key.")
    gflags.DEFINE_integer("eval_data_limit", -1, "Truncate evaluation set. -1 indicates no truncation.")
    gflags.DEFINE_boolean("bucket_eval", True, "Bucket evaluation data for speed improvement.")
    gflags.DEFINE_boolean("shuffle_eval", False, "Shuffle evaluation data.")
    gflags.DEFINE_integer("shuffle_eval_seed", 123, "Seed shuffling of eval data.")
    gflags.DEFINE_string("embedding_data_path", None,
        "If set, load GloVe-formatted embeddings from here.")

    # Model architecture settings.
    gflags.DEFINE_enum("model_type", "RNN", ["CBOW", "RNN", "SPINN", "RLSPINN", "RAESPINN", "GENSPINN"], "")
    gflags.DEFINE_integer("gpu", -1, "")
    gflags.DEFINE_integer("model_dim", 8, "")
    gflags.DEFINE_integer("word_embedding_dim", 8, "")
    gflags.DEFINE_boolean("lowercase", False, "When True, ignore case.")
    gflags.DEFINE_boolean("use_internal_parser", False, "Use predicted parse.")
    gflags.DEFINE_boolean("validate_transitions", True,
        "Constrain predicted transitions to ones that give a valid parse tree.")
    gflags.DEFINE_float("embedding_keep_rate", 0.9,
        "Used for dropout on transformed embeddings and in the encoder RNN.")
    gflags.DEFINE_boolean("use_l2_cost", True, "")
    gflags.DEFINE_boolean("use_difference_feature", True, "")
    gflags.DEFINE_boolean("use_product_feature", True, "")

    # Tracker settings.
    gflags.DEFINE_integer("tracking_lstm_hidden_dim", None, "Set to none to avoid using tracker.")
    gflags.DEFINE_float("transition_weight", None, "Set to none to avoid predicting transitions.")
    gflags.DEFINE_boolean("lateral_tracking", True,
        "Use previous tracker state as input for new state.")
    gflags.DEFINE_boolean("use_tracking_in_composition", True,
        "Use tracking lstm output as input for the reduce function.")
    gflags.DEFINE_boolean("predict_use_cell", True,
        "Use cell output as feature for transition net.")
    gflags.DEFINE_boolean("use_lengths", False, "The transition net will be biased.")

    # Attention settings.
    gflags.DEFINE_boolean("use_attention", False, "Attend over hidden states.")
    gflags.DEFINE_integer("attention_dim", 128, "Size for output of attention networks.")

    # Encode settings.
    gflags.DEFINE_boolean("use_encode", False, "Encode embeddings with sequential network.")
    gflags.DEFINE_enum("encode_style", None, ["LSTM", "CNN", "QRNN"], "Encode embeddings with sequential context.")
    gflags.DEFINE_boolean("encode_reverse", False, "Encode in reverse order.")
    gflags.DEFINE_boolean("encode_bidirectional", False, "Encode in both directions.")
    gflags.DEFINE_integer("encode_num_layers", 1, "RNN layers in encoding net.")

    # RL settings.
    gflags.DEFINE_float("rl_mu", 0.1, "Use in exponential moving average baseline.")
    gflags.DEFINE_enum("rl_baseline", "ema", ["ema", "greedy"],
        "Different configurations to approximate reward function.")
    gflags.DEFINE_enum("rl_reward", "standard", ["standard", "xent"],
        "Different reward functions to use.")
    gflags.DEFINE_float("rl_weight", 1.0, "Hyperparam for REINFORCE loss.")
    gflags.DEFINE_boolean("rl_whiten", False, "Reduce variance in advantage.")
    gflags.DEFINE_boolean("rl_entropy", False, "Entropy regularization on transition policy.")
    gflags.DEFINE_float("rl_entropy_beta", 0.001, "Entropy regularization on transition policy.")
    gflags.DEFINE_float("rl_epsilon", 1.0, "Percent of sampled actions during train time.")
    gflags.DEFINE_float("rl_epsilon_decay", 50000, "Percent of sampled actions during train time.")

    # RAE settings.
    gflags.DEFINE_boolean("predict_leaf", True, "Predict whether a node is a leaf or not.")

    # GEN settings.
    gflags.DEFINE_boolean("gen_h", True, "Use generator output as feature.")

    # MLP settings.
    gflags.DEFINE_integer("mlp_dim", 1024, "Dimension of intermediate MLP layers.")
    gflags.DEFINE_integer("num_mlp_layers", 2, "Number of MLP layers.")
    gflags.DEFINE_boolean("mlp_bn", True, "When True, batch normalization is used between MLP layers.")
    gflags.DEFINE_float("semantic_classifier_keep_rate", 0.9,
        "Used for dropout in the semantic task classifier.")

    # Optimization settings.
    gflags.DEFINE_enum("optimizer_type", "Adam", ["Adam", "RMSprop"], "")
    gflags.DEFINE_integer("training_steps", 500000, "Stop training after this point.")
    gflags.DEFINE_integer("batch_size", 32, "SGD minibatch size.")
    gflags.DEFINE_float("learning_rate", 0.001, "Used in optimizer.")
    gflags.DEFINE_float("learning_rate_decay_per_10k_steps", 0.75, "Used in optimizer.")
    gflags.DEFINE_boolean("actively_decay_learning_rate", True, "Used in optimizer.")
    gflags.DEFINE_float("clipping_max_value", 5.0, "")
    gflags.DEFINE_float("l2_lambda", 1e-5, "")
    gflags.DEFINE_float("init_range", 0.005, "Mainly used for softmax parameters. Range for uniform random init.")

    # Display settings.
    gflags.DEFINE_integer("statistics_interval_steps", 100, "Print training set results at this interval.")
    gflags.DEFINE_integer("metrics_interval_steps", 10, "Evaluate at this interval.")
    gflags.DEFINE_integer("eval_interval_steps", 100, "Evaluate at this interval.")
    gflags.DEFINE_integer("ckpt_interval_steps", 5000, "Update the checkpoint on disk at this interval.")
    gflags.DEFINE_boolean("ckpt_on_best_dev_error", True, "If error on the first eval set (the dev set) is "
        "at most 0.99 of error at the previous checkpoint, save a special 'best' checkpoint.")
    gflags.DEFINE_boolean("evalb", False, "Print transition statistics.")
    gflags.DEFINE_integer("num_samples", 0, "Print sampled transitions.")

    # Evaluation settings
    gflags.DEFINE_boolean("expanded_eval_only_mode", False,
        "If set, a checkpoint is loaded and a forward pass is done to get the predicted "
        "transitions. The inferred parses are written to the supplied file(s) along with example-"
        "by-example accuracy information. Requirements: Must specify checkpoint path.")
    gflags.DEFINE_boolean("write_eval_report", False, "")
    gflags.DEFINE_boolean("eval_report_use_preds", True, "If False, use the given transitions in the report, "
        "otherwise use predicted transitions. Note that when predicting transitions but not using them, the "
        "reported predictions will look very odd / not valid.")


def flag_defaults(FLAGS):
    if not FLAGS.experiment_name:
        timestamp = str(int(time.time()))
        FLAGS.experiment_name = "{}-{}-{}".format(
            FLAGS.data_type,
            FLAGS.model_type,
            timestamp,
            )

    if not FLAGS.branch_name:
        FLAGS.branch_name = os.popen('git rev-parse --abbrev-ref HEAD').read().strip()

    if not FLAGS.sha:
        FLAGS.sha = os.popen('git rev-parse HEAD').read().strip()

    if not FLAGS.ckpt_path:
        FLAGS.ckpt_path = FLAGS.log_path

    if not FLAGS.metrics_path:
        FLAGS.metrics_path = FLAGS.log_path

    # HACK: The "use_encode" flag will be deprecated. Instead use something like encode_style=LSTM.
    if FLAGS.use_encode:
        FLAGS.encode_style = "LSTM"

    if FLAGS.model_type == "CBOW" or FLAGS.model_type == "RNN":
        FLAGS.num_samples = 0


def init_model(FLAGS, logger, initial_embeddings, vocab_size, num_classes, data_manager):
    # Choose model.
    logger.Log("Building model.")
    if FLAGS.model_type == "CBOW":
        build_model = spinn.cbow.build_model
    elif FLAGS.model_type == "RNN":
        build_model = spinn.plain_rnn.build_model
    elif FLAGS.model_type == "SPINN":
        build_model = spinn.fat_stack.build_model
    elif FLAGS.model_type == "RLSPINN":
        build_model = spinn.rl_spinn.build_model
    elif FLAGS.model_type == "RAESPINN":
        build_model = spinn.rae_spinn.build_model
    elif FLAGS.model_type == "GENSPINN":
        build_model = spinn.gen_spinn.build_model
    else:
        raise Exception("Requested unimplemented model type %s" % FLAGS.model_type)

    model = build_model(data_manager, initial_embeddings, vocab_size, num_classes, FLAGS)

    # Build optimizer.
    if FLAGS.optimizer_type == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=FLAGS.learning_rate, betas=(0.9, 0.999), eps=1e-08)
    elif FLAGS.optimizer_type == "RMSprop":
        optimizer = optim.RMSprop(model.parameters(), lr=FLAGS.learning_rate, eps=1e-08)
    else:
        raise NotImplementedError

    # Build trainer.
    trainer = ModelTrainer(model, optimizer)

    # Print model size.
    logger.Log("Architecture: {}".format(model))
    total_params = sum([reduce(lambda x, y: x * y, w.size(), 1.0) for w in model.parameters()])
    logger.Log("Total params: {}".format(total_params))

    return model, optimizer, trainer


def main_loop(FLAGS, model, optimizer, trainer, training_data_iter, eval_iterators, logger, step, best_dev_error):
    # Accumulate useful statistics.
    A = Accumulator(maxlen=FLAGS.deque_length)
    M = MetricsWriter(os.path.join(FLAGS.metrics_path, FLAGS.experiment_name))

    # Checkpoint paths.
    standard_checkpoint_path = get_checkpoint_path(FLAGS.ckpt_path, FLAGS.experiment_name)
    best_checkpoint_path = get_checkpoint_path(FLAGS.ckpt_path, FLAGS.experiment_name, best=True)

    # Build log format strings.
    model.train()
    X_batch, transitions_batch, y_batch, num_transitions_batch, structure_transitions, train_ids = get_batch(training_data_iter.next())
    model(X_batch, transitions_batch, y_batch,
            use_internal_parser=FLAGS.use_internal_parser,
            validate_transitions=FLAGS.validate_transitions
            )

    logger.Log("")
    logger.Log("# ----- BEGIN: Log Configuration ----- #")

    # Preview train string template.
    train_str = train_format(model)
    logger.Log("Train-Format: {}".format(train_str))
    train_extra_str = train_extra_format(model)
    logger.Log("Train-Extra-Format: {}".format(train_extra_str))

    # Preview eval string template.
    eval_str = eval_format(model)
    logger.Log("Eval-Format: {}".format(eval_str))
    eval_extra_str = eval_extra_format(model)
    logger.Log("Eval-Extra-Format: {}".format(eval_extra_str))

    logger.Log("# ----- END: Log Configuration ----- #")
    logger.Log("")

    # Train.
    logger.Log("Training.")

    # New Training Loop
    progress_bar = SimpleProgressBar(msg="Training", bar_length=60, enabled=FLAGS.show_progress_bar)
    progress_bar.step(i=0, total=FLAGS.statistics_interval_steps)

    for step in range(step, FLAGS.training_steps):
        model.train()

        start = time.time()

        batch = get_batch(training_data_iter.next())
        X_batch, transitions_batch, y_batch, num_transitions_batch, structure_transitions, train_ids = batch

        total_tokens = sum([(nt+1)/2 for nt in num_transitions_batch.reshape(-1)])

        # Reset cached gradients.
        optimizer.zero_grad()

        if FLAGS.model_type == "RLSPINN":
            model.spinn.epsilon = FLAGS.rl_epsilon * math.exp(-step/FLAGS.rl_epsilon_decay)

        # Run model.
        output = model(X_batch, transitions_batch, y_batch,
            use_internal_parser=FLAGS.use_internal_parser,
            validate_transitions=FLAGS.validate_transitions
            )

        # Normalize output.
        logits = F.log_softmax(output)

        # Calculate class accuracy.
        target = torch.from_numpy(y_batch).long()
        pred = logits.data.max(1)[1].cpu() # get the index of the max log-probability
        class_acc = pred.eq(target).sum() / float(target.size(0))

        # Calculate class loss.
        xent_loss = nn.NLLLoss()(logits, to_gpu(Variable(target, volatile=False)))

        # Optionally calculate transition loss.
        transition_loss = model.transition_loss if hasattr(model, 'transition_loss') else None

        # Extract L2 Cost
        l2_loss = l2_cost(model, FLAGS.l2_lambda) if FLAGS.use_l2_cost else None

        # Accumulate Total Loss Variable
        total_loss = 0.0
        total_loss += xent_loss
        if l2_loss is not None:
            total_loss += l2_loss
        if transition_loss is not None and model.optimize_transition_loss:
            total_loss += transition_loss
        total_loss += auxiliary_loss(model)

        # Backward pass.
        total_loss.backward()

        # Hard Gradient Clipping
        clip = FLAGS.clipping_max_value
        for p in model.parameters():
            if p.requires_grad:
                p.grad.data.clamp_(min=-clip, max=clip)

        # Learning Rate Decay
        if FLAGS.actively_decay_learning_rate:
            optimizer.lr = FLAGS.learning_rate * (FLAGS.learning_rate_decay_per_10k_steps ** (step / 10000.0))

        # Gradient descent step.
        optimizer.step()

        end = time.time()

        total_time = end - start

        train_accumulate(model, A, batch)
        A.add('class_acc', class_acc)
        A.add('total_tokens', total_tokens)
        A.add('total_time', total_time)

        if step % FLAGS.statistics_interval_steps == 0:
            progress_bar.step(i=FLAGS.statistics_interval_steps, total=FLAGS.statistics_interval_steps)
            progress_bar.finish()

            A.add('xent_cost', xent_loss.data[0])
            A.add('l2_cost', l2_loss.data[0])
            stats_args = train_stats(model, optimizer, A, step)

            metric_stats = ['class_acc', 'total_cost', 'transition_acc', 'transition_cost']
            for key in metric_stats:
                M.write(key, stats_args[key], step)

            logger.Log(train_str.format(**stats_args))
            logger.Log(train_extra_str.format(**stats_args))

            if FLAGS.num_samples > 0:
                transition_str = "Samples:"
                transitions_per_example = model.spinn.get_transitions_per_example()
                if model.use_sentence_pair and len(transitions_batch.shape) == 3:
                    transitions_batch = np.concatenate([
                        transitions_batch[:,:,0], transitions_batch[:,:,1]], axis=0)
                for t_idx in range(FLAGS.num_samples):
                    gold = transitions_batch[t_idx]
                    pred = transitions_per_example[t_idx]
                    _, crossing = evalb.crossing(gold, pred)
                    transition_str += "\n{}. crossing={}".format(t_idx, crossing)
                    transition_str += "\n     g{}".format("".join(map(str, gold)))
                    transition_str += "\n     p{}".format("".join(map(str, pred)))
                logger.Log(transition_str)

        if step > 0 and step % FLAGS.eval_interval_steps == 0:
            for index, eval_set in enumerate(eval_iterators):
                acc, tacc = evaluate(model, eval_set, logger, step)
                if FLAGS.ckpt_on_best_dev_error and index == 0 and (1 - acc) < 0.99 * best_dev_error and step > FLAGS.ckpt_step:
                    best_dev_error = 1 - acc
                    logger.Log("Checkpointing with new best dev accuracy of %f" % acc)
                    trainer.save(best_checkpoint_path, step, best_dev_error)
                if index == 0:
                    M.write('eval_class_acc', acc, step)
                    M.write('eval_transition_acc', tacc, step)
            progress_bar.reset()

        if step > FLAGS.ckpt_step and step % FLAGS.ckpt_interval_steps == 0:
            logger.Log("Checkpointing.")
            trainer.save(standard_checkpoint_path, step, best_dev_error)

        progress_bar.step(i=step % FLAGS.statistics_interval_steps, total=FLAGS.statistics_interval_steps)
