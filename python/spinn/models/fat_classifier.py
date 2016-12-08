"""From the project root directory (containing data files), this can be run with:

Boolean logic evaluation:
python -m spinn.models.fat_classifier --training_data_path ../bl-data/pbl_train.tsv \
       --eval_data_path ../bl-data/pbl_dev.tsv

SST sentiment (Demo only, model needs a full GloVe embeddings file to do well):
python -m spinn.models.fat_classifier --data_type sst --training_data_path sst-data/train.txt \
       --eval_data_path sst-data/dev.txt --embedding_data_path spinn/tests/test_embedding_matrix.5d.txt \
       --model_dim 10 --word_embedding_dim 5

SNLI entailment (Demo only, model needs a full GloVe embeddings file to do well):
python -m spinn.models.fat_classifier --data_type snli --training_data_path snli_1.0/snli_1.0_dev.jsonl \
       --eval_data_path snli_1.0/snli_1.0_dev.jsonl --embedding_data_path spinn/tests/test_embedding_matrix.5d.txt \
       --model_dim 10 --word_embedding_dim 5

Note: If you get an error starting with "TypeError: ('Wrong number of dimensions..." during development,
    there may already be a saved checkpoint in ckpt_path that matches the name of the model you're developing.
    Move or delete it as appropriate.
"""

from functools import partial
import os
import pprint
import sys
import time
from collections import deque

# Chainer Type Check is a major bottleneck for small networks,
# sometimes giving more than 4x speedup by turning it off.
# We turn type check off by default, but can set an env var to
# turn it on if debugging.
if os.environ.get('FORCE_CHAINER_TYPE_CHECK', '0') == '0':
    os.environ['CHAINER_TYPE_CHECK'] = '0'

import gflags
import numpy as np

from spinn import afs_safe_logger
from spinn import util
from spinn.data.arithmetic import load_simple_data
from spinn.data.boolean import load_boolean_data
from spinn.data.sst import load_sst_data
from spinn.data.snli import load_snli_data
from spinn.util.data import SimpleProgressBar
from spinn.util.chainer_blocks import gradient_check, l2_cost, flatten

import spinn.fat_stack
import spinn.cbow
import spinn.nti

# Try to avoid chainer imports as much as possible.
from chainer import optimizers
import chainer.functions as F

from spinn.util.data import print_tree
from sklearn import metrics


FLAGS = gflags.FLAGS


def build_sentence_pair_model(model_cls, trainer_cls, vocab_size, model_dim, word_embedding_dim,
                              seq_length, num_classes, initial_embeddings, use_sentence_pair,
                              gpu):
    model = model_cls(model_dim, word_embedding_dim, vocab_size,
             seq_length, initial_embeddings, num_classes, mlp_dim=1024,
             input_keep_rate=FLAGS.embedding_keep_rate,
             classifier_keep_rate=FLAGS.semantic_classifier_keep_rate,
             use_input_dropout=FLAGS.use_input_dropout,
             use_input_norm=FLAGS.use_input_norm,
             tracker_dropout_rate=FLAGS.tracker_dropout_rate,
             use_tracker_dropout=FLAGS.use_tracker_dropout,
             use_classifier_norm=FLAGS.use_classifier_norm,
             tracking_lstm_hidden_dim=FLAGS.tracking_lstm_hidden_dim,
             transition_weight=FLAGS.transition_weight,
             use_tracking_lstm=FLAGS.use_tracking_lstm,
             use_shift_composition=FLAGS.use_shift_composition,
             use_history=FLAGS.use_history,
             save_stack=FLAGS.save_stack,
             use_sentence_pair=use_sentence_pair,
             gpu=gpu,
             use_reinforce=FLAGS.use_reinforce,
             use_skips=FLAGS.use_skips,
             use_encode=FLAGS.use_encode,
             projection_dim=FLAGS.projection_dim,
            )

    classifier_trainer = trainer_cls(model, gpu=gpu)

    return classifier_trainer


def build_rewards(logits, y, xent_reward=False):
    if xent_reward:
        return np.mean(logits.data[np.arange(y.shape[0]), y])
    else:
        return metrics.accuracy_score(logits.data.argmax(axis=1), y)


def hamming_distance(s1, s2):
    """ source: https://en.wikipedia.org/wiki/Hamming_distance
        Return the Hamming distance between equal-length sequences
    """
    if len(s1) != len(s2):
        raise ValueError("Undefined for sequences of unequal length")
    return sum(el1 != el2 for el1, el2 in zip(s1, s2))


def evaluate(classifier_trainer, eval_set, logger, step, eval_data_limit=-1,
             use_internal_parser=False, vocabulary=None):
    # Evaluate
    acc_accum = 0.0
    action_acc_accum = 0.0
    eval_batches = 0.0
    total_batches = len(eval_set[1])
    progress_bar = SimpleProgressBar(msg="Run Eval", bar_length=60, enabled=FLAGS.show_progress_bar)
    progress_bar.step(0, total=total_batches)

    if vocabulary:
        inv_vocab = {v: k for k, v in vocabulary.iteritems()}

    if FLAGS.print_tree:
        graph_path = "graphs" if not FLAGS.use_reinforce else "graphs-rl"
        summaries_file = "{}/summaries.txt".format(graph_path)
        with open(summaries_file, "w") as f:
            f.write("id,hamming,gold,pred\n")

    for i, (eval_X_batch, eval_transitions_batch, eval_y_batch, eval_num_transitions_batch) in enumerate(eval_set[1]):
        # Calculate Local Accuracies
        if eval_data_limit == -1 or i < eval_data_limit:
            ret = classifier_trainer.forward({
                "sentences": eval_X_batch,
                "transitions": eval_transitions_batch,
                }, eval_y_batch, train=False, predict=False,
                use_internal_parser=use_internal_parser,
                validate_transitions=FLAGS.validate_transitions,
                use_random=FLAGS.use_random)
            y, loss, class_loss, transition_acc, transition_loss = ret
            acc_value = float(classifier_trainer.model.accuracy.data)
            action_acc_value = transition_acc
        else:
            break

        # Update Aggregate Accuracies
        acc_accum += acc_value
        action_acc_accum += action_acc_value
        eval_batches += 1.0

        if FLAGS.print_tree:
            memories = classifier_trainer.model.spinn.memories
            all_preds = [el['preds'] for el in memories]
            for ii in range(len(eval_X_batch)):
                print(i, ii)
                sentence = eval_X_batch[ii]
                sentence = [s for s in sentence if s != 0]
                sentence = [inv_vocab[s] for s in sentence]
                transitions = eval_transitions_batch[ii]
                offset = len(transitions)
                transitions = [t for t in transitions if t != 2]
                offset = offset - len(transitions)
                preds = np.array(all_preds[offset:])[:, ii]

                tree_id = "{:03}_{:03}".format(i, ii)

                print_tree(sentence, transitions, "{}/gold/tree_{}.png".format(graph_path, tree_id))
                print_tree(sentence, preds, "{}/pred/tree_{}.png".format(graph_path, tree_id))

                with open(summaries_file, "a") as f:
                    f.write("{},{},{},{}\n".format(
                        tree_id,
                        hamming_distance(transitions, preds),
                        "".join([str(t) for t in transitions]),
                        "".join([str(t) for t in preds]),
                        ))

        # Print Progress
        progress_bar.step(i+1, total=total_batches)
    progress_bar.finish()
    logger.Log("Step: %i\tEval acc: %f\t %f\t%s" %
              (step, acc_accum / eval_batches, action_acc_accum / eval_batches, eval_set[0]))
    return acc_accum / eval_batches


def reinforce(optimizer, lr, baseline, mu, reward, memories):
    statistics = zip(*[
        (m["hyp_acc"], m["truth_acc"], m["hyp_xent"], m["truth_xent"])
        for m in memories])

    statistics = [
        F.squeeze(F.concat([F.expand_dims(ss, 1) for ss in s], axis=0))
        if isinstance(s[0], Variable) else
        np.array(reduce(lambda x, y: x + y.tolist(), s, []))
        for s in statistics]

    hyp_acc, truth_acc, hyp_xent, truth_xent = statistics

    transition_loss = F.softmax_cross_entropy(
                hyp_xent, truth_xent.astype(np.int32),
                normalize=False) * reward

    new_lr = (lr*(reward - baseline))
    baseline = baseline*(1-mu)+mu*reward

    transition_loss.backward()
    transition_loss.unchain_backward()
    optimizer.lr = new_lr
    optimizer.update()

    return new_lr, baseline


def run(only_forward=False):
    logger = afs_safe_logger.Logger(os.path.join(FLAGS.log_path, FLAGS.experiment_name) + ".log")

    if FLAGS.data_type == "bl":
        data_manager = load_boolean_data
    elif FLAGS.data_type == "sst":
        data_manager = load_sst_data
    elif FLAGS.data_type == "snli":
        data_manager = load_snli_data
    elif FLAGS.data_type == "arithmetic":
        data_manager = load_simple_data
    else:
        logger.Log("Bad data type.")
        return

    pp = pprint.PrettyPrinter(indent=4)
    logger.Log("Flag values:\n" + pp.pformat(FLAGS.FlagValuesDict()))

    # Load the data.
    raw_training_data, vocabulary = data_manager.load_data(
        FLAGS.training_data_path)

    # Load the eval data.
    raw_eval_sets = []
    if FLAGS.eval_data_path:
        for eval_filename in FLAGS.eval_data_path.split(":"):
            eval_data, _ = data_manager.load_data(eval_filename)
            raw_eval_sets.append((eval_filename, eval_data))

    # Prepare the vocabulary.
    if not vocabulary:
        logger.Log("In open vocabulary mode. Using loaded embeddings without fine-tuning.")
        train_embeddings = False
        vocabulary = util.BuildVocabulary(
            raw_training_data, raw_eval_sets, FLAGS.embedding_data_path, logger=logger,
            sentence_pair_data=data_manager.SENTENCE_PAIR_DATA)
    else:
        logger.Log("In fixed vocabulary mode. Training embeddings.")
        train_embeddings = True

    # Load pretrained embeddings.
    if FLAGS.embedding_data_path:
        logger.Log("Loading vocabulary with " + str(len(vocabulary))
                   + " words from " + FLAGS.embedding_data_path)
        initial_embeddings = util.LoadEmbeddingsFromASCII(
            vocabulary, FLAGS.word_embedding_dim, FLAGS.embedding_data_path)
    else:
        initial_embeddings = None

    # Trim dataset, convert token sequences to integer sequences, crop, and
    # pad.
    logger.Log("Preprocessing training data.")
    training_data = util.PreprocessDataset(
        raw_training_data, vocabulary, FLAGS.seq_length, data_manager, eval_mode=False, logger=logger,
        sentence_pair_data=data_manager.SENTENCE_PAIR_DATA,
        for_rnn=FLAGS.model_type == "RNN" or FLAGS.model_type == "CBOW",
        use_left_padding=FLAGS.use_left_padding)
    training_data_iter = util.MakeTrainingIterator(
        training_data, FLAGS.batch_size, FLAGS.smart_batching, FLAGS.use_peano)

    eval_iterators = []
    for filename, raw_eval_set in raw_eval_sets:
        logger.Log("Preprocessing eval data: " + filename)
        e_X, e_transitions, e_y, e_num_transitions = util.PreprocessDataset(
            raw_eval_set, vocabulary, FLAGS.seq_length, data_manager, eval_mode=True, logger=logger,
            sentence_pair_data=data_manager.SENTENCE_PAIR_DATA,
            for_rnn=FLAGS.model_type == "RNN" or FLAGS.model_type == "CBOW",
            use_left_padding=FLAGS.use_left_padding)
        eval_it = util.MakeEvalIterator((e_X, e_transitions, e_y, e_num_transitions),
            FLAGS.batch_size, shuffle=FLAGS.shuffle_eval, rseed=FLAGS.shuffle_eval_seed)
        eval_iterators.append((filename, eval_it))

    # Set up the placeholders.

    logger.Log("Building model.")

    if FLAGS.model_type == "CBOW":
        model_module = spinn.cbow
    elif FLAGS.model_type == "RNN":
        model_module = spinn.plain_rnn_chainer
    elif FLAGS.model_type == "NTI":
        model_module = spinn.nti
    elif FLAGS.model_type == "SPINN":
        model_module = spinn.fat_stack
    else:
        raise Exception("Requested unimplemented model type %s" % FLAGS.model_type)


    if data_manager.SENTENCE_PAIR_DATA:
        if hasattr(model_module, 'SentencePairTrainer') and hasattr(model_module, 'SentencePairModel'):
            trainer_cls = model_module.SentencePairTrainer
            model_cls = model_module.SentencePairModel
        else:
            raise Exception("Unimplemented for model type %s" % FLAGS.model_type)

        num_classes = len(data_manager.LABEL_MAP)
        use_sentence_pair = True
        classifier_trainer = build_sentence_pair_model(model_cls, trainer_cls,
                              len(vocabulary), FLAGS.model_dim, FLAGS.word_embedding_dim,
                              FLAGS.seq_length, num_classes, initial_embeddings,
                              use_sentence_pair,
                              FLAGS.gpu)
    else:
        if hasattr(model_module, 'SentenceTrainer') and hasattr(model_module, 'SentenceModel'):
            trainer_cls = model_module.SentenceTrainer
            model_cls = model_module.SentenceModel
        else:
            raise Exception("Unimplemented for model type %s" % FLAGS.model_type)

        num_classes = len(data_manager.LABEL_MAP)
        use_sentence_pair = False
        classifier_trainer = build_sentence_pair_model(model_cls, trainer_cls,
                              len(vocabulary), FLAGS.model_dim, FLAGS.word_embedding_dim,
                              FLAGS.seq_length, num_classes, initial_embeddings,
                              use_sentence_pair,
                              FLAGS.gpu)

    if ".ckpt" in FLAGS.ckpt_path:
        checkpoint_path = FLAGS.ckpt_path
    else:
        checkpoint_path = os.path.join(FLAGS.ckpt_path, FLAGS.experiment_name + ".ckpt")
    if os.path.isfile(checkpoint_path):
        # TODO: Check that resuming works fine with tf summaries.
        logger.Log("Found checkpoint, restoring.")
        step, best_dev_error = classifier_trainer.load(checkpoint_path)
        logger.Log("Resuming at step: {} with best dev accuracy: {}".format(step, 1. - best_dev_error))
    else:
        assert not only_forward, "Can't run an eval-only run without a checkpoint. Supply a checkpoint."
        step = 0
        best_dev_error = 1.0

    if FLAGS.write_summaries:
        from spinn.tf_logger import TFLogger
        train_summary_logger = TFLogger(summary_dir=os.path.join(FLAGS.summary_dir, FLAGS.experiment_name, 'train'))
        dev_summary_logger = TFLogger(summary_dir=os.path.join(FLAGS.summary_dir, FLAGS.experiment_name, 'dev'))

    # Do an evaluation-only run.
    if only_forward:
        for index, eval_set in enumerate(eval_iterators):
            acc = evaluate(classifier_trainer, eval_set, logger, step,
                use_internal_parser=FLAGS.use_internal_parser,
                vocabulary=vocabulary,
                eval_data_limit=FLAGS.eval_data_limit)
    else:
         # Train
        logger.Log("Training.")

        classifier_trainer.init_optimizer(
            clip=FLAGS.clipping_max_value, decay=FLAGS.l2_lambda,
            lr=FLAGS.learning_rate,
            )

        model = classifier_trainer.optimizer.target

        if FLAGS.use_reinforce:
            optimizer_lr = 0.01
            baseline = 0
            mu = 0.1
            transition_optimizer = optimizers.SGD(lr=optimizer_lr)
            transition_optimizer.setup(model.spinn.tracker)

        # New Training Loop
        progress_bar = SimpleProgressBar(msg="Training", bar_length=60, enabled=FLAGS.show_progress_bar)
        accum_class_acc = deque(maxlen=FLAGS.deq_length)
        accum_preds = deque(maxlen=FLAGS.deq_length)
        accum_truth = deque(maxlen=FLAGS.deq_length)
        printed_total_weights = False
        for step in range(step, FLAGS.training_steps):
            X_batch, transitions_batch, y_batch, _ = training_data_iter.next()

            # Reset cached gradients.
            classifier_trainer.optimizer.zero_grads()

            # Calculate loss and update parameters.
            ret = classifier_trainer.forward({
                "sentences": X_batch,
                "transitions": transitions_batch,
                }, y_batch, train=True, predict=False,
                    validate_transitions=FLAGS.validate_transitions,
                    use_internal_parser=FLAGS.use_internal_parser,
                    use_random=FLAGS.use_random)
            y, xent_loss, class_acc, transition_acc, transition_loss = ret

            if not printed_total_weights:
                printed_total_weights = True
                def prod(l):
                    return reduce(lambda x, y: x * y, l, 1.0)
                total_weights = sum([prod(w.shape) for w in model.params()])
                logger.Log("Total Weights: {}".format(total_weights))

            # Accumulate stats for confusion matrix.
            preds = [m["preds_cm"] for m in model.spinn.memories]
            truth = [m["truth_cm"] for m in model.spinn.memories]
            accum_preds.append(preds)
            accum_truth.append(truth)

            if FLAGS.use_reinforce:
                rewards = build_rewards(y, y_batch)
                logger.Log("\nReward :"+str(rewards))

            # Boilerplate for calculating loss.
            transition_cost_val = transition_loss.data if transition_loss is not None else 0.0
            accum_class_acc.append(class_acc)

            # Extract L2 Cost
            l2_loss = l2_cost(model, FLAGS.l2_lambda)

            # Accumulate Total Loss Data
            total_cost_val = 0.0
            total_cost_val += xent_loss.data
            total_cost_val += l2_loss.data
            if not FLAGS.use_reinforce:
                total_cost_val += transition_cost_val

            # Accumulate Total Loss Variable
            total_loss = 0.0
            total_loss += xent_loss
            total_loss += l2_loss
            if hasattr(transition_loss, 'backward') and not FLAGS.use_reinforce:
                total_loss += transition_loss

            # [(n,w.grad) for n,w in model.namedparams()]

            total_loss.backward()

            if FLAGS.gradient_check:
                def get_loss():
                    _, check_loss, _, _ = classifier_trainer.forward({
                    "sentences": X_batch,
                    "transitions": transitions_batch,
                    }, y_batch, train=True, predict=False)
                    return check_loss
                gradient_check(classifier_trainer.model, get_loss)

            try:
                classifier_trainer.update()
            except:
                import ipdb; ipdb.set_trace()
                pass

            if FLAGS.use_reinforce:
                transition_optimizer.zero_grads()
                optimizer_lr, baseline = reinforce(transition_optimizer, optimizer_lr, baseline, mu, rewards, model.spinn.memories)

            # Accumulate accuracy for current interval.
            acc_val = float(classifier_trainer.model.accuracy.data)

            if FLAGS.write_summaries:
                train_summary_logger.log(step=step, loss=total_cost_val, accuracy=acc_val)

            progress_bar.step(
                i=max(0, step-1) % FLAGS.statistics_interval_steps + 1,
                total=FLAGS.statistics_interval_steps)

            if step % FLAGS.statistics_interval_steps == 0:
                progress_bar.finish()
                avg_class_acc = np.array(accum_class_acc).mean()
                all_preds = flatten(accum_preds)
                all_truth = flatten(accum_truth)
                if transition_loss is not None:
                    avg_trans_acc = metrics.accuracy_score(all_preds, all_truth) if len(all_preds) > 0 else 0.0
                else:
                    avg_trans_acc = 0.0
                logger.Log(
                    "Step: %i\tAcc: %f\t%f\tCost: %5f %5f %5f %5f"
                    % (step, avg_class_acc, avg_trans_acc, total_cost_val, xent_loss.data, transition_cost_val, l2_loss.data))
                if FLAGS.print_confusion_matrix:
                    cm = metrics.confusion_matrix(
                        np.array(all_preds),
                        np.array(all_truth),
                        )
                    logger.Log("{}".format(cm))
                    cm = cm.astype(np.float32) / cm.sum(axis=1)[:, np.newaxis]
                    logger.Log("{}".format(cm))
                accum_class_acc.clear()
                accum_preds.clear()
                accum_truth.clear()

            if step > 0 and step % FLAGS.ckpt_interval_steps == 0:
                for index, eval_set in enumerate(eval_iterators):
                    acc = evaluate(classifier_trainer, eval_set, logger, step,
                        eval_data_limit=-1, use_internal_parser=FLAGS.use_internal_parser)
                    if FLAGS.ckpt_on_best_dev_error and index == 0 and (1 - acc) < best_dev_error and step > FLAGS.ckpt_step:
                        best_dev_error = 1 - acc
                        logger.Log("Checkpointing with new best dev accuracy of %f" % acc)
                        classifier_trainer.save(checkpoint_path, step, best_dev_error)
                    if FLAGS.write_summaries:
                        dev_summary_logger.log(step=step, loss=0.0, accuracy=acc)
                progress_bar.reset()
            elif step > 0 and step % FLAGS.eval_interval_steps == 0:
                for index, eval_set in enumerate(eval_iterators):
                    acc = evaluate(classifier_trainer, eval_set, logger, step, eval_data_limit=FLAGS.eval_data_limit)
                    if FLAGS.ckpt_on_best_dev_error and index == 0 and (1 - acc) < 0.99 * best_dev_error and step > FLAGS.ckpt_step:
                        best_dev_error = 1 - acc
                        logger.Log("Checkpointing with new best dev accuracy of %f" % acc)
                        classifier_trainer.save(checkpoint_path, step, best_dev_error)
                    if FLAGS.write_summaries:
                        dev_summary_logger.log(step=step, loss=0.0, accuracy=acc)
                progress_bar.reset()


            if FLAGS.profile and step >= FLAGS.profile_steps:
                break


if __name__ == '__main__':
    # Debug settings.
    gflags.DEFINE_bool("debug", False, "Set to True to disable debug_mode and type_checking.")
    gflags.DEFINE_bool("print_confusion_matrix", False, "Periodically print CM on transitions.")
    gflags.DEFINE_bool("gradient_check", False, "Randomly check that gradients match estimates.")
    gflags.DEFINE_bool("profile", False, "Set to True to quit after a few batches.")
    gflags.DEFINE_bool("write_summaries", False, "Toggle which controls whether summaries are written.")
    gflags.DEFINE_bool("show_progress_bar", True, "Turn this off when running experiments on HPC.")
    gflags.DEFINE_bool("show_intermediate_stats", False, "Print stats more frequently than regular interval."
                                                         "Mostly to retain timing with progress bar")
    gflags.DEFINE_integer("profile_steps", 3, "Specify how many steps to profile.")
    gflags.DEFINE_string("branch_name", "", "")
    gflags.DEFINE_string("sha", "", "")
    gflags.DEFINE_boolean("print_tree", False, "Print trees to file.")

    # Experiment naming.
    gflags.DEFINE_string("experiment_name", "", "")

    # Data types.
    gflags.DEFINE_enum("data_type", "bl", ["bl", "sst", "snli", "arithmetic"],
        "Which data handler and classifier to use.")

    # Where to store checkpoints
    gflags.DEFINE_string("ckpt_path", ".", "Where to save/load checkpoints. Can be either "
        "a filename or a directory. In the latter case, the experiment name serves as the "
        "base for the filename.")
    gflags.DEFINE_string("log_path", ".", "A directory in which to write logs.")
    gflags.DEFINE_string("summary_dir", ".", "A directory in which to write summaries.")

    # Data settings.
    gflags.DEFINE_string("training_data_path", None, "")
    gflags.DEFINE_string("eval_data_path", None, "Can contain multiple file paths, separated "
        "using ':' tokens. The first file should be the dev set, and is used for determining "
        "when to save the early stopping 'best' checkpoints.")
    gflags.DEFINE_integer("ckpt_step", 1000, "Steps to run before considering saving checkpoint.")
    gflags.DEFINE_integer("deq_length", 10, "Max trailing examples to use for statistics.")
    gflags.DEFINE_integer("seq_length", 30, "")
    gflags.DEFINE_integer("eval_seq_length", 30, "")
    gflags.DEFINE_boolean("use_internal_parser", False, "Use predicted parse rather than ground truth.")
    gflags.DEFINE_boolean("smart_batching", True, "Organize batches using sequence length.")
    gflags.DEFINE_boolean("use_peano", True, "A mind-blowing sorting key.")
    gflags.DEFINE_integer("eval_data_limit", -1, "Truncate evaluation set. -1 indicates no truncation.")
    gflags.DEFINE_boolean("shuffle_eval", False, "Shuffle evaluation data.")
    gflags.DEFINE_integer("shuffle_eval_seed", 123, "Seed shuffling of eval data.")
    gflags.DEFINE_string("embedding_data_path", None,
        "If set, load GloVe-formatted embeddings from here.")

    # Model architecture settings.
    gflags.DEFINE_enum("model_type", "RNN",
                       ["CBOW", "RNN", "SPINN", "NTI"],
                       "")
    gflags.DEFINE_boolean("allow_gt_transitions_in_eval", False,
        "Whether to use ground truth transitions in evaluation when appropriate "
        "(i.e., in Model 1 and Model 2S.)")
    gflags.DEFINE_integer("gpu", -1, "")
    gflags.DEFINE_integer("model_dim", 8, "")
    gflags.DEFINE_integer("word_embedding_dim", 8, "")

    gflags.DEFINE_float("transition_weight", None, "")
    gflags.DEFINE_integer("tracking_lstm_hidden_dim", 4, "")

    gflags.DEFINE_boolean("use_reinforce", False, "Use RL to provide tracking lstm gradients")
    gflags.DEFINE_boolean("xent_reward", False, "Use cross entropy instead of accuracy as RL reward")
    gflags.DEFINE_boolean("use_encode", False, "Encode output of projection layer using bidirectional RNN")
    gflags.DEFINE_integer("projection_dim", -1, "Dimension for projection network.")
    gflags.DEFINE_boolean("use_shift_composition", True, "")
    gflags.DEFINE_boolean("use_history", False, "")
    gflags.DEFINE_boolean("use_skips", False, "Pad transitions with SKIP actions.")
    gflags.DEFINE_boolean("use_left_padding", True, "Pad transitions only on the RHS.")
    gflags.DEFINE_boolean("validate_transitions", True, "Constrain predicted transitions to ones"
                                                        "that give a valid parse tree.")
    gflags.DEFINE_boolean("save_stack", False, "")
    gflags.DEFINE_boolean("use_tracking_lstm", True,
                          "Whether to use LSTM in the tracking unit")
    gflags.DEFINE_float("semantic_classifier_keep_rate", 0.9,
        "Used for dropout in the semantic task classifier.")
    gflags.DEFINE_float("embedding_keep_rate", 0.9,
        "Used for dropout on transformed embeddings.")
    gflags.DEFINE_boolean("use_input_dropout", False, "")
    gflags.DEFINE_boolean("use_random", False, "When predicting parse, rather than logits,"
                                               "use a uniform distribution over actions.")
    gflags.DEFINE_boolean("use_input_norm", False, "")
    gflags.DEFINE_boolean("use_tracker_dropout", False, "")
    gflags.DEFINE_boolean("use_classifier_norm", False, "")
    gflags.DEFINE_float("tracker_dropout_rate", 0.1, "")
    gflags.DEFINE_boolean("lstm_composition", True, "")

    # Optimization settings.
    gflags.DEFINE_integer("training_steps", 500000, "Stop training after this point.")
    gflags.DEFINE_integer("batch_size", 32, "SGD minibatch size.")
    gflags.DEFINE_float("learning_rate", 0.001, "Used in RMSProp.")
    gflags.DEFINE_float("learning_rate_decay_per_10k_steps", 0.75, "Used in RMSProp.")
    gflags.DEFINE_float("clipping_max_value", 5.0, "")
    gflags.DEFINE_float("l2_lambda", 1e-5, "")
    gflags.DEFINE_float("init_range", 0.005, "Mainly used for softmax parameters. Range for uniform random init.")
    gflags.DEFINE_float("transition_cost_scale", 1.0, "Multiplied by the transition cost.")

    # Display settings.
    gflags.DEFINE_integer("statistics_interval_steps", 100, "Print training set results at this interval.")
    gflags.DEFINE_integer("eval_interval_steps", 100, "Evaluate at this interval.")

    gflags.DEFINE_integer("ckpt_interval_steps", 5000, "Update the checkpoint on disk at this interval.")
    gflags.DEFINE_boolean("ckpt_on_best_dev_error", True, "If error on the first eval set (the dev set) is "
        "at most 0.99 of error at the previous checkpoint, save a special 'best' checkpoint.")

    # Evaluation settings
    gflags.DEFINE_boolean("expanded_eval_only_mode", False,
        "If set, a checkpoint is loaded and a forward pass is done to get the predicted "
        "transitions. The inferred parses are written to the supplied file(s) along with example-"
        "by-example accuracy information. Requirements: Must specify checkpoint path.")
    gflags.DEFINE_string("eval_output_paths", None,
        "Used when expanded_eval_only_mode is set. The number of supplied paths should be same"
        "as the number of eval sets.")
    gflags.DEFINE_boolean("write_predicted_label", False,
        "Write the predicted labels in a <eval_output_name>.lbl file.")

    # Parse command line flags.
    FLAGS(sys.argv)

    if not FLAGS.experiment_name:
        timestamp = str(int(time.time()))
        FLAGS.experiment_name = "{}-{}-{}".format(
            FLAGS.data_type,
            FLAGS.model_type,
            timestamp,
            )

    if FLAGS.debug:
        chainer.set_debug(True)

    if not FLAGS.branch_name:
        FLAGS.branch_name = os.popen('git rev-parse --abbrev-ref HEAD').read().strip()

    if not FLAGS.sha:
        FLAGS.sha = os.popen('git rev-parse HEAD').read().strip()

    run(only_forward=FLAGS.expanded_eval_only_mode)
