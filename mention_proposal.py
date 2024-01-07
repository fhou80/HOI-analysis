import torch
import torch.nn as nn
#from transformers import BertModel
from bert_model import BertModel
from transformers import ElectraModel
import util
import logging
from collections import Iterable
import numpy as np
import torch.nn.init as init
import sys
from datetime import datetime
from os.path import join
from tensorize import CorefDataProcessor
from torch.utils.tensorboard import SummaryWriter
from transformers import AdamW
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
import time
import random
import pickle


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger()


class MentionProposal(nn.Module):
    def __init__(self, config, device, config_name, num_genres=None):
        super().__init__()
        self.config = config
        self.device = device
        self.config_name = config_name

        self.num_genres = num_genres if num_genres else len(config['genres'])
        self.max_seg_len = config['max_segment_len']  #384 ,512
        self.max_span_width = config['max_span_width'] #30
        assert config['loss_type'] in ['marginalized', 'hinge']

        # Model
        self.dropout = nn.Dropout(p=config['dropout_rate'])
        if 'electra' in self.config_name:
            self.bert = ElectraModel.from_pretrained(config['bert_pretrained_name_or_path'])
            self.bert.to(self.device)
        elif 'spanbert' in self.config_name:
            self.bert = BertModel.from_pretrained(config['bert_pretrained_name_or_path'])

        self.bert_emb_size = self.bert.config.hidden_size
        self.span_emb_size = self.bert_emb_size * 3 #[g_start, g_end, \hat{g}]
        #self.span_emb_size = self.bert_emb_size * 2 #[g_start, g_end]
        if config['use_features']:
            self.span_emb_size += config['feature_emb_size']

        # span width
        self.emb_span_width = self.make_embedding(self.max_span_width) if config['use_features'] else None
        self.emb_span_width_prior = self.make_embedding(self.max_span_width) if config['use_width_prior'] else None

        # for building \hat{g}
        self.mention_token_attn = self.make_ffnn(self.bert_emb_size, 0, output_size=1) if config['model_heads'] else None
        self.span_emb_score_ffnn = self.make_ffnn(self.span_emb_size, [config['ffnn_size']] * config['ffnn_depth'], output_size=1)
        self.span_width_score_ffnn = self.make_ffnn(config['feature_emb_size'], [config['ffnn_size']] * config['ffnn_depth'], output_size=1) if config['use_width_prior'] else None

        self.update_steps = 0  # Internal use for debug
        self.debug = True

    def make_embedding(self, dict_size, std=0.02):
        emb = nn.Embedding(dict_size, self.config['feature_emb_size'])
        init.normal_(emb.weight, std=std)
        return emb

    def make_linear(self, in_features, out_features, bias=True, std=0.02):
        linear = nn.Linear(in_features, out_features, bias)
        init.normal_(linear.weight, std=std)
        if bias:
            init.zeros_(linear.bias)
        return linear

    def make_ffnn(self, feat_size, hidden_size, output_size):
        if hidden_size is None or hidden_size == 0 or hidden_size == [] or hidden_size == [0]:
            return self.make_linear(feat_size, output_size)

        if not isinstance(hidden_size, Iterable):
            hidden_size = [hidden_size]
        ffnn = [self.make_linear(feat_size, hidden_size[0]), nn.ReLU(), self.dropout]
        for i in range(1, len(hidden_size)):
            ffnn += [self.make_linear(hidden_size[i-1], hidden_size[i]), nn.ReLU(), self.dropout]
        ffnn.append(self.make_linear(hidden_size[-1], output_size))
        return nn.Sequential(*ffnn)

    def get_params(self, named=False):
        bert_based_param, task_param = [], []
        for name, param in self.named_parameters():
            if name.startswith('bert'):
                to_add = (name, param) if named else param
                bert_based_param.append(to_add)
            else:
                to_add = (name, param) if named else param
                task_param.append(to_add)
        return bert_based_param, task_param

    def forward(self, *input):
        return self.get_predictions_and_loss(*input)

    def get_predictions_and_loss(self, input_ids, input_mask, speaker_ids, word_ids, ent_ids, subtoken_map, sentence_len, genre, sentence_map,
                                 is_training, gold_starts=None, gold_ends=None, gold_mention_cluster_map=None):
        """ Model and input are already on the device """
        device = self.device
        conf = self.config
        num_gold_spans = 0
        do_loss = False
        self.debug = False
        if gold_mention_cluster_map is not None:
            assert gold_starts is not None
            assert gold_ends is not None
            do_loss = True
            num_gold_spans = gold_ends.shape[0]
            self.debug = True

        # Get token emb
        if 'electra' in self.config_name:
            mention_doc, hidden_layers = self.bert(input_ids, attention_mask=input_mask,
                                                   output_hidden_states=True)  # [num seg, num max tokens, emb size]
            mention_doc_one = hidden_layers[1]
        elif 'spanbert' in self.config_name:
            mention_doc, _, hidden_layers = self.bert(input_ids, attention_mask=input_mask,
                                                      output_hidden_states=True)  # [num seg, num max tokens, emb size]
            mention_doc_one = hidden_layers[1]

        input_mask = input_mask.to(torch.bool)
        mention_doc = mention_doc[input_mask] # [num_word, emb_size]
        mention_doc_one = mention_doc_one[input_mask]
        num_words = mention_doc.shape[0]

        # Get candidate span
        sentence_indices = sentence_map  # [num tokens]
        candidate_starts = torch.unsqueeze(torch.arange(0, num_words, device=device), 1).repeat(1, self.max_span_width)
        candidate_ends = candidate_starts + torch.arange(0, self.max_span_width, device=device)
        candidate_start_sent_idx = sentence_indices[candidate_starts]
        candidate_end_sent_idx = sentence_indices[torch.min(candidate_ends, torch.tensor(num_words - 1, device=device))]
        # candidate span should be in the same sentence
        candidate_mask = (candidate_ends < num_words) & (candidate_start_sent_idx == candidate_end_sent_idx)
        candidate_starts, candidate_ends = candidate_starts[candidate_mask], candidate_ends[candidate_mask]  # [num valid candidates]
        num_candidates = candidate_starts.shape[0]
        #candidate_starts
        #[0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8, 9, 9, 9, 10, 10, 10, 11, 11, 11]
        #candidate_ends
        #[0, 1, 2, 1, 2, 3, 2, 3, 4, 3, 4, 5, 4, 5, 6, 5, 6, 7, 6, 7, 8, 7, 8, 9, 8, 9, 10, 9, 10, 11, 10, 11, 12, 11, 12, 13]

        # Get candidate labels
        # gold_starts = [24  29  33  42  52  57  72  75  88 110 124 124 132 132 142 147 151 157]
        # gold_ends =   [25  29  38  43  58  57  72  75  88 111 125 127 133 135 142 148 152 158]
        # gold_mention_cluster_map = [ 1.  2.  2.  1.  4.  3.  3.  3.  4.  1.  1.  4.  1.  4.  5.  5.  1.  3.]
        if do_loss:
            same_start = (torch.unsqueeze(gold_starts, 1) == torch.unsqueeze(candidate_starts, 0))
            same_end = (torch.unsqueeze(gold_ends, 1) == torch.unsqueeze(candidate_ends, 0))
            same_span = (same_start & same_end).to(torch.long)
            candidate_labels = torch.matmul(torch.unsqueeze(gold_mention_cluster_map, 0).to(torch.float), same_span.to(torch.float))
            candidate_labels = torch.squeeze(candidate_labels.to(torch.long), 0)  # [num candidates]; non-gold span has label 0
        #[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        #0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        #0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        #0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        # Get span embedding
        span_start_emb, span_end_emb = mention_doc[candidate_starts], mention_doc[candidate_ends]
        hybrid_mention_doc = mention_doc * (1 - conf['alpha']) + mention_doc_one * conf['alpha']
        candidate_emb_list = [span_start_emb, span_end_emb]
        # use span width feature
        if conf['use_features']:
            candidate_width_idx = candidate_ends - candidate_starts
            candidate_width_emb = self.emb_span_width(candidate_width_idx)
            candidate_width_emb = self.dropout(candidate_width_emb)
            candidate_emb_list.append(candidate_width_emb)
        # Use attended head or avg token
        candidate_tokens = torch.unsqueeze(torch.arange(0, num_words, device=device), 0).repeat(num_candidates, 1) #[num_words, num_candidates]
        candidate_tokens_mask = (candidate_tokens >= torch.unsqueeze(candidate_starts, 1)) & (candidate_tokens <= torch.unsqueeze(candidate_ends, 1))
        if conf['model_heads']:
            token_attn = torch.squeeze(self.mention_token_attn(hybrid_mention_doc), 1)
        else:
            token_attn = torch.ones(num_words, dtype=torch.float, device=device)  # Use avg if no attention
        candidate_tokens_attn_raw = torch.log(candidate_tokens_mask.to(torch.float)) + torch.unsqueeze(token_attn, 0)
        candidate_tokens_attn = nn.functional.softmax(candidate_tokens_attn_raw, dim=1)
        head_attn_emb = torch.matmul(candidate_tokens_attn, hybrid_mention_doc)
        candidate_emb_list.append(head_attn_emb)
        candidate_span_emb = torch.cat(candidate_emb_list, dim=1)  # [num candidates, new emb size]

        # Get span score
        candidate_mention_scores = torch.squeeze(self.span_emb_score_ffnn(candidate_span_emb), 1)
        if conf['use_width_prior']:
            width_score = torch.squeeze(self.span_width_score_ffnn(self.emb_span_width_prior.weight), 1)
            candidate_width_score = width_score[candidate_width_idx]
            candidate_mention_scores += candidate_width_score

        # Extract top spans
        candidate_idx_sorted_by_score = torch.argsort(candidate_mention_scores, descending=True).tolist()
        candidate_starts_cpu, candidate_ends_cpu = candidate_starts.tolist(), candidate_ends.tolist()
        num_top_spans = int(min(conf['max_num_extracted_spans'], conf['top_span_ratio'] * num_words))
        #num_top_spans = int(min(gold_starts.shape[0], conf['top_span_ratio'] * num_words))
        selected_idx_cpu = self._extract_top_spans(candidate_idx_sorted_by_score, candidate_starts_cpu, candidate_ends_cpu, num_top_spans)
        assert len(selected_idx_cpu) == num_top_spans
        selected_idx = torch.tensor(selected_idx_cpu, device=device, dtype=torch.long)
        top_span_starts, top_span_ends = candidate_starts[selected_idx], candidate_ends[selected_idx]
        #top_span_emb = candidate_span_emb[selected_idx]
        top_span_cluster_ids = candidate_labels[selected_idx] if do_loss else None
        top_span_mention_scores = candidate_mention_scores[selected_idx]

        # find the top ranking gold spans
        _, top_gold_indx = torch.topk(top_span_mention_scores, k=num_gold_spans)
        top_gold_cluster_ids = top_span_cluster_ids[top_gold_indx] if do_loss else None

        if self.debug:
            if self.update_steps % 20 == 0:
                logger.info('---------debug step: %d---------' % self.update_steps)
                rec = (top_span_cluster_ids > 0).sum().cpu().numpy()
                num_top_gold = (top_gold_cluster_ids > 0).sum().cpu().numpy()
                p = rec / num_top_spans * 100
                r = rec / num_gold_spans * 100
                a = num_top_gold / num_gold_spans * 100
                logger.info('precision: %.2f, recall: %.2f, accuracy: %.2f' % (p, r, a))

        self.update_steps += 1
        # Get loss
        #  mention loss
        if do_loss:
            # sigmoid log loss
            gold_mention_scores = top_span_mention_scores[top_span_cluster_ids > 0]
            non_gold_mention_scores = top_span_mention_scores[top_span_cluster_ids == 0]
            loss_mention = -torch.sum(torch.log(torch.sigmoid(gold_mention_scores)))
            loss_mention += -torch.sum(torch.log(1 - torch.sigmoid(non_gold_mention_scores)))
            return loss_mention
            """
            # use margin ranking loss
            loss_func = nn.MarginRankingLoss(0.1)
            input_cri = torch.zeros(num_top_spans).to(self.device)
            target_pos = top_span_cluster_ids > 0
            target_neg = torch.logical_not(target_pos)
            target = target_pos.to(torch.long) + target_neg.to(torch.long) * (-1)
            loss = loss_func(top_span_mention_scores, input_cri, target)
            return loss
            """
        else:
            return top_span_mention_scores, top_span_starts, top_span_ends


    def _extract_top_spans(self, candidate_idx_sorted, candidate_starts, candidate_ends, num_top_spans):
        """ Keep top non-cross-overlapping candidates ordered by scores; compute on CPU because of loop """
        selected_candidate_idx = []
        start_to_max_end, end_to_min_start = {}, {}
        for candidate_idx in candidate_idx_sorted:
            if len(selected_candidate_idx) >= num_top_spans:
                break
            # Perform overlapping check
            span_start_idx = candidate_starts[candidate_idx]
            span_end_idx = candidate_ends[candidate_idx]
            cross_overlap = False
            for token_idx in range(span_start_idx, span_end_idx + 1):
                max_end = start_to_max_end.get(token_idx, -1)
                if token_idx > span_start_idx and max_end > span_end_idx:
                    cross_overlap = True
                    break
                min_start = end_to_min_start.get(token_idx, -1)
                if token_idx < span_end_idx and 0 <= min_start < span_start_idx:
                    cross_overlap = True
                    break
            if not cross_overlap:
                # Pass check; select idx and update dict stats
                selected_candidate_idx.append(candidate_idx)
                max_end = start_to_max_end.get(span_start_idx, -1)
                if span_end_idx > max_end:
                    start_to_max_end[span_start_idx] = span_end_idx
                min_start = end_to_min_start.get(span_end_idx, -1)
                if min_start == -1 or span_start_idx < min_start:
                    end_to_min_start[span_end_idx] = span_start_idx
        # Sort selected candidates by span idx
        selected_candidate_idx = sorted(selected_candidate_idx, key=lambda idx: (candidate_starts[idx], candidate_ends[idx]))
        if len(selected_candidate_idx) < num_top_spans:  # Padding
            selected_candidate_idx += ([selected_candidate_idx[0]] * (num_top_spans - len(selected_candidate_idx)))
        return selected_candidate_idx


class Trainer:
    def __init__(self, config_name, gpu_id=0, seed=None):
        self.name = config_name
        self.name_suffix = datetime.now().strftime('%b%d_%H-%M-%S')
        self.gpu_id = gpu_id
        self.seed = seed

        # Set up config
        self.config = util.parse_config(config_name)
        self.language = 'english'
        self.max_seg_len = self.config['max_segment_len']
        self.max_training_seg = self.config['max_training_sentences']

        # Set up logger
        log_path = join(self.config['log_dir'], 'log_' + self.name_suffix + '.txt')
        logger.addHandler(logging.FileHandler(log_path, 'a'))
        logger.info('Log file path: %s' % log_path)

        # Set up seed
        if seed:
            util.set_seed(seed)

        # Set up device
        self.device = torch.device('cpu' if gpu_id is None else f'cuda:{gpu_id}')

        # Set up data, WordNet vocab and embeddings are in it.
        self.data = CorefDataProcessor(self.config)

    def initialize_model(self, saved_suffix=None):
        model = MentionProposal(self.config, self.device, self.name)
        if saved_suffix:
            self.load_model_checkpoint(model, saved_suffix)
        return model

    def get_cache_path(self):
        cache_path = join(self.config['data_dir'],
                          f'cached.intermediate.tensors.{self.language}.{self.max_seg_len}.{self.max_training_seg}.bin')
        return cache_path

    def cache_mention_proposals(self, saved_suffix):
        model = MentionProposal(self.config, self.device, self.name)
        self.load_model_checkpoint(model, saved_suffix)
        model.to(self.device)
        with torch.no_grad():
            model.eval()
            intermediate_exapmles = {}
            tensorized_examples = {}
            tensorized_examples['trn'], tensorized_examples['dev'], tensorized_examples[
                'tst'] = self.data.get_tensor_examples()
            stored_info = self.data.get_stored_info()

            for split in ['trn', 'dev', 'tst']:
                intermediate_tensors = []
                for doc_key, example in tensorized_examples[split]:
                    example_gpu = [d.to(self.device) for d in example[:10]]
                    top_span_mention_scores, top_span_starts, top_span_ends = model(*example_gpu)
                    doc_tensors = list(example)[:10] + [top_span_mention_scores.to('cpu'), top_span_starts.to('cpu'), top_span_ends.to('cpu')] + list(example)[10:]
                    intermediate_tensors.append((doc_key, tuple(doc_tensors)))
                intermediate_exapmles[split] = intermediate_tensors
        cache_path = self.get_cache_path()

        with open(cache_path, 'wb') as f:
            pickle.dump((intermediate_exapmles, stored_info), f)


    def train(self, model):
        conf = self.config
        logger.info(conf)
        epochs, grad_accum = conf['num_epochs'], conf['gradient_accumulation_steps']

        model.to(self.device)
        logger.info('Model parameters:')
        for name, param in model.named_parameters():
            logger.info('%s: %s' % (name, tuple(param.shape)))

        # Set up tensorboard
        tb_path = join(conf['tb_dir'], self.name + '_' + self.name_suffix)
        tb_writer = SummaryWriter(tb_path, flush_secs=30)
        logger.info('Tensorboard summary path: %s' % tb_path)

        # Set up data
        examples_train, examples_dev, examples_test = self.data.get_tensor_examples()
        stored_info = self.data.get_stored_info()

        # Set up optimizer and scheduler
        total_update_steps = len(examples_train) * epochs // grad_accum
        optimizers = self.get_optimizer(model)
        schedulers = self.get_scheduler(optimizers, total_update_steps)

        # Get model parameters for grad clipping
        bert_param, task_param = model.get_params()

        # Start training
        logger.info('*******************Training*******************')
        logger.info('Num samples: %d' % len(examples_train))
        logger.info('Num epochs: %d' % epochs)
        logger.info('Gradient accumulation steps: %d' % grad_accum)
        logger.info('Total update steps: %d' % total_update_steps)

        loss_during_accum = []  # To compute effective loss at each update
        loss_during_report = 0.0  # Effective loss during logging step
        loss_history = []  # Full history of effective loss; length equals total update steps
        max_f1 = 0
        start_time = time.time()
        model.zero_grad()
        for epo in range(epochs):
            random.shuffle(examples_train)  # Shuffle training set
            for doc_key, example in examples_train:
                # Forward pass
                model.train()
                example_gpu = [d.to(self.device) for d in example]
                # training, only loss
                loss = model(*example_gpu)

                # Backward; accumulate gradients and clip by grad norm
                if grad_accum > 1:
                    loss /= grad_accum
                loss.backward()
                if conf['max_grad_norm']:
                    torch.nn.utils.clip_grad_norm_(bert_param, conf['max_grad_norm'])
                    torch.nn.utils.clip_grad_norm_(task_param, conf['max_grad_norm'])
                loss_during_accum.append(loss.item())

                # Update
                if len(loss_during_accum) % grad_accum == 0:
                    for optimizer in optimizers:
                        optimizer.step()
                    model.zero_grad()
                    for scheduler in schedulers:
                        scheduler.step()

                    # Compute effective loss
                    effective_loss = np.sum(loss_during_accum).item()
                    loss_during_accum = []
                    loss_during_report += effective_loss
                    loss_history.append(effective_loss)

                    # Report
                    if len(loss_history) % conf['report_frequency'] == 0:
                        # Show avg loss during last report interval
                        avg_loss = loss_during_report / conf['report_frequency']
                        loss_during_report = 0.0
                        end_time = time.time()
                        logger.info('Step %d: avg loss %.2f; steps/sec %.2f' %
                                    (len(loss_history), avg_loss, conf['report_frequency'] / (end_time - start_time)))
                        start_time = end_time

                        tb_writer.add_scalar('Training_Loss', avg_loss, len(loss_history))
                        tb_writer.add_scalar('Learning_Rate_Bert', schedulers[0].get_last_lr()[0], len(loss_history))
                        tb_writer.add_scalar('Learning_Rate_Task', schedulers[1].get_last_lr()[-1], len(loss_history))

                    # Evaluate
                    if len(loss_history) > 0 and len(loss_history) % conf['eval_frequency'] == 0:
                        f1, info_metric = self.evaluate(model, examples_dev, stored_info, len(loss_history))
                        if f1 > max_f1:
                            max_f1 = f1
                            self.save_model_checkpoint(model, len(loss_history))
                        logger.info(info_metric + 'Eval max f1: %.2f' % max_f1)
                        start_time = time.time()

        logger.info('**********Finished training**********')
        logger.info('Actual update steps: %d' % len(loss_history))

        # Wrap up
        tb_writer.close()
        return loss_history

    def evaluate(self, model, tensor_examples, stored_info, step):
        logger.info('Step %d: evaluating on %d samples...' % (step, len(tensor_examples)))
        model.to(self.device)

        model.eval()
        num_gold_spans_list = []
        num_gold_top_spans_list = []
        num_top_spans_list = []
        # of the top ranked num_gold spans, how many is gold
        num_top_gold_list = []
        for i, (doc_key, tensor_example) in enumerate(tensor_examples):
            #tensor_example = tensor_example_full[:10]  # Strip out gold
            example_gpu = [d.to(self.device) for d in tensor_example[:10]]
            with torch.no_grad(): # output predictions
                top_span_mention_scores, top_span_starts, top_span_ends = model(*example_gpu)
            gold_starts = tensor_example[10].to(self.device)
            gold_ends = tensor_example[11].to(self.device)
            gold_mention_cluster_map = tensor_example[12].to(self.device)
            same_start = (torch.unsqueeze(gold_starts, 1) == torch.unsqueeze(top_span_starts, 0))  #[num_gold, num_top_spans]
            same_end = (torch.unsqueeze(gold_ends, 1) == torch.unsqueeze(top_span_ends, 0)) #[]
            same_span = (same_start & same_end).to(torch.long)
            top_span_labels = torch.matmul(torch.unsqueeze(gold_mention_cluster_map, 0).to(torch.float),
                                            same_span.to(torch.float))
            top_span_labels = torch.squeeze(top_span_labels.to(torch.long), 0)
            num_gold_top_spans = (top_span_labels > 0).sum()
            num_gold_top_spans_list.append(num_gold_top_spans.cpu().numpy())
            num_gold_spans_list.append(gold_starts.shape[0])
            num_top_spans_list.append(top_span_starts.shape[0])

            # top_gold_recall
            _, top_gold_indx = torch.topk(top_span_mention_scores, k=gold_starts.shape[0])
            top_gold_starts, top_gold_ends = top_span_starts[top_gold_indx], top_span_ends[top_gold_indx]
            same_start = (torch.unsqueeze(gold_starts, 1) == torch.unsqueeze(top_gold_starts,
                                                                             0))  # [num_gold, num_gold]
            same_end = (torch.unsqueeze(gold_ends, 1) == torch.unsqueeze(top_gold_ends, 0))  # []
            same_span = (same_start & same_end).to(torch.long)
            top_span_labels = torch.matmul(torch.unsqueeze(gold_mention_cluster_map, 0).to(torch.float),
                                           same_span.to(torch.float))
            top_span_labels = torch.squeeze(top_span_labels.to(torch.long), 0)
            num_top_gold = (top_span_labels > 0).sum()
            num_top_gold_list.append(num_top_gold.cpu().numpy())

        gold = sum(num_gold_spans_list)
        rec = sum(num_gold_top_spans_list)
        spans = sum(num_top_spans_list)
        top_gold = sum(num_top_gold_list)

        p = rec/spans * 100
        r = rec/gold * 100
        a = top_gold /gold * 100
        metrics = 'Precision: ' + str(p) + ', Recall: ' + str(r) + ', Accuracy: ' + str(a)
        return a, metrics


    def get_optimizer(self, model):
        no_decay = ['bias', 'LayerNorm.weight']
        bert_param, task_param = model.get_params(named=True)
        grouped_bert_param = [
            {
                'params': [p for n, p in bert_param if not any(nd in n for nd in no_decay)],
                'lr': self.config['bert_learning_rate'],
                'weight_decay': self.config['adam_weight_decay']
            }, {
                'params': [p for n, p in bert_param if any(nd in n for nd in no_decay)],
                'lr': self.config['bert_learning_rate'],
                'weight_decay': 0.0
            }
        ]
        optimizers = [
            AdamW(grouped_bert_param, lr=self.config['bert_learning_rate'], eps=self.config['adam_eps']),
            Adam(model.get_params()[1], lr=self.config['task_learning_rate'], eps=self.config['adam_eps'], weight_decay=0)
        ]
        return optimizers
        # grouped_parameters = [
        #     {
        #         'params': [p for n, p in bert_param if not any(nd in n for nd in no_decay)],
        #         'lr': self.config['bert_learning_rate'],
        #         'weight_decay': self.config['adam_weight_decay']
        #     }, {
        #         'params': [p for n, p in bert_param if any(nd in n for nd in no_decay)],
        #         'lr': self.config['bert_learning_rate'],
        #         'weight_decay': 0.0
        #     }, {
        #         'params': [p for n, p in task_param if not any(nd in n for nd in no_decay)],
        #         'lr': self.config['task_learning_rate'],
        #         'weight_decay': self.config['adam_weight_decay']
        #     }, {
        #         'params': [p for n, p in task_param if any(nd in n for nd in no_decay)],
        #         'lr': self.config['task_learning_rate'],
        #         'weight_decay': 0.0
        #     }
        # ]
        # optimizer = AdamW(grouped_parameters, lr=self.config['task_learning_rate'], eps=self.config['adam_eps'])
        # return optimizer

    def get_scheduler(self, optimizers, total_update_steps):
        # Only warm up bert lr
        warmup_steps = int(total_update_steps * self.config['warmup_ratio'])

        def lr_lambda_bert(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return max(
                0.0, float(total_update_steps - current_step) / float(max(1, total_update_steps - warmup_steps))
            )

        def lr_lambda_task(current_step):
            return max(0.0, float(total_update_steps - current_step) / float(max(1, total_update_steps)))

        schedulers = [
            LambdaLR(optimizers[0], lr_lambda_bert),
            LambdaLR(optimizers[1], lr_lambda_task)
        ]
        return schedulers
        # return LambdaLR(optimizer, [lr_lambda_bert, lr_lambda_bert, lr_lambda_task, lr_lambda_task])

    def save_model_checkpoint(self, model, step):
        if step < 2000:
            return  # Debug
        path_ckpt = join(self.config['log_dir'], f'model_{self.name_suffix}_{step}.bin')
        torch.save(model.state_dict(), path_ckpt)
        logger.info('Saved model to %s' % path_ckpt)

    def load_model_checkpoint(self, model, suffix):
        path_ckpt = join(self.config['log_dir'], f'model_{suffix}.bin')
        model.load_state_dict(torch.load(path_ckpt, map_location=torch.device('cpu')), strict=False)
        logger.info('Loaded model from %s' % path_ckpt)


if __name__ == '__main__':
    training = True
    if len(sys.argv) > 3:
        training = False
    config_name, gpu_id = sys.argv[1], int(sys.argv[2])
    trainer = Trainer(config_name, gpu_id)
    if training:
        model = trainer.initialize_model()
        trainer.train(model)
    else:
        cached_model_name = sys.argv[3]
        trainer.cache_mention_proposals(cached_model_name)

