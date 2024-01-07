import torch
import torch.nn as nn
from transformers import ElectraModel
from transformers import BertModel
import util
import logging
from collections import Iterable
import numpy as np
import torch.nn.init as init


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger()


class CorefModel(nn.Module):
    def __init__(self, config, device, wn_embed, ent_embed, config_name, num_genres=None):
        super().__init__()
        self.config = config
        self.device = device
        self.config_name = config_name

        self.num_genres = num_genres if num_genres else len(config['genres'])
        self.max_seg_len = config['max_segment_len']
        self.max_span_width = config['max_span_width']

        # Model
        self.dropout = nn.Dropout(p=config['dropout_rate'])
        if 'electra' in self.config_name:
            self.bert = ElectraModel.from_pretrained(config['bert_pretrained_name_or_path'])
        elif 'spanbert' in self.config_name:
            self.bert = BertModel.from_pretrained(config['bert_pretrained_name_or_path'])


        self.bert_emb_size = self.bert.config.hidden_size
        self.span_emb_size = self.bert_emb_size * 3

        self.pair_emb_size = self.span_emb_size * 3
        if config['use_metadata']:
            self.pair_emb_size += 2 * config['feature_emb_size']
        if config['use_features']:
            self.pair_emb_size += config['feature_emb_size']
        if config['use_segment_distance']:
            self.pair_emb_size += config['feature_emb_size']

        # WOrdNet embeddings
        if wn_embed is not None:
            num, dims = wn_embed.shape
            self.wn_embeddings = nn.Embedding(num, dims)
            self.wn_embeddings.weight = nn.Parameter(torch.Tensor(wn_embed))
            self.wn_embeddings.weight.requires_grad = False
        # Entity Embeddings
        if ent_embed is not None:
            num, dims = ent_embed.shape
            self.ent_embeddings = nn.Embedding(num, dims)
            self.ent_embeddings.weight = nn.Parameter(torch.Tensor(ent_embed))
            self.ent_embeddings.weight.requires_grad = False
            if config['use_ent_info']:
                self.pair_emb_size += dims

        self.emb_antecedent_distance_prior = self.make_embedding(10) if config['use_distance_prior'] else None
        self.emb_genre = self.make_embedding(self.num_genres)
        self.emb_same_speaker = self.make_embedding(2) if config['use_metadata'] else None
        self.emb_segment_distance = self.make_embedding(config['max_training_sentences']) if config['use_segment_distance'] else None
        self.emb_top_antecedent_distance = self.make_embedding(10)
        self.emb_cluster_size = self.make_embedding(10) if config['higher_order'] == 'cluster_merging' else None

        self.mention_token_attn = self.make_ffnn(self.bert_emb_size, 0, output_size=1) if config['model_heads'] else None
        self.span_emb_score_ffnn = self.make_ffnn(self.span_emb_size, [config['ffnn_size']] * config['ffnn_depth'], output_size=1)
        self.span_width_score_ffnn = self.make_ffnn(config['feature_emb_size'], [config['ffnn_size']] * config['ffnn_depth'], output_size=1) if config['use_width_prior'] else None
        self.coarse_bilinear = self.make_ffnn(self.span_emb_size, 0, output_size=self.span_emb_size)
        self.antecedent_distance_score_ffnn = self.make_ffnn(config['feature_emb_size'], 0, output_size=1) if config['use_distance_prior'] else None
        self.coref_score_ffnn = self.make_ffnn(self.pair_emb_size, [config['ffnn_size']] * config['ffnn_depth'], output_size=1) if config['fine_grained'] else None

        self.gate_ffnn = self.make_ffnn(2 * self.span_emb_size, 0, output_size=self.span_emb_size) if config['coref_depth'] > 1 else None
        self.span_attn_ffnn = self.make_ffnn(self.span_emb_size, 0, output_size=1) if config['higher_order'] == 'span_clustering' else None
        self.cluster_score_ffnn = self.make_ffnn(3 * self.span_emb_size + config['feature_emb_size'], [config['cluster_ffnn_size']] * config['ffnn_depth'], output_size=1) if config['higher_order'] == 'cluster_merging' else None

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
            #print(name, param.requires_grad)
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
                                 is_training, top_span_mention_scores, top_span_starts, top_span_ends, gold_starts=None, gold_ends=None, gold_mention_cluster_map=None):
        """ Model and input are already on the device """
        device = self.device
        conf = self.config

        do_loss = False
        if gold_mention_cluster_map is not None:
            assert gold_starts is not None
            assert gold_ends is not None
            do_loss = True

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
        mention_doc = mention_doc[input_mask]
        mention_doc_one = mention_doc_one[input_mask]
        speaker_ids = speaker_ids[input_mask]
        num_words = mention_doc.shape[0]

        num_top_spans = top_span_starts.shape[0]
        if do_loss:
            same_start = (torch.unsqueeze(gold_starts, 1) == torch.unsqueeze(top_span_starts, 0))
            same_end = (torch.unsqueeze(gold_ends, 1) == torch.unsqueeze(top_span_ends, 0))
            same_span = (same_start & same_end).to(torch.long)
            top_span_cluster_ids = torch.matmul(torch.unsqueeze(gold_mention_cluster_map, 0).to(torch.float),
                                       same_span.to(torch.float))
            top_span_cluster_ids = torch.squeeze(top_span_cluster_ids.to(torch.long), 0)

        # get top span embeddings for linking
        top_span_start_emb, top_span_end_emb = mention_doc[top_span_starts], mention_doc[top_span_ends]
        hybrid_mention_doc = mention_doc * (1 - conf['alpha']) + mention_doc_one * conf['alpha']
        top_span_emb_list = [top_span_start_emb, top_span_end_emb]
        # Use attended head or avg token
        top_span_tokens = torch.unsqueeze(torch.arange(0, num_words, device=device), 0).repeat(num_top_spans,
                                                                                                   1)  # [num_top_spans, num_words]
        top_span_tokens_mask = (top_span_tokens >= torch.unsqueeze(top_span_starts, 1)) & (
                    top_span_tokens <= torch.unsqueeze(top_span_ends, 1))
        if conf['model_heads']:
            top_span_token_attn = torch.squeeze(self.mention_token_attn(hybrid_mention_doc), 1)  # [num_words]
        else:
            top_span_token_attn = torch.ones(num_words, dtype=torch.float, device=device)  # Use avg if no attention
        top_span_tokens_attn_raw = torch.log(top_span_tokens_mask.to(torch.float)) + torch.unsqueeze(
            top_span_token_attn, 0)
        top_span_tokens_attn = nn.functional.softmax(top_span_tokens_attn_raw, dim=1)
        head_attn_emb = torch.matmul(top_span_tokens_attn, hybrid_mention_doc)
        top_span_emb_list.append(head_attn_emb)
        top_span_emb = torch.cat(top_span_emb_list, dim=1)  # [num candidates, new emb size]

        # Coarse pruning on each mention's antecedents
        max_top_antecedents = min(num_top_spans, conf['max_top_antecedents'])
        top_span_range = torch.arange(0, num_top_spans, device=device)
        antecedent_offsets = torch.unsqueeze(top_span_range, 1) - torch.unsqueeze(top_span_range, 0) #[num_top_spans, num_top_spans]
        antecedent_mask = (antecedent_offsets >= 1)
        pairwise_mention_score_sum = torch.unsqueeze(top_span_mention_scores, 1) + torch.unsqueeze(top_span_mention_scores, 0)
        source_span_emb = self.dropout(self.coarse_bilinear(top_span_emb))
        target_span_emb = self.dropout(torch.transpose(top_span_emb, 0, 1))
        pairwise_coref_scores = torch.matmul(source_span_emb, target_span_emb)

        pairwise_fast_scores = pairwise_mention_score_sum + pairwise_coref_scores
        pairwise_fast_scores += torch.log(antecedent_mask.to(torch.float))
        if conf['use_distance_prior']:
            distance_score = torch.squeeze(self.antecedent_distance_score_ffnn(self.dropout(self.emb_antecedent_distance_prior.weight)), 1)
            bucketed_distance = util.bucket_distance(antecedent_offsets)
            antecedent_distance_score = distance_score[bucketed_distance]
            pairwise_fast_scores += antecedent_distance_score
        # antecedent pruning
        top_pairwise_fast_scores, top_antecedent_idx = torch.topk(pairwise_fast_scores, k=max_top_antecedents)
        top_antecedent_mask = util.batch_select(antecedent_mask, top_antecedent_idx, device)  # [num top spans, max top antecedents]
        top_antecedent_offsets = util.batch_select(antecedent_offsets, top_antecedent_idx, device)

        # Slow antecedents ranking
        if conf['fine_grained']:
            same_speaker_emb, genre_emb, seg_distance_emb, top_antecedent_distance_emb = None, None, None, None
            if conf['use_metadata']:
                top_span_speaker_ids = speaker_ids[top_span_starts]
                top_antecedent_speaker_id = top_span_speaker_ids[top_antecedent_idx]
                same_speaker = torch.unsqueeze(top_span_speaker_ids, 1) == top_antecedent_speaker_id
                same_speaker_emb = self.emb_same_speaker(same_speaker.to(torch.long))
                genre_emb = self.emb_genre(genre)
                genre_emb = torch.unsqueeze(torch.unsqueeze(genre_emb, 0), 0).repeat(num_top_spans, max_top_antecedents, 1)
            if conf['use_segment_distance']:
                num_segs, seg_len = input_ids.shape[0], input_ids.shape[1]
                token_seg_ids = torch.arange(0, num_segs, device=device).unsqueeze(1).repeat(1, seg_len)
                token_seg_ids = token_seg_ids[input_mask]
                top_span_seg_ids = token_seg_ids[top_span_starts]
                top_antecedent_seg_ids = token_seg_ids[top_span_starts[top_antecedent_idx]]
                top_antecedent_seg_distance = torch.unsqueeze(top_span_seg_ids, 1) - top_antecedent_seg_ids
                top_antecedent_seg_distance = torch.clamp(top_antecedent_seg_distance, 0, self.config['max_training_sentences'] - 1)
                seg_distance_emb = self.emb_segment_distance(top_antecedent_seg_distance)
            if conf['use_features']:  # Antecedent distance
                top_antecedent_distance = util.bucket_distance(top_antecedent_offsets)
                top_antecedent_distance_emb = self.emb_top_antecedent_distance(top_antecedent_distance)

            if conf['use_ent_info']:
                # use WordNet relations,
                # Word2Vec embeddings of words [num_nontokenized_words, emb_dim], word_ids
                wn_embed_doc = self.wn_embeddings(word_ids)  # [num_nontokenized_words, wn_emb_dim]
                num_nontokenized_words = word_ids.shape[0]
                top_span_orig_tokens = torch.squeeze(torch.arange(0, num_nontokenized_words, device=device), 0).repeat(
                    num_top_spans, 1)
                orig_top_span_starts = subtoken_map[top_span_starts]
                orig_top_span_ends = subtoken_map[top_span_ends]
                top_span_orig_tokens_mask = (top_span_orig_tokens >= torch.unsqueeze(orig_top_span_starts, 1)) & (
                        top_span_orig_tokens <= torch.unsqueeze(orig_top_span_ends,
                                                                1))  # [num_top_spans, num_nontokenzied_words]
                top_span_orig_token_attn = torch.ones(num_nontokenized_words, dtype=torch.float,
                                                      device=device)  # Use avg if no attention
                top_span_orig_tokens_attn_raw = torch.log(top_span_orig_tokens_mask.to(torch.float)) + torch.unsqueeze(
                    top_span_orig_token_attn, 0)
                top_span_orig_tokens_attn = nn.functional.softmax(top_span_orig_tokens_attn_raw,
                                                                  dim=1)  # [num_top_span, num_nontokenized_words]
                wn_top_span_embs = torch.matmul(top_span_orig_tokens_attn, wn_embed_doc)  # [num_top_span, wn_emb_dim]

                # use entity information from Wikipedia
                ent_id_masks = ent_ids > -1  # [num_nontokenized_words]
                ent_id_masks = ent_id_masks.to(torch.float).repeat(num_top_spans,
                                                                   1)  # [num_top_spans, num_nontokenized_words]
                ent_and_top_span = top_span_orig_tokens_mask.to(torch.float) * ent_id_masks
                detected_nem_mask = torch.sum(ent_and_top_span, dim=0) > 0  # [num_tokenized_words]
                detected_nem_mask = detected_nem_mask.to(torch.int)
                ent_embed_doc = self.ent_embeddings(ent_ids * detected_nem_mask)  # [num_tokenized_words, ent_emb_dim]
                ent_top_span_embs = torch.matmul(top_span_orig_tokens_attn, ent_embed_doc)  # [num_top_span, wn_emb_dim]
                nem_top_span_mask = torch.sum(ent_and_top_span, dim=1) > 0  # [num_top_span], whether the span is NEM
                wn_top_span_mask = torch.logical_not(nem_top_span_mask)
                nem_top_span_mask = nem_top_span_mask.to(torch.int)
                ent_top_span_embs = ent_top_span_embs * nem_top_span_mask[:, None]
                wn_top_span_mask = wn_top_span_mask.to(torch.int)
                wn_top_span_embs = wn_top_span_embs * wn_top_span_mask[:, None]
                wnent_top_span_embs = ent_top_span_embs + wn_top_span_embs
                # top_antecedent_idx [num_top_spans, max_top_antecedents]
                top_antecedent_wnent = wnent_top_span_embs[top_antecedent_idx] #[]
                target_wnent = torch.unsqueeze(wnent_top_span_embs, 1).repeat(1, max_top_antecedents, 1)
                simila_wnent = target_wnent * top_antecedent_wnent

            top_antecedent_emb = top_span_emb[top_antecedent_idx]  # [num top spans, max top antecedents, emb size]
            feature_list = []
            if conf['use_metadata']:  # speaker, genre
                feature_list.append(same_speaker_emb)
                feature_list.append(genre_emb)
            if conf['use_segment_distance']:
                feature_list.append(seg_distance_emb)
            if conf['use_features']:  # Antecedent distance
                feature_list.append(top_antecedent_distance_emb)
            if conf['use_ent_info']:
                feature_list.append(simila_wnent)
            feature_emb = torch.cat(feature_list, dim=2)
            feature_emb = self.dropout(feature_emb)
            target_emb = torch.unsqueeze(top_span_emb, 1).repeat(1, max_top_antecedents, 1) #[num_top_spans, mant_top_antecedents, emb_dim]
            similarity_emb = target_emb * top_antecedent_emb
            pair_emb = torch.cat([target_emb, top_antecedent_emb, similarity_emb, feature_emb], 2)
            top_pairwise_slow_scores = torch.squeeze(self.coref_score_ffnn(pair_emb), 2)
            top_pairwise_scores = top_pairwise_slow_scores + top_pairwise_fast_scores
        else:
            top_pairwise_scores = top_pairwise_fast_scores  # [num top spans, max top antecedents]

        if not do_loss:
            top_antecedent_scores = torch.cat([torch.zeros(num_top_spans, 1, device=device), top_pairwise_scores], dim=1)  # [num top spans, max top antecedents + 1]
            return top_span_starts, top_span_ends, top_antecedent_idx, top_antecedent_scores

        # Get gold labels
        top_antecedent_cluster_ids = top_span_cluster_ids[top_antecedent_idx]
        top_antecedent_cluster_ids += (top_antecedent_mask.to(torch.long) - 1) * 100000  # Mask id on invalid antecedents
        same_gold_cluster_indicator = (top_antecedent_cluster_ids == torch.unsqueeze(top_span_cluster_ids, 1))
        non_dummy_indicator = torch.unsqueeze(top_span_cluster_ids > 0, 1)
        pairwise_labels = same_gold_cluster_indicator & non_dummy_indicator
        dummy_antecedent_labels = torch.logical_not(pairwise_labels.any(dim=1, keepdims=True))
        top_antecedent_gold_labels = torch.cat([dummy_antecedent_labels, pairwise_labels], dim=1)

        # Get loss
        top_antecedent_scores = torch.cat([torch.zeros(num_top_spans, 1, device=device), top_pairwise_scores], dim=1)
        log_marginalized_antecedent_scores = torch.logsumexp(top_antecedent_scores + torch.log(top_antecedent_gold_labels.to(torch.float)), dim=1)
        log_norm = torch.logsumexp(top_antecedent_scores, dim=1)
        loss = torch.sum(log_norm - log_marginalized_antecedent_scores)

        # Debug
        if self.debug:
            if self.update_steps % 20 == 0:
                logger.info('---------debug step: %d---------' % self.update_steps)
                # logger.info('candidates: %d; antecedents: %d' % (num_candidates, max_top_antecedents))
                logger.info('spans/gold: %d/%d; ratio: %.2f' % (num_top_spans, (top_span_cluster_ids > 0).sum(), (top_span_cluster_ids > 0).sum()/num_top_spans))
                if conf['mention_loss_coef']:
                    logger.info('mention loss: %.4f' % loss_mention)
                if conf['loss_type'] == 'marginalized':
                    logger.info('norm/gold: %.4f/%.4f' % (torch.sum(log_norm), torch.sum(log_marginalized_antecedent_scores)))
                else:
                    logger.info('loss: %.4f' % loss)
        self.update_steps += 1

        return [top_span_starts, top_span_ends, top_antecedent_idx, top_antecedent_scores], loss

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

    def get_predicted_antecedents(self, antecedent_idx, antecedent_scores):
        """ CPU list input """
        predicted_antecedents = []
        for i, idx in enumerate(np.argmax(antecedent_scores, axis=1) - 1):
            if idx < 0:
                predicted_antecedents.append(-1)
            else:
                predicted_antecedents.append(antecedent_idx[i][idx])
        return predicted_antecedents

    def get_predicted_clusters(self, span_starts, span_ends, antecedent_idx, antecedent_scores):
        """ CPU list input """
        # Get predicted antecedents
        predicted_antecedents = self.get_predicted_antecedents(antecedent_idx, antecedent_scores)

        # Get predicted clusters
        mention_to_cluster_id = {}
        predicted_clusters = []
        for i, predicted_idx in enumerate(predicted_antecedents):
            if predicted_idx < 0:
                continue
            assert i > predicted_idx, f'span idx: {i}; antecedent idx: {predicted_idx}'
            # Check antecedent's cluster
            antecedent = (int(span_starts[predicted_idx]), int(span_ends[predicted_idx]))
            antecedent_cluster_id = mention_to_cluster_id.get(antecedent, -1)
            if antecedent_cluster_id == -1:
                antecedent_cluster_id = len(predicted_clusters)
                predicted_clusters.append([antecedent])
                mention_to_cluster_id[antecedent] = antecedent_cluster_id
            # Add mention to cluster
            mention = (int(span_starts[i]), int(span_ends[i]))
            predicted_clusters[antecedent_cluster_id].append(mention)
            mention_to_cluster_id[mention] = antecedent_cluster_id

        predicted_clusters = [tuple(c) for c in predicted_clusters]
        return predicted_clusters, mention_to_cluster_id, predicted_antecedents

    def update_evaluator(self, span_starts, span_ends, antecedent_idx, antecedent_scores, gold_clusters, evaluator):
        predicted_clusters, mention_to_cluster_id, _ = self.get_predicted_clusters(span_starts, span_ends, antecedent_idx, antecedent_scores)
        mention_to_predicted = {m: predicted_clusters[cluster_idx] for m, cluster_idx in mention_to_cluster_id.items()}
        gold_clusters = [tuple(tuple(m) for m in cluster) for cluster in gold_clusters]
        mention_to_gold = {m: cluster for cluster in gold_clusters for m in cluster}
        evaluator.update(predicted_clusters, gold_clusters, mention_to_predicted, mention_to_gold)
        return predicted_clusters
