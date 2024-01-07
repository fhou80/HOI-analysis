from tensorize import Tensorizer
import util
import os
import random
from os.path import join
import json
import torch
import pickle
import math
import sys
import numpy as np
import logging
from transformers import BertTokenizer

logger = logging.getLogger(__name__)

class DataAugmentation:
    def __init__(self, config, max_seg_length, truncate=True):
        self.language = 'english'
        self.max_seg_len = max_seg_length
        self.max_training_seg = config['max_training_sentences']
        self.data_dir = config['data_dir']
        if config['use_ent_info'] and os.path.exists(self.data_dir+'/w2vec_vocab.txt'):
            self.wn_vocab, self.wn_embed = util.load_voca_embs(self.data_dir+'/w2vec_vocab.txt', self.data_dir + '/w2vec_embeddings.npy')
        else:
            self.wn_vocab, self.wn_embed = None, None
        # entity vocab and embeddings
        if config['use_ent_info'] and os.path.exists(self.data_dir+'/entity_dict_w2.txt'):
            self.ent_vocab, self.ent_embed = util.load_voca_embs(self.data_dir+'/entity_dict_w2.txt', self.data_dir + '/entity_vec_w2.npy', insert_unk=False)
        else:
            self.ent_vocab, self.ent_embed = None, None
        self.tensorizer = Tensorizer(config, self.wn_vocab, self.ent_vocab)
        self.tensor_samples = {}
        self.truncate = truncate
        # for mask
        self.tokenizer = BertTokenizer.from_pretrained(config['bert_tokenizer_name'])
        self.mask_id = self.tokenizer.convert_tokens_to_ids('[MASK]')
        self.ment_swap_num = 0
        self.ment_mask_num = 0

    def convert_to_torch_tensor(self, input_ids, input_mask, speaker_ids, word_ids, ent_ids, subtoken_map, sentence_len, genre, sentence_map,
                                is_training, gold_starts, gold_ends, gold_mention_cluster_map):
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        input_mask = torch.tensor(input_mask, dtype=torch.long)
        speaker_ids = torch.tensor(speaker_ids, dtype=torch.long)
        sentence_len = torch.tensor(sentence_len, dtype=torch.long)
        genre = torch.tensor(genre, dtype=torch.long)
        sentence_map = torch.tensor(sentence_map, dtype=torch.long)
        is_training = torch.tensor(is_training, dtype=torch.bool)
        gold_starts = torch.tensor(gold_starts, dtype=torch.long)
        gold_ends = torch.tensor(gold_ends, dtype=torch.long)
        gold_mention_cluster_map = torch.tensor(gold_mention_cluster_map, dtype=torch.long)

        word_ids = torch.tensor(word_ids, dtype=torch.long)
        ent_ids = torch.tensor(ent_ids, dtype=torch.long)
        subtoken_map = torch.tensor(subtoken_map, dtype=torch.long)

        return input_ids, input_mask, speaker_ids, word_ids, ent_ids, subtoken_map, sentence_len, genre, sentence_map, \
               is_training, gold_starts, gold_ends, gold_mention_cluster_map

    def get_cache_path(self):
        if self.truncate:
            self.max_training_seg = 2
        cache_path = join(self.data_dir, f'augmented.tensors.{self.language}.{self.max_seg_len}.{self.max_training_seg}.bin')
        return cache_path

    def aug_and_save(self):
        path_512 = join(self.data_dir, f'train.{self.language}.512.jsonlines')
        path_384 = join(self.data_dir, f'train.{self.language}.384.jsonlines')
        with open(path_512, 'r') as f:
            samples_512 = [json.loads(line) for line in f.readlines()]
        with open(path_384, 'r') as f:
            samples_384 = [json.loads(line) for line in f.readlines()]
        is_training = True
        tensor_samples_512 = [self.tensorizer.tensorize_example(sample, is_training) for sample in samples_512]
        tensor_samples_384 = [self.tensorizer.tensorize_example(sample, is_training) for sample in samples_384]
        augmented_tensor_samples = self.augment_tensorized_examples(tensor_samples_512, tensor_samples_384, truncate=self.truncate)
        self.tensor_samples['trn'] = [(doc_key, self.convert_to_torch_tensor(*tensor)) for doc_key, tensor in augmented_tensor_samples]
        paths = {
            'dev': join(self.data_dir, f'dev.{self.language}.{self.max_seg_len}.jsonlines'),
            'tst': join(self.data_dir, f'test.{self.language}.{self.max_seg_len}.jsonlines')
        }
        print('mention swap number: {}'.format(self.ment_swap_num))
        print('tokens being masked: {}'.format(self.ment_mask_num))
        for split, path in paths.items():
            logger.info('Tensorizing examples from %s; results will be cached)' % path)
            is_training = False
            with open(path, 'r') as f:
                samples = [json.loads(line) for line in f.readlines()]
            tensor_samples = [self.tensorizer.tensorize_example(sample, is_training) for sample in samples]
            self.tensor_samples[split] = [(doc_key, self.convert_to_torch_tensor(*tensor)) for doc_key, tensor in
                                          tensor_samples]
        self.stored_info = self.tensorizer.stored_info
        # Cache tensorized samples
        cache_path = self.get_cache_path()
        with open(cache_path, 'wb') as f:
            pickle.dump((self.tensor_samples, self.stored_info), f)


    def augment_tensorized_examples(self, examples_512, examples_384, truncate=False, round=1):
        # extract from non-truncated examples
        # {'bc': 283, 'bn': 763, 'mz': 409, 'nw': 745, 'pt': 319, 'tc': 110, 'wb': 173}
        #{'bc': {1: 141, 3: 18, 2: 124}, 'bn': {1: 670, 3: 7, 2: 86}, 'mz': {1: 283, 2: 126==}, 'nw': {3: 136, 2: 181, 1: 428},
        # 'pt': {3: 31, 2: 215, 1: 73}, 'tc': {3: 17, 2: 87, 1: 6}, 'wb': {1: 32, 3: 84, 2: 57}}
        # hybrid generate from documents of the same genre
        # should generate 583 new documents
        # sentence_map is to guarantee that candidate spans do not come across sentences.
        augmented_samples = []
        if truncate:
            for key, example in examples_512:
                if len(example[6]) == 3:
                    augmented_samples.append((key, self.truncate_example(*example)))
                else:
                    augmented_samples.append((key, example))
        else:
            augmented_samples = examples_512
        genre_sentnum_dict = {} # a list of documents of the same genre
        doc_dict_512 = {}
        doc_dict_384 = {}
        i = 0
        for key, np_arrays in examples_512:
            i += 1
            genre = key[:2]
            sent_num = len(np_arrays[6])
            if (genre, sent_num) in genre_sentnum_dict.keys():
                genre_sentnum_dict[genre, sent_num] += [key]
            else:
                genre_sentnum_dict[genre, sent_num] = [key]
            doc_dict_512[key] = np_arrays
        for doc_key, np_arrays in examples_384:
            doc_dict_384[doc_key] = np_arrays
        print('original documents: {}----{}'.format(len(augmented_samples), i))
        print(genre_sentnum_dict.keys())
        # how many iterates to randomly combine
        for r in range(round):
            for genre in ['bc', 'bn', 'mz', 'nw', 'pt', 'tc', 'wb']:
                if (genre, 3) not in genre_sentnum_dict.keys():
                    genre_sentnum_dict[genre, 3] = []
                doc_one_segs = len(genre_sentnum_dict[genre, 1])
                doc_two_segs = len(genre_sentnum_dict[genre, 2])
                doc_tri_segs = len(genre_sentnum_dict[genre, 3])
                print(doc_one_segs, doc_two_segs, doc_tri_segs)
                doc_longer_one_segs = doc_two_segs + doc_tri_segs

                new_doc_two_segs_keys = random.sample(genre_sentnum_dict[genre, 2] + genre_sentnum_dict[genre, 3],
                                                      doc_longer_one_segs)
                doc_one_segs_keys = random.sample(genre_sentnum_dict[genre, 1], doc_one_segs)
                # corss-combine the docs
                if doc_longer_one_segs > doc_one_segs:
                    # combine docs_one_segs with docs_two_segs
                    for i in range(doc_one_segs):
                        aug_key = new_doc_two_segs_keys[i] + ';' + doc_one_segs_keys[i]
                        one_seg_tensors = doc_dict_512[doc_one_segs_keys[i]]
                        two_segs_tensors = doc_dict_512[new_doc_two_segs_keys[i]]
                        if len(two_segs_tensors[6]) == 2 and two_segs_tensors[6][1] < 256:
                            two_segs_tensors = doc_dict_384[new_doc_two_segs_keys[i]]
                        elif len(two_segs_tensors[6]) == 3 and two_segs_tensors[6][2] < 128:
                            two_segs_tensors = doc_dict_384[new_doc_two_segs_keys[i]]
                        augmented_samples.append((aug_key, self.combine_two_docs(*two_segs_tensors, *one_seg_tensors)))
                    print('based on one_seg_docs {}'.format(doc_one_segs))
                    # cross-combine the left doc_two_segs
                    if doc_longer_one_segs - doc_one_segs < 2:
                        continue
                    half_left = math.floor((doc_longer_one_segs - doc_one_segs) / 2)
                    for i in range(half_left):
                        aug_key = new_doc_two_segs_keys[doc_one_segs + i] + ':' + new_doc_two_segs_keys[
                            doc_one_segs + i + 1]
                        one_seg_tensors = doc_dict_512[new_doc_two_segs_keys[i + doc_one_segs]]
                        two_segs_tensors = doc_dict_512[new_doc_two_segs_keys[i + doc_one_segs + 1]]
                        if len(two_segs_tensors[6]) == 2 and two_segs_tensors[6][1] < 256:
                            two_segs_tensors = doc_dict_384[new_doc_two_segs_keys[i + doc_one_segs + 1]]
                        elif len(two_segs_tensors[6]) == 3 and two_segs_tensors[6][2] < 128:
                            two_segs_tensors = doc_dict_384[new_doc_two_segs_keys[i + doc_one_segs + 1]]
                        augmented_samples.append(
                            (aug_key, self.combine_two_docs(*(two_segs_tensors + one_seg_tensors))))
                    print('left two_segs_docs combined {} docs'.format(half_left))
                elif doc_longer_one_segs <= doc_one_segs:
                    for i in range(doc_longer_one_segs):
                        aug_key = new_doc_two_segs_keys[i] + ';' + doc_one_segs_keys[i]
                        one_seg_tensors = doc_dict_512[doc_one_segs_keys[i]]
                        two_segs_tensors = doc_dict_512[new_doc_two_segs_keys[i]]
                        if len(two_segs_tensors[6]) == 2 and two_segs_tensors[6][1] < 256:
                            two_segs_tensors = doc_dict_384[new_doc_two_segs_keys[i]]
                        elif len(two_segs_tensors[6]) == 3 and two_segs_tensors[6][2] < 128:
                            two_segs_tensors = doc_dict_384[new_doc_two_segs_keys[i]]
                        augmented_samples.append(
                            (aug_key, self.combine_two_docs(*(two_segs_tensors + one_seg_tensors))))
                    print('based on two_segs_docsdoc_two_segs: {}'.format(doc_two_segs))
                    """
                    if doc_one_segs - doc_longer_one_segs < 2:
                        continue
                    half_left = math.floor((doc_one_segs - doc_longer_one_segs) / 2)
                    for i in range(half_left):
                        aug_key = doc_one_segs_keys[doc_longer_one_segs + i] + ':' + doc_one_segs_keys[
                            doc_longer_one_segs + i + 1]
                        one_seg_tensors = doc_dict_512[doc_one_segs_keys[i + doc_longer_one_segs]]
                        two_segs_tensors = doc_dict_512[doc_one_segs_keys[i + doc_longer_one_segs + 1]]
                        augmented_samples.append(
                            (aug_key, self.combine_two_docs(*(two_segs_tensors + one_seg_tensors))))
                    """
        # two rounds can generates more examples
        print('all training documents: {}'.format(len(augmented_samples)))
        return augmented_samples

    def combine_two_docs(self, *input_two_docs):
        two_segs_doc = input_two_docs[:13]
        num_segs = len(two_segs_doc[6])
        two_segs_doc_tensors = list(two_segs_doc)
        two_segs_doc_tensors.append(num_segs-1)
        one_seg_doc = input_two_docs[13:]
        one_seg_doc_tensors = list(one_seg_doc)
        input_ids_1, input_mask_1, speaker_ids_1, word_ids_1, ent_ids_1, subtoken_map_1, sentence_len_1, genre_1, sentence_map_1, \
          is_training_1, gold_starts_1, gold_ends_1, gold_mention_cluster_map_1 = self.slide_last_seg_to_start(*two_segs_doc_tensors)
        non_piece_offset = subtoken_map_1[-1] + 1
        one_seg_doc_tensors.append(non_piece_offset)
        word_offset = sentence_len_1[0]
        one_seg_doc_tensors.append(word_offset)
        sentence_offset = sentence_map_1[-1] + 1
        one_seg_doc_tensors.append(sentence_offset)
        cluster_offset = self.find_num_of_cluster(gold_mention_cluster_map_1)
        one_seg_doc_tensors.append(cluster_offset)
        input_ids_2, input_mask_2, speaker_ids_2, word_ids_2, ent_ids_2, subtoken_map_2, sentence_len_2, genre_2, sentence_map_2, \
        is_training_2, gold_starts_2, gold_ends_2, gold_mention_cluster_map_2 = self.slide_first_seg_to_end(*one_seg_doc_tensors)

        return (np.concatenate((input_ids_1, input_ids_2)), np.concatenate((input_mask_1, input_mask_2)), np.concatenate((speaker_ids_1, speaker_ids_2)), \
               np.append(word_ids_1, word_ids_2), np.append(ent_ids_1, ent_ids_2), np.append(subtoken_map_1, subtoken_map_2), \
               np.append(sentence_len_1, sentence_len_2), genre_1, np.append(sentence_map_1, sentence_map_2), is_training_1, \
               np.append(gold_starts_1, gold_starts_2), np.append(gold_ends_1, gold_ends_2), np.append(gold_mention_cluster_map_1, gold_mention_cluster_map_2))

    def slide_last_seg_to_start(self, input_ids, input_mask, speaker_ids, word_ids, ent_ids, subtoken_map, sentence_len, genre, sentence_map, is_training,
                         gold_starts, gold_ends, gold_mention_cluster_map, sentence_offset=1):

        sent_offset = sentence_offset
        word_offset = sentence_len[:sent_offset].sum()
        num_words = sentence_len[sent_offset:].sum()

        # truncate subtoken map
        word_ids = np.array(word_ids[subtoken_map[word_offset]:])
        ent_ids = np.array(ent_ids[subtoken_map[word_offset]:])
        subtoken_map = np.array(subtoken_map)
        subtoken_map = subtoken_map[word_offset:] - subtoken_map[word_offset]

        input_ids = np.array(input_ids[sent_offset:, :])
        input_mask = np.array(input_mask[sent_offset:, :])
        speaker_ids = np.array(speaker_ids[sent_offset:, :])
        sentence_len = np.array(sentence_len[sent_offset:])

        sentence_map = np.array(sentence_map)
        sentence_map = sentence_map[word_offset:] - sentence_map[word_offset]
        gold_starts, gold_ends = np.array(gold_starts), np.array(gold_ends)
        gold_mention_cluster_map = np.array(gold_mention_cluster_map)
        gold_spans = (gold_starts < word_offset + num_words) & (gold_ends >= word_offset)
        # gold_starts = [ 24,  29,  33,  42,  52,  57,  72,  75,  88, 110, 124, 124, 131, 131]
        # gold_ends =   [ 25,  29,  38,  43,  58,  57,  72,  75,  88, 111, 125, 126, 132, 133]
        # cluster_map = [ 1,   2,   2,   1,   4,   3,   3,   3,   4,  1,   1,   4,   1,   4, ]
        gold_starts = gold_starts[gold_spans] - word_offset
        gold_ends = gold_ends[gold_spans] - word_offset
        gold_mention_cluster_map = gold_mention_cluster_map[gold_spans]

        # mention swap
        if sent_offset == 0 and len(sentence_len) == 1:
            # randomly swap mentions in shorter docs with only 1 segment
            # build cluster dict, map cluster id to gold mention id
            cluster_dict = {}
            for i, cluster_id in enumerate(gold_mention_cluster_map.tolist()):
                if cluster_id in cluster_dict.keys():
                    cluster_dict[cluster_id] += [i]
                else:
                    cluster_dict[cluster_id] = [i]
            # remove clusters with only one mention
            for k, v in cluster_dict.items():
                if len(v) == 1:
                    cluster_dict.pop(k)
            cluster_keys = cluster_dict.keys()
            # if have clusters of > 1 mentions
            if len(cluster_keys) > 0:
                num_cluster = len(cluster_keys)
                selected_cluster_ids = random.sample(cluster_keys, num_cluster)
                for i in range(num_cluster):
                    mention_ids = cluster_dict[selected_cluster_ids[i]]
                    for j in mention_ids:
                        for k in mention_ids:
                            if j != k and input_ids[0][gold_starts[j]] != input_ids[0][gold_starts[k]]:
                                m1, m2 = min(j, k), max(j, k)
                                s1, s2 = gold_starts[m1], gold_starts[m2]
                                e1, e2 = gold_ends[m1], gold_ends[m2]
                                # make sure m1, m2 are mention in mention
                                if s2 in range(s1, e1+1) or e1 in range(s2, e2+1):
                                    continue
                                # find a pair of mention to swap
                                self.ment_swap_num += 1
                                # update sentence_map, input_ids, speaker_ids,
                                input_ids[0] = np.array(self.swap_ids(input_ids[0].tolist(), s1, s2, e1, e2))
                                speaker_ids = np.array(self.swap_ids(speaker_ids.tolist(), s1, s2, e1, e2))
                                sentence_map = np.array(self.update_map(sentence_map.tolist(), s1, s2, e1, e2))
                                # swap gold_starts, gold_ends
                                gold_ends[m1] = s1 + (e2 - s2)
                                gold_starts[m2] = e2 - ( e1- s1)
                                return (input_ids, input_mask, speaker_ids, word_ids, ent_ids, subtoken_map, sentence_len, \
                                        genre, sentence_map, is_training, gold_starts, gold_ends, gold_mention_cluster_map)
        """
        else:
            # do mention masking
            ment_num = len(gold_starts)
            pos_list = np.arange(ment_num, dtype=int)
            sample_num = math.floor(ment_num * 0.1)
            self.ment_mask_num += 1
            sample_ids = random.sample(pos_list.tolist(), sample_num)
            for id in sample_ids:
                for i range(gold_starts[id], gold_ends[id]+1):
                    input_ids[o][i] = self.mask_id
        """
        return (input_ids, input_mask, speaker_ids, word_ids, ent_ids, subtoken_map, sentence_len, genre, sentence_map, \
               is_training, gold_starts, gold_ends, gold_mention_cluster_map)

    def slide_first_seg_to_end(self, input_ids, input_mask, speaker_ids, word_ids, ent_ids, subtoken_map, sentence_len, genre, sentence_map, is_training,
                         gold_starts, gold_ends, gold_mention_cluster_map, non_piece_offset, word_offset, sentence_offset, cluster_offset):

        num_words = sentence_len[:1].sum()

        # truncate subtoken map
        subtoken_map = np.array(subtoken_map)
        word_ids = word_ids[:subtoken_map[num_words-1]+1] #non-pieced words
        word_ids = np.array(word_ids)
        ent_ids = ent_ids[:subtoken_map[num_words-1]+1]
        ent_ids = np.array(ent_ids)
        subtoken_map = subtoken_map[:num_words] + non_piece_offset

        input_ids = np.array(input_ids[:1, :])
        input_mask = np.array(input_mask[:1, :])
        speaker_ids = np.array(speaker_ids[:1, :])
        sentence_len = np.array(sentence_len[:1])

        sentence_map = np.array(sentence_map)
        sentence_map = sentence_map[:num_words] + sentence_offset
        gold_starts, gold_ends = np.array(gold_starts), np.array(gold_ends)
        gold_mention_cluster_map = np.array(gold_mention_cluster_map)
        gold_spans = (gold_starts < num_words) & (gold_ends >= 0)
        # gold_starts = [ 24,  29,  33,  42,  52,  57,  72,  75,  88, 110, 124, 124, 131, 131]
        # gold_ends =   [ 25,  29,  38,  43,  58,  57,  72,  75,  88, 111, 125, 126, 132, 133]
        # cluster_map = [ 1,   2,   2,   1,   4,   3,   3,   3,   4,  1,   1,   4,   1,   4, ]
        gold_starts = gold_starts[gold_spans] + word_offset
        gold_ends = gold_ends[gold_spans] + word_offset
        gold_mention_cluster_map = gold_mention_cluster_map[gold_spans] + cluster_offset

        """
        # do mention masking
        ment_num = len(gold_starts)
        pos_list = np.arange(ment_num, dtype=int)
        sample_num = math.floor(ment_num * 0.1)
        self.ment_mask_num += 1
        sample_ids = random.sample(pos_list.tolist(), sample_num)
        for id in sample_ids:
            for i range(gold_starts[id], gold_ends[id] + 1):
                input_ids[o][i] = self.mask_id
        """
        return (input_ids, input_mask, speaker_ids, word_ids, ent_ids, subtoken_map, sentence_len, genre, sentence_map, \
               is_training, gold_starts, gold_ends, gold_mention_cluster_map)

    def find_num_of_cluster(self, gold_mention_cluster_map):
        j = 0
        for i in gold_mention_cluster_map:
            if i > j:
                j = i
        return j

    def truncate_example(self, input_ids, input_mask, speaker_ids, word_ids, ent_ids, subtoken_map, sentence_len, genre, sentence_map, is_training,
                         gold_starts, gold_ends, gold_mention_cluster_map, sentence_offset=0):
        """ only truncate the last segment, i.e., the 3rd segment """
        max_sentences = 2
        num_sentences = input_ids.shape[0]
        assert num_sentences > max_sentences

        sent_offset = sentence_offset
        word_offset = sentence_len[:sent_offset].sum()
        num_words = sentence_len[sent_offset: sent_offset + max_sentences].sum()

        #truncate subtoken map
        subtoken_map = subtoken_map[word_offset:word_offset+num_words]

        input_ids = input_ids[sent_offset: sent_offset + max_sentences, :]
        input_mask = input_mask[sent_offset: sent_offset + max_sentences, :]
        speaker_ids = speaker_ids[sent_offset: sent_offset + max_sentences, :]
        sentence_len = sentence_len[sent_offset: sent_offset + max_sentences]

        sentence_map = sentence_map[word_offset: word_offset + num_words]
        gold_spans = (gold_starts < word_offset + num_words) & (gold_ends >= word_offset)
        gold_starts = gold_starts[gold_spans] - word_offset
        gold_ends = gold_ends[gold_spans] - word_offset
        gold_mention_cluster_map = gold_mention_cluster_map[gold_spans]

        return (input_ids, input_mask, speaker_ids, word_ids, ent_ids, subtoken_map, sentence_len, genre, sentence_map, \
               is_training, gold_starts, gold_ends, gold_mention_cluster_map)

    def swap_ids(self, id_list, s1, s2, e1, e2):
        new_list = []
        new_list += id_list[:s1]
        new_list += id_list[s2:e2+1]
        new_list += id_list[e1+1:s2]
        new_list += id_list[s1:e1+1]
        new_list += id_list[e2+1:]

        return new_list

    def update_map(self, sentence_map, s1, s2, e1, e2):
        new_map = []
        sent1 = sentence_map[s1]
        sent2 = sentence_map[s2]
        sentence_map_s1 = np.full_like(sentence_map, sent1).tolist()
        sentence_map_s2 = np.full_like(sentence_map, sent2).tolist()
        new_map += sentence_map[:s1]
        new_map += sentence_map_s1[s2:e2+1]
        new_map += sentence_map[e1+1:s2]
        new_map += sentence_map_s2[s1:e1+1]
        new_map += sentence_map[e2+1:]

        return np.array(new_map)

if __name__ == '__main__':
    config_name = sys.argv[1]
    config = util.initialize_config(config_name)
    data_aug = DataAugmentation(config, 512, truncate=False)
    data_aug.aug_and_save()


