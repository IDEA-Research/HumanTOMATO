import numpy as np
import pickle
from os.path import join as pjoin

POS_enumerator = {
    "VERB": 0,
    "NOUN": 1,
    "DET": 2,
    "ADP": 3,
    "NUM": 4,
    "AUX": 5,
    "PRON": 6,
    "ADJ": 7,
    "ADV": 8,
    "Loc_VIP": 9,
    "Body_VIP": 10,
    "Obj_VIP": 11,
    "Act_VIP": 12,
    "Desc_VIP": 13,
    "OTHER": 14,
}

Loc_list = (
    "left",
    "right",
    "clockwise",
    "counterclockwise",
    "anticlockwise",
    "forward",
    "back",
    "backward",
    "up",
    "down",
    "straight",
    "curve",
)

Body_list = (
    "arm",
    "chin",
    "foot",
    "feet",
    "face",
    "hand",
    "mouth",
    "leg",
    "waist",
    "eye",
    "knee",
    "shoulder",
    "thigh",
)

Obj_List = (
    "stair",
    "dumbbell",
    "chair",
    "window",
    "floor",
    "car",
    "ball",
    "handrail",
    "baseball",
    "basketball",
)

Act_list = (
    "walk",
    "run",
    "swing",
    "pick",
    "bring",
    "kick",
    "put",
    "squat",
    "throw",
    "hop",
    "dance",
    "jump",
    "turn",
    "stumble",
    "dance",
    "stop",
    "sit",
    "lift",
    "lower",
    "raise",
    "wash",
    "stand",
    "kneel",
    "stroll",
    "rub",
    "bend",
    "balance",
    "flap",
    "jog",
    "shuffle",
    "lean",
    "rotate",
    "spin",
    "spread",
    "climb",
)

Desc_list = (
    "slowly",
    "carefully",
    "fast",
    "careful",
    "slow",
    "quickly",
    "happy",
    "angry",
    "sad",
    "happily",
    "angrily",
    "sadly",
)

VIP_dict = {
    "Loc_VIP": Loc_list,
    "Body_VIP": Body_list,
    "Obj_VIP": Obj_List,
    "Act_VIP": Act_list,
    "Desc_VIP": Desc_list,
}


class WordVectorizer(object):
    def __init__(self, meta_root, prefix, text_encode_way):

        self.text_encode_way = text_encode_way

        vectors = np.load(pjoin(meta_root, "%s_data.npy" % prefix))
        words = pickle.load(open(pjoin(meta_root, "%s_words.pkl" % prefix), "rb"))
        word2idx = pickle.load(open(pjoin(meta_root, "%s_idx.pkl" % prefix), "rb"))
        self.word2vec = {w: vectors[word2idx[w]] for w in words}

        if "glove_6B" in self.text_encode_way:
            from torchtext.vocab import GloVe

            glove_6b = GloVe(name="6B", dim=300)
            self.word2vec_glove_6b = glove_6b.get_vecs_by_tokens

    def _get_pos_ohot(self, pos):
        pos_vec = np.zeros(len(POS_enumerator))
        if pos in POS_enumerator:
            pos_vec[POS_enumerator[pos]] = 1
        else:
            pos_vec[POS_enumerator["OTHER"]] = 1
        return pos_vec

    def __len__(self):
        return len(self.word2vec)

    def __getitem__(self, item):
        word, pos = item.split("/")
        if "given_glove" in self.text_encode_way:
            if word in self.word2vec:
                word_vec = self.word2vec[word]
                vip_pos = None
                for key, values in VIP_dict.items():
                    if word in values:
                        vip_pos = key
                        break
                if vip_pos is not None:
                    pos_vec = self._get_pos_ohot(vip_pos)
                else:
                    pos_vec = self._get_pos_ohot(pos)
            else:
                word_vec = self.word2vec["unk"]
                pos_vec = self._get_pos_ohot("OTHER")

        elif "glove_6B" in self.text_encode_way:
            word_vec = self.word2vec_glove_6b([word]).squeeze()

            if word in self.word2vec:
                vip_pos = None
                for key, values in VIP_dict.items():
                    if word in values:
                        vip_pos = key
                        break
                if vip_pos is not None:
                    pos_vec = self._get_pos_ohot(vip_pos)
                else:
                    pos_vec = self._get_pos_ohot(pos)
            else:
                pos_vec = self._get_pos_ohot("OTHER")

        return word_vec, pos_vec


class WordVectorizer_only_text_token(object):
    def __init__(self, meta_root, prefix, text_encode_way):

        self.text_encode_way = text_encode_way

        vectors = np.load(pjoin(meta_root, "%s_data.npy" % prefix))
        words = pickle.load(open(pjoin(meta_root, "%s_words.pkl" % prefix), "rb"))
        word2idx = pickle.load(open(pjoin(meta_root, "%s_idx.pkl" % prefix), "rb"))
        self.word2vec = {w: vectors[word2idx[w]] for w in words}

        if "glove_6B" in self.text_encode_way:
            from torchtext.vocab import GloVe

            glove_6b = GloVe(name="6B", dim=300)
            self.word2vec_glove_6b = glove_6b.get_vecs_by_tokens

    def __len__(self):
        return len(self.word2vec)

    def __getitem__(self, item):
        word = item

        if "given_glove" in self.text_encode_way:
            if word in self.word2vec:
                word_vec = self.word2vec[word]
            else:
                word_vec = self.word2vec["unk"]

        elif "glove_6B" in self.text_encode_way:
            word_vec = self.word2vec_glove_6b([word]).squeeze()

        return word_vec
