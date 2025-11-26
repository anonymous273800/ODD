from datetime import datetime
import math

import tiktoken

from DataFineTuning import AplacaFineTuner
from DriftManager import PlaceholdersUtil, AbruptDriftPlaceholders


class TrieNode:
    def __init__(self, depth=0):
        self.children = dict()
        self.is_end = False
        self.count = 0            # Frequency count
        self.last_update = 0.0    # Timestamp
        self.depth = depth        # Depth = L


class Trie:
    def __init__(self):
        self.root = TrieNode(depth=0)

    # This method for n-gra, insertion
    # def insert_sentence(self, tokens, timestamp=None):
    #     now_ts = timestamp if timestamp is not None else datetime.now().timestamp()
    #     seen_nodes = set()
    #
    #     min_grams, max_grams = 2, len(tokens)
    #     for n in range(min_grams, max_grams + 1):
    #         for i in range(len(tokens) - n + 1):
    #
    #             node = self.root
    #             for tok in tokens[i:i + n]:
    #                 if tok not in node.children:
    #                     node.children[tok] = TrieNode(depth=node.depth + 1)
    #
    #                 node = node.children[tok]
    #
    #                 if id(node) not in seen_nodes:
    #                     node.count += 1
    #                     node.last_update = now_ts
    #                     seen_nodes.add(id(node))
    #
    #             node.is_end = True

    def insert_sentence(self, tokens, timestamp=None):
        """Insert the full tokenized sentence as a single path."""
        now_ts = timestamp if timestamp is not None else datetime.now().timestamp()
        node = self.root

        for tok in tokens:
            if tok not in node.children:
                node.children[tok] = TrieNode(depth=node.depth + 1)

            node = node.children[tok]
            node.count += 1
            node.last_update = now_ts

        node.is_end = True



    def collect_candidate_features(self, prefix_tokens):
        """
        Collect all candidate next tokens with their raw features.
        Returns: list of (token, features).
        """
        candidates = []

        for start in range(len(prefix_tokens)):
            node = self.root
            ok = True

            # walk down from this suffix
            for tok in prefix_tokens[start:]:
                if tok in node.children:
                    node = node.children[tok]
                else:
                    ok = False
                    break

            if not ok:
                continue

            # children are candidate next tokens
            for next_tok, child in node.children.items():
                new_feat = {
                    "L": child.depth,
                    "count": child.count,
                    "last_update": child.last_update,
                    "node": child,
                }
                candidates.append((next_tok, new_feat))

        # print("\n--- [Collected Features] ---")
        # for tok, f in candidates:
        #     print(f"Token={tok} | L={f['L']}, F={f['count']}, R={f['last_update']}")
        return candidates

    def score_next_tokens(
            self,
            prefix_tokens,
            w_len=1.0,
            w_freq=1.0,
            w_recency=1.0,
            eps=1e-9
    ):
        feats = self.collect_candidate_features(prefix_tokens)

        if not feats:
            return []

        prefix_len = len(prefix_tokens)
        # print("prefix_len", prefix_len)

        # gather timestamps
        times = [f["last_update"] for _, f in feats if f["last_update"]]
        # print("times", times)
        if not times:
            return []
        t_max, t_min = max(times), min(times)
        # print("t_max", t_max, "t_min", t_min)
        delta_max = max(eps, t_max - t_min)
        # print("delta_max",delta_max)

        # compute max_Fpp for normalization
        all_Fpp = [math.log1p(f["count"]) for _, f in feats]
        # print("all_Fpp",all_Fpp)
        max_Fpp = max(all_Fpp) + eps
        # print("max_Fpp",max_Fpp)

        best_per_token = {}

        for tok, f in feats:
            # raw features
            L = f["L"] - 1
            F = f["count"]
            R = f["last_update"]

            # transformations
            F_pp = math.log1p(F)  # F''
            F_p = F_pp / max_Fpp  # normalized freq
            L_p = L / (prefix_len + eps)
            R_p = math.exp(-(t_max - R) / delta_max)

            score =  (w_len * L_p + w_freq * F_p + w_recency * R_p)

            details = {
                "L": L, "F": F, "R": R,
                "F_pp": F_pp, "F_p": F_p, "L_p": L_p, "R_p": R_p
            }

            if tok not in best_per_token or score > best_per_token[tok][1]:
                best_per_token[tok] = (tok, score, details)

        # sort candidates by score
        scored = list(best_per_token.values())
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored


def print_trie(node, prefix="", is_tail=True, token="<root>"):
    """
    Pretty-print the trie with F, L, R.
    """
    marker = "└── " if is_tail else "├── "
    if node.depth == 0:
        print("(root)")
    else:
        print(f"{prefix}{marker}{token} (F={node.count}, L={node.depth}, R={int(node.last_update)})")

    children = list(node.children.items())
    for i, (tok, child) in enumerate(children):
        new_prefix = prefix + ("    " if is_tail else "│   ")
        print_trie(child, new_prefix, i == len(children) - 1, tok)


def load_dataset_to_ngram_prefix_tree_text(trie, data, timestamp, tokenizer):
    for idx, entry in enumerate(data):
        formatted_instruction = AplacaFineTuner.format_input(entry)
        response = entry["response"]
        formatted_response = f"\n\n### Response:\n{response}"
        desired_entry = formatted_instruction + formatted_response

        tokens = tokenizer.encode(desired_entry)
        if tokenizer.eos_token_id not in tokens:
            tokens.append(tokenizer.eos_token_id)
        trie.insert_sentence(tokens, timestamp)
    return trie

def load_dataset_to_ngram_prefix_tree_text2(trie, data, timestamp, tokenizer):
    for idx, entry in enumerate(data):
        response = entry["response"]
        tokens = tokenizer.encode(response)
        print(tokens)
        # if tokenizer.eos_token_id not in tokens:
        #     tokens.append(tokenizer.eos_token_id)
        trie.insert_sentence(tokens, timestamp)
    return trie




# if __name__ == "__main__":
#     print("=== Scoring Test Case ===")
#
#     trie = Trie()
#     s1 = ["Every", "effort", "moves", "you", "close"]
#     s2 = ["effort", "moves", "you", "forward"]
#     print("s1",s1)
#     print("s2",s2)
#
#     now = datetime.now().timestamp()
#     trie.insert_sentence(s1, timestamp=now - 7 * 24 * 3600)  # 7 days ago
#     trie.insert_sentence(s2, timestamp=now)  # today
#
#     prefix = ["Every", "effort", "moves", "you"]
#     print("\nQuery prefix:", " ".join(prefix))
#
#     SCORING_CONFIG = {
#         "w_len": 0.3,
#         "w_freq": 0.2,
#         "w_recency": 0.5
#     }
#
#     print_trie(trie.root)
#
#     ranked = trie.score_next_tokens(prefix, **SCORING_CONFIG)
#
#
#     valid_next = {tok for tok, _, _ in ranked}
#     print("valid_next: ",valid_next)
#
#     print("\n--- Final Ranking ---")
#     for tok, score, details in ranked:
#         print(f"{tok} | score={score:.3f} | "
#               f"L={details['L']} (L'={details['L_p']:.3f}), "
#               f"F={details['F']} (F'={details['F_p']:.3f}), "
#               f"R={details['R']} (R'={details['R_p']:.3f})")

# if __name__ == "__main__":
#     print("=== Scoring Test Case ===")
#
#     trie = Trie()
#     s1 = ["Activate", "your", "plan", "4G"]
#     s2 = ["Please", "activate", "your", "plan", "5G"]
#     print("s1",s1)
#     print("s2", s2)
#
#     now = datetime.now().timestamp()
#     one_week_ago =now - 7 * 24 * 3600
#     print("one_week_ago", one_week_ago)
#     print("now", now)
#     trie.insert_sentence(s1, timestamp=one_week_ago)  # 7 days ago
#     trie.insert_sentence(s2, timestamp=now)  # today
#
#     prefix = ["please","activate", "your", "plan"]
#     print("\nQuery prefix:", " ".join(prefix))
#
#     SCORING_CONFIG = {
#         "w_len": 0.3,
#         "w_freq": 0.2,
#         "w_recency": 0.5
#     }
#
#     print_trie(trie.root)
#
#     ranked = trie.score_next_tokens(prefix, **SCORING_CONFIG)
#
#
#     valid_next = {tok for tok, _, _ in ranked}
#     print("valid_next: ",valid_next)
#
#     print("\n--- Final Ranking ---")
#     for tok, score, details in ranked:
#         print(f"{tok} | score={score:.3f} | "
#               f"L={details['L']} (L'={details['L_p']:.3f}), "
#               f"F={details['F']} (F'={details['F_p']:.3f}), "
#               f"R={details['R']} (R'={details['R_p']:.3f})")


if __name__ == "__main__":
    # sentnce = "{{WEBSITE_URL}} www.global-communications.com"
    # trie = Trie()
    # now = datetime.now().timestamp()
    # tokens = [27007, 8845, 33, 12, 21886, 11709, 7324, 13, 20541, 12, 20860, 13, 785]
    # trie.insert_sentence(tokens, now )
    # print_trie(trie.root)

    tokenizer = tiktoken.get_encoding("gpt2")
    tokenizer.eos_token_id = 50256
    data1 = PlaceholdersUtil.generate_placeholder_dataset(AbruptDriftPlaceholders.PLACEHOLDER_VALUES_P1)
    print(data1)
    now = datetime.now().timestamp()
    trie1 = Trie()
    trie1 = load_dataset_to_ngram_prefix_tree_text2(trie1, data1, timestamp=now - 30 * 24 * 3600, tokenizer=tokenizer)

    SCORING_CONFIG = {
        "w_len": 1 / 3,  # 30% weight
        "w_freq": 1 / 3,  # 20% weight
        "w_recency": 1 / 3  # 50% weight
    }
    scoring_args = {k: SCORING_CONFIG[k] for k in ["w_len", "w_freq", "w_recency"] if k in SCORING_CONFIG}

    trie_ranked = trie1.score_next_tokens([22935], **scoring_args)
    print(trie_ranked)