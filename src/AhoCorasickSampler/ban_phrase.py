import warnings
#import gc
import torch
from transformers import PreTrainedModel, AutoTokenizer, AutoModelForCausalLM, PretrainedConfig
import torch.nn.functional as F
from collections import deque, defaultdict
from typing import List, Dict, Tuple, Deque, Any
from dataclasses import dataclass

Token = int  # The language model token ID type
AhoCorasickIndex = int  # Type for unique identifiers of nodes on the trie

@dataclass
class ProbTrieNode:
    """
    If we think of each node Aho Corasick as representing a letter in a GUI word processor, then
    `ProbTrieNode`s reside at the positions inhabitable by the cursor, i.e. in between letters.
    `probs` gives the cached probabilites for letters (tokens) for the position immediately after
    the cursor position.
    """
    state: AhoCorasickIndex # Where we were on the Aho-Corasick trie when we created this ProbTrieNode
    parent: 'ProbTrieNode'
    probs: torch.Tensor # Initialized to the LLM-calculated probabilities, and edited and renormalized when we ban tokens.
    children: Dict[Token, 'ProbTrieNode']

class AhoCorasickSamplerConfig(PretrainedConfig):
    model_type = "AhoCorasickSampler"

    def __init__(self,
        inner_model_spec: Dict = None,
            # Parameters that should be passed to `AutoModelForCausalLM.from_pretrained()` to load the inner model.
            # Must include `pretrained_model_name_or_path`. 
            # `None` doesn't make sense but somehow `PretrainedConfig.__init__()` must provide default values;
            # I ran into https://discuss.huggingface.co/t/custom-config-error-when-model-save-pretrained/26361 so I have to comply.
        eos_token_id: Token = None,
            # This can be `None` and results in sampling never finishing.
        banned_phrases: List[List[Token]] = None,
            # This can be `None` and results in an empty set of banned phrases.
        **kwargs
    ):
        super().__init__(**kwargs)
        if inner_model_spec is not None:
            assert "pretrained_model_name_or_path" in inner_model_spec
        self.inner_model_spec = inner_model_spec
        self.eos_token_id = eos_token_id
        self.banned_phrases = banned_phrases or []

class AhoCorasickSampler(PreTrainedModel):
    config_class = AhoCorasickSamplerConfig

    def __init__(self, config):
        super().__init__(PretrainedConfig())

        self.inner_model = AutoModelForCausalLM.from_pretrained(**(config.inner_model_spec))
        self.eos_token_id = config.eos_token_id

        self.trie: List[Dict[Token, AhoCorasickIndex]] = []
        self.match: List[Bool] = []             # whether path leading to a node matches a banned phrase
        self.fail: List[AhoCorasickIndex] = []  # failure links
        self.depth: List[AhoCorasickIndex] = [] # depth of each node
        self._build_trie(config.banned_phrases)
        self._build_automaton()

    def _build_trie(self, banned_phrases: List[List[Token]]) -> None:
        # initialize root
        self.trie.append({})
        self.match.append(False)
        self.fail.append(0)
        self.depth.append(0)

        for idx, pat in enumerate(banned_phrases):
            state = 0
            for token in pat:
                if token not in self.trie[state]:
                    self.trie[state][token] = len(self.trie)
                    self.trie.append({})
                    self.match.append(False)
                    self.fail.append(0)
                    self.depth.append(self.depth[state] + 1)
                state = self.trie[state][token]
            # mark output at pattern end state
            self.match[state] |= True;

    def _build_automaton(self) -> None:
        # initialize queue
        queue: Deque[int] = deque()
        # set depth-1 nodes fail to 0, push them
        for token, nxt in self.trie[0].items():
            self.fail[nxt] = 0
            queue.append(nxt)
        # BFS
        while queue:
            r = queue.popleft()
            for token, s in self.trie[r].items():
                queue.append(s)
                # compute f = failure(r)
                state = self.fail[r]
                while state != 0 and token not in self.trie[state]:
                    state = self.fail[state]
                self.fail[s] = self.trie[state].get(token, 0)
                # merge output
                self.match[s] += self.match[self.fail[s]]

    def _get_depth(self, node: AhoCorasickIndex) -> int:
        return self.depth[node]

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        *model_args,
        banned_phrases=None,
        **kwargs
    ):
        # Load the base config (reads config.json)
        config = cls.config_class.from_pretrained( # This `from_pretrained()` is inherited and good enough
            pretrained_model_name_or_path,
            **{k: v for k, v in kwargs.items() if k in cls.config_class.__init__.__code__.co_varnames}
        )
        # Override with caller’s list if provided
        if banned_phrases is not None:
            config.banned_phrases = banned_phrases
        # Now instantiate model
        return AhoCorasickSampler(config)

    def visualize_trie(self, tablength=4) -> str:
        output = ''
        def _visualize_node(node_id, token_from_parent, indent):
            """
            Visualize the trie structure. It prints each node with its token (for the edge leading to it)
            and whether it contains a match (i.e., end of a pattern) into a string.
            """
            nonlocal output
            # Print the current node with token edge coming from parent and match status
            match_marker = " (match)" if self.match[node_id] else ""
            # For node_id 0 (root) we may not have a token_from_parent
            token_str = f"[{token_from_parent}]" if token_from_parent is not None else "[root]"
            output += " " * indent + f"{token_str} -> Node {node_id}{match_marker}" + '\n'
            # Iterate through children
            for token, child_id in self.trie[node_id].items():
                _visualize_node(child_id, token, indent + tablength)

        _visualize_node(0, None, 0)
        return output

    @torch.no_grad()
    def generate(self, inputs: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        `inputs` should take the form of the return value of AutoTokenizer, namely: {'input_ids': tensor, 'attention_mask': tensor}.
        returns completion (including inputs) that follows the distribution as in the writeup.
        """
        assert isinstance(inputs, torch.Tensor) and len(inputs.shape) == 2, 'Unsupported inputs tensor shape'
        assert inputs.shape[0] == 1, 'Batch size > 1 is not supported'
        if 'return_dict_in_generate' in kwargs and kwargs['return_dict_in_generate']:
            raise NotImplementedError('`ModelOutput` output type not supported yet; we can only return tensors for now.')
        max_length = kwargs.pop('max_length', 20)
        # Default is 20 according to https://github.com/huggingface/transformers/blob/471958b6208bb9e94e6305d279fad9a05aa42c36/src/transformers/generation/flax_utils.py#L386
        if kwargs:
            warnings.warn(f'`AhoCorasickSampler.generate()` does not have the following options implemented yet:\n{kwargs.keys()}')
        if torch.isin(self.eos_token_id, inputs):
            return inputs # Nothing to do

        device = next(self.inner_model.parameters()).device # `next()` here gets the first in iterable

        state: AhoCorasickIndex = 0
        # First go through the given inputs; if the inputs end with part of a banned phrase, `state` can end up being non-root
        for i, id in enumerate(inputs[0].tolist()):
            while id not in self.trie[state] and self._get_depth(state) != 0: # Going out of trie; can commit at least one token.
                state = self.fail[state]
            if id in self.trie[state]:
                state = self.trie[state][id]
            assert not self.match[state], f"The inputs already contain the banned phrase ending at node {state} on trie.\n" +\
                f"Offending sequence:\n{inputs[0].tolist()[i-self._get_depth(state)+1:]}" +\
                f"The trie:\n{self.visualize_trie()}" 

        completion = inputs.detach().clone()
        probs_trie_pointer = ProbTrieNode(state, None, None, {})
        # This pointer now points to this "root" node, but soon will point to its descendants;
        # this root itself will be freed at some point.

        while completion.shape[1] < max_length: # Every iteration of this loop starts with fresh (empty) `uncommitted_tokens`.
                    # `state` will also have been dailed back upon encountering a banned phrase.
            uncommitted_tokens: List[Token] = []

            while True: # This breaks only if we get a banned phrase.
                if probs_trie_pointer.probs is not None:
                    probs = probs_trie_pointer.probs
                else:
                    logits = self.inner_model(torch.cat((completion, torch.tensor([uncommitted_tokens], dtype=inputs.dtype).to(device)), dim=-1)).logits[:, -1, :]
                    probs = F.softmax(logits, dim=-1)  # (batch, vocab_size)
                    probs_trie_pointer.probs = probs
                id: Token = torch.multinomial(probs, num_samples=1).item()
                if id == self.eos_token_id:
                    # All set; just return and no need to maintain all the data structures anymore.
                    completion = torch.cat((completion, torch.tensor([uncommitted_tokens], dtype=inputs.dtype).to(device)), dim=-1)
                    return completion
                uncommitted_tokens.append(id)

                # Maintain the prob tree data structure.
                if id not in probs_trie_pointer.children:
                    probs_trie_pointer.children[id] = ProbTrieNode(state, probs_trie_pointer, None, {})
                probs_trie_pointer = probs_trie_pointer.children[id]

                # Move along the Aho-Corasick trie
                old_depth = self._get_depth(state)
                while id not in self.trie[state] and self._get_depth(state) != 0: # Going out of trie; can commit at least one token.
                    state = self.fail[state]
                if id in self.trie[state]:
                    state = self.trie[state][id]
                #print(f'Sampled token "{self.tokenizer.decode(id)}", {id=}')
                #print(f'Trie {state=}, depth={self._get_depth(state)}')

                if self.match[state]: # Just got a banned phrase; need to backtrack
                    #print('Got banned phrase! Backtracking...')
                    Z = 0
                    for i in range(self._get_depth(state)-1, 0-1, -1):
                        # At this point, Z is sum of unnormalized prob of all not-banned (i+1)-th tokens conditioned on the <=i-th
                        probs_trie_pointer = probs_trie_pointer.parent
                        probs_trie_pointer.probs[0,uncommitted_tokens[i]] *= Z
                        Z = torch.sum(probs_trie_pointer.probs)
                        # At this point, Z is sum of unnormalized prob of all not-banned i-th tokens conditioned on the <=(i-1)-th
                        probs_trie_pointer.probs /= Z
                    state = probs_trie_pointer.state
                    break # Now we should break to the outer loop, and sample again from the modified distribution.
                    # Sanity check: states are correct: `state` and `probs_trie_pointer` have been returned to as far back as they should,
                    # and at the beginning of the loop, `uncommitted_tokens` are cleared.

                # Not a match yet, and we're safe for now
                new_depth = self._get_depth(state)
                num_to_commit = old_depth + 1 - new_depth
                if num_to_commit > 0: # It's also fine if we don't check this, because if num_to_commit == 0 then the following code is a no-op.
                    num_to_commit = min(max_length - completion.shape[1], num_to_commit)
                    completion = torch.cat((completion, torch.tensor([uncommitted_tokens[:num_to_commit]], dtype=inputs.dtype).to(device)), dim=-1)
                    uncommitted_tokens = uncommitted_tokens[num_to_commit:]
                    
                    # We now trace upwards in the prob cache trie and "disconnect" the nodes for already committed
                    # tokens, so that the memory will be freed by Python GC. (We may disconnect than one node and
                    # they may reference each other, but Python is powerful to detect loops too and free them all.)
                    pruner_pointer = probs_trie_pointer
                    for i in range(new_depth):
                        pruner_pointer = pruner_pointer.parent
                    pruner_pointer.parent = None
                    #print(f'Collected {gc.collect()} objects.') # `gc.collect()` triggers GC immediately and returns stats.

        # If we reach here, we just overran the `max_length`.
        completion[0][-1] = self.eos_token_id
        return completion

    def save_pretrained(self, save_directory, **kwargs):
        raise NotImplementedError("Not seeing the need to save this now; training is not supported yet anyway.")
