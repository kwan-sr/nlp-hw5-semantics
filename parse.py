#!/usr/bin/env python3
"""
Determine whether sentences are grammatical under a CFG, using Earley's algorithm.
(Starting from this basic recognizer, you should write a probabilistic parser
that reconstructs the highest-probability parse of each given sentence.)
"""

# Recognizer code by Arya McCarthy, Alexandra DeLucia, Jason Eisner, 2020-10, 2021-10.
# This code is hereby released to the public domain.

############ Vanilla Probabilistic Earley Parsing without speedups ##########################
#####   - uses -(log2prob) as weight; min weight is the best tree                       #####
#####   - stores 2 backpointers each to allow for tree recreation                       #####
#####   - modified duplicate checking to ignore weights and backpointers                #####
#####   - includes O(1) reprocessing (remove heavier tree by setting old entry to None) #####
#####     and re-appending to end of column                                          ########
#############################################################################################

from __future__ import annotations
import argparse
import logging
import math
import tqdm
from dataclasses import dataclass
from pathlib import Path
from collections import Counter
from typing import Counter as CounterType, Iterable, List, Optional, Dict, Tuple

log = logging.getLogger(Path(__file__).stem)  # For usage, see findsim.py in earlier assignment.

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "grammar", type=Path, help="Path to .gr file containing a PCFG'"
    )
    parser.add_argument(
        "sentences", type=Path, help="Path to .sen file containing tokenized input sentences"
    )
    parser.add_argument(
        "-s",
        "--start_symbol", 
        type=str,
        help="Start symbol of the grammar (default is ROOT)",
        default="ROOT",
    )

    parser.add_argument(
        "--progress", 
        action="store_true",
        help="Display a progress bar",
        default=False,
    )

    # for verbosity of logging
    parser.set_defaults(logging_level=logging.INFO)
    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument(
        "-v", "--verbose", dest="logging_level", action="store_const", const=logging.DEBUG
    )
    verbosity.add_argument(
        "-q", "--quiet",   dest="logging_level", action="store_const", const=logging.WARNING
    )

    return parser.parse_args()


class EarleyChart:
    """A chart for Earley's algorithm."""
    
    def __init__(self, tokens: List[str], grammar: Grammar, progress: bool = False) -> None:
        """Create the chart based on parsing `tokens` with `grammar`.  
        `progress` says whether to display progress bars as we parse."""
        self.tokens = tokens
        self.grammar = grammar
        self.progress = progress
        self.profile: CounterType[str] = Counter()

        self.cols: List[Agenda]
        self._run_earley() 

    # * This function is heavily modified to print the lowest weight tree 
    # * instead of being just a recognizer
    def accepted(self) -> bool:
        """Was the sentence accepted?
        That is, does the finished chart contain an item corresponding to a parse of the sentence?
        This method answers the recognition question, but not the parsing question."""
        there_is_parse = False 
        min_prob = float(math.inf)  # min weight starts at infinity
        min_item = None             # keeps track of min weight complete parse
        for item in self.cols[-1].all():    # examine the last column
            if (item is not None) and(item.rule.lhs == self.grammar.start_symbol # find all completed ROOT entries in last col
                and item.next_symbol() is None          
                and item.start_position == 0):               
                    there_is_parse = True # then we know we have a parse, but not necessarily the lowest weight one
                    if item.subtree_prob <= min_prob : # keeps track of min prob tree by checking with min so far
                        min_prob = item.subtree_prob
                        min_item = item
        if there_is_parse == True : # we have a tree, recursively prints the parse tree, then the tree weight
            print(self.print_tree(min_item, ""))
            print(min_prob)
            return True
        else : # we didn't find a complete parse tree
            print("NONE")
            return False   
    
    # * Takes a completed tree item (with start symbol as lhs, start at 0, ends at last column) 
    # * recursively follows backpointers to print the parse tree, adding parentheses as we go
    # * returns a str of the parse tree, compatible with ./prettyprint
    def print_tree(self, item: Item, result:str) :
        if item is None: # base case: empty tree
            return result
        elif (item.backpointer1 is None) and (item.backpointer2 is None) : # no parents, don't print anything
            return result
        else : # all cases where there is at least 1 backpointer -> terminal and non-terminal nodes
            rule_open = ""
            rule_end = ""
            terminal = ""

            # if current node is a completed consituent, it should be contained in ( )
            # we know it is a complete consituent if the rule is finished, with dot at the end
            if item.next_symbol() is None :  
                rule_open = "(" + item.rule.lhs + " "
                rule_end = ")"

            # now recursively print the left tree
            left_tree = self.print_tree(item.backpointer2, result)

            # in our system, terminals have only 1 backpointer (the latest scanned element)
            # thus if backpointer1 is None but backpointer2 exists, it is a terminal.
            if item.backpointer1 is None : # we print that non-terminal out
                terminal = item.rule.rhs[item.dot_position-1]
            
            # then we recursively print the latest attached consituent, which is in the right tree
            right_tree = self.print_tree(item.backpointer1, result)

            # put it all together in a string with correct order
            result += rule_open + left_tree + " " + terminal + right_tree + rule_end
            return result

    def _run_earley(self) -> None:
        """Fill in the Earley chart."""
        # Initially empty column for each position in sentence
        self.cols = [Agenda() for _ in range(len(self.tokens) + 1)]

        # Start looking for ROOT at position 0
        self._predict(self.grammar.start_symbol, 0)

        for i, column in tqdm.tqdm(enumerate(self.cols),
                                   total=len(self.cols),
                                   disable=not self.progress):
            log.debug("")
            log.debug(f"Processing items in column {i}")
            while column:    # while agenda isn't empty
                item = column.pop()   # dequeue the next unprocessed item
                while item is None:   # skip all the heavier things that we killed and reprocessed
                    item = column.pop() 
                next = item.next_symbol()
                if next is None:
                    log.debug(f"{item} => ATTACH")
                    self._attach(item, i)   
                elif self.grammar.is_nonterminal(next):
                    log.debug(f"{item} => PREDICT")
                    self._predict(next, i)
                else:
                    log.debug(f"{item} => SCAN")
                    self._scan(item, i)                 

    def _predict(self, nonterminal: str, position: int) -> None:
        """Start looking for this nonterminal at the given position."""
        for rule in self.grammar.expansions(nonterminal):
            # When predicting, the weight is the rule weight, there are no backpointers yet
            new_item = Item(rule, dot_position=0, start_position=position,subtree_prob=rule.weight, backpointer1=None, backpointer2 = None)
            self.cols[position].push(new_item)

            log.debug(f"\tPredicted: {new_item} in column {position}")
            self.profile["PREDICT"] += 1

    def _scan(self, item: Item, position: int) -> None:
        """Attach the next word to this item that ends at position, 
        if it matches what this item is looking for next."""
        if position < len(self.tokens) and self.tokens[position] == item.next_symbol():
            # When advancing the dot after scanning, we pass in None and item as backpointers
            # None says that we don't have the latest attached constituent (because we're not attaching here)
            # item is the previous version of the rule, with the dot one step earlier. 
            # This helps us keep track of previously attached constituents if any
            new_item = item.with_dot_advanced(None, item) 
            self.cols[position + 1].push(new_item)

            log.debug(f"\tScanned to get: {new_item} in column {position+1}")
            self.profile["SCAN"] += 1

    def _attach(self, item: Item, position: int) -> None:
        """Attach this complete item to its customers in previous columns, advancing the
        customers' dots to create new items in this column.  (This operation is sometimes
        called "complete," but actually it attaches an item that was already complete.)
        """
        mid = item.start_position   # start position of this item = end position of item to its left
        for customer in self.cols[mid].all():  # TODO: could you eliminate this inefficient linear search?
            if (customer is not None) and (customer.next_symbol() == item.rule.lhs):
                # pass in two parents to create next item: 
                # parent1: the current rule which has backpointers to all preceding consitutents
                # parent2: the constituent being attached right now (that allows us to move the dot)
                new_item = customer.with_dot_advanced(item, customer) 

                # add the weight of constituent being attached to the subtree weight
                new_item = Item(rule=new_item.rule, dot_position=new_item.dot_position, start_position=new_item.start_position, subtree_prob=new_item.subtree_prob+item.subtree_prob, backpointer1=new_item.backpointer1, backpointer2=new_item.backpointer2)
                
                # add the new item to the column
                self.cols[position].push(new_item)

                log.debug(f"\tAttached to get: {new_item} in column {position}")
                self.profile["ATTACH"] += 1


# * We modified the push() method of this class to account for reprocessing
# * please see further comments in push() method
class Agenda:
    """An agenda of items that need to be processed.  Newly built items 
    may be enqueued for processing by `push()`, and should eventually be 
    dequeued by `pop()`.

    This implementation of an agenda also remembers which items have
    been pushed before, even if they have subsequently been popped.
    This is because already popped items must still be found by
    duplicate detection and as customers for attach.  

    (In general, AI algorithms often maintain a "closed list" (or
    "chart") of items that have already been popped, in addition to
    the "open list" (or "agenda") of items that are still waiting to pop.)

    In Earley's algorithm, each end position has its own agenda -- a column
    in the parse chart.  (This contrasts with agenda-based parsing, which uses
    a single agenda for all items.)

    Standardly, each column's agenda is implemented as a FIFO queue
    with duplicate detection, and that is what is implemented here.
    However, other implementations are possible -- and could be useful
    when dealing with weights, backpointers, and optimizations.


    """

    def __init__(self) -> None:
        self._items: List[Item] = []       # list of all items that were *ever* pushed
        self._index: Dict[Item, int] = {}  # stores index of an item if it was ever pushed
        self._next = 0                     # index of first item that has not yet been popped

    def __len__(self) -> int:
        """Returns number of items that are still waiting to be popped.
        Enables `len(my_agenda)`."""
        return len(self._items) - self._next

    #* modified for reprocessing
    def push(self, item: Item) -> None:
        """Add (enqueue) the item, unless it was previously added."""
        if item not in self._index:    # O(1) lookup in hash table
            self._items.append(item)
            self._index[item] = len(self._items) - 1
        else: # if it's a duplicate, and has lower weight, we reprocess
            if item.subtree_prob < self._items[self._index[item]].subtree_prob :
                # reprocess
                self._items[self._index[item]] = None # kill the heavier tree
                self._items.append(item)    # re-append to the end
                self._index[item] = len(self._items) - 1

    def pop(self) -> Item:
        """Returns one of the items that was waiting to be popped (dequeued).
        Raises IndexError if there are no items waiting."""
        if len(self)==0:
            raise IndexError
        item = self._items[self._next]
        self._next += 1
        return item

    def all(self) -> Iterable[Item]:
        """Collection of all items that have ever been pushed, even if 
        they've already been popped."""
        return self._items

    def __repr__(self):
        """Provide a human-readable string REPResentation of this Agenda."""
        next = self._next
        return f"{self.__class__.__name__}({self._items[:next]}; {self._items[next:]})"

class Grammar:
    """Represents a weighted context-free grammar."""
    def __init__(self, start_symbol: str, *files: Path) -> None:
        """Create a grammar with the given start symbol, 
        adding rules from the specified files if any."""
        self.start_symbol = start_symbol
        self._expansions: Dict[str, List[Rule]] = {}    # maps each LHS to the list of rules that expand it
        # Read the input grammar files
        for file in files:
            self.add_rules_from_file(file)

    def add_rules_from_file(self, file: Path) -> None:
        """Add rules to this grammar from a file (one rule per line).
        Each rule is preceded by a normalized probability p,
        and we take -log2(p) to be the rule's weight."""
        with open(file, "r") as f:
            for line in f:
                # remove any comment from end of line, and any trailing whitespace
                line = line.split("#")[0].rstrip()
                # skip empty lines
                if line == "":
                    continue
                # Parse tab-delimited line of format <probability>\t<lhs>\t<rhs>
                _prob, lhs, _rhs = line.split("\t")
                prob = float(_prob)
                rhs = tuple(_rhs.split())  
                rule = Rule(lhs=lhs, rhs=rhs, weight=-math.log2(prob))
                if lhs not in self._expansions:
                    self._expansions[lhs] = []
                self._expansions[lhs].append(rule)

    def expansions(self, lhs: str) -> Iterable[Rule]:
        """Return an iterable collection of all rules with a given lhs"""
        return self._expansions[lhs]

    def is_nonterminal(self, symbol: str) -> bool:
        """Is symbol a nonterminal symbol?"""
        return symbol in self._expansions


# A dataclass is a class that provides some useful defaults for you. If you define
# the data that the class should hold, it will automatically make things like an
# initializer and an equality function.  This is just a shortcut.  
# More info here: https://docs.python.org/3/library/dataclasses.html
# Using a dataclass here lets us declare that instances are "frozen" (immutable),
# and therefore can be hashed and used as keys in a dictionary.
@dataclass(frozen=True)
class Rule:
    """
    A grammar rule has a left-hand side (lhs), a right-hand side (rhs), and a weight.

    >>> r = Rule('S',('NP','VP'),3.14)
    >>> r
    S → NP VP
    >>> r.weight
    3.14
    >>> r.weight = 2.718
    Traceback (most recent call last):
    dataclasses.FrozenInstanceError: cannot assign to field 'weight'
    """
    lhs: str
    rhs: Tuple[str, ...]
    weight: float = 0.0

    def __repr__(self) -> str:
        """Complete string used to show this rule instance at the command line"""
        # Note: You might want to modify this to include the weight.
        return f"{self.lhs} → {' '.join(self.rhs)}"

    
# We particularly want items to be immutable, since they will be hashed and 
# used as keys in a dictionary (for duplicate detection).  
@dataclass(frozen=True)
class Item:
    """An item in the Earley parse chart, representing one or more subtrees
    that could yield a particular substring."""
    rule: Rule
    dot_position: int
    start_position: int
    subtree_prob: float
    backpointer1: Item
    backpointer2: Item


    # * Redefine __hash__ and __eq__ to compare just the rule, dot, start;
    # * ignoring weights & backpointers for duplicate checking
    def __hash__(self):
        return hash((self.rule, self.dot_position, self.start_position))
    
    def __eq__(self, other):
        if not isinstance(other, Item):
            return False
        return self.rule == other.rule and self.start_position == other.start_position and self.dot_position == other.dot_position


    def next_symbol(self) -> Optional[str]:
        """What's the next, unprocessed symbol (terminal, non-terminal, or None) in this partially matched rule?"""
        assert 0 <= self.dot_position <= len(self.rule.rhs)
        if self.dot_position == len(self.rule.rhs):
            return None
        else:
            return self.rule.rhs[self.dot_position]

    def with_dot_advanced(self, parent1:Item, parent2:Item) -> Item:
        if self.next_symbol() is None:
            raise IndexError("Can't advance the dot past the end of the rule")
    
        return Item(rule=self.rule, dot_position=self.dot_position + 1, start_position=self.start_position, subtree_prob = self.subtree_prob, backpointer1=parent1, backpointer2 = parent2)

    def __repr__(self) -> str:
        """Human-readable representation string used when printing this item."""
        DOT = "·"
        rhs = list(self.rule.rhs)  # Make a copy.
        rhs.insert(self.dot_position, DOT)
        dotted_rule = f"{self.rule.lhs} → {' '.join(rhs)}"
        return f"({self.start_position}, {self.subtree_prob}, {dotted_rule})" # we also print the weight


def main():
    # Parse the command-line arguments
    args = parse_args()
    logging.basicConfig(level=args.logging_level) 

    grammar = Grammar(args.start_symbol, args.grammar)

    with open(args.sentences) as f:
        for sentence in f.readlines():
            sentence = sentence.strip()
            if sentence != "":  # skip blank lines
                # analyze the sentence
                log.debug("="*70)
                log.debug(f"Parsing sentence: {sentence}")
                chart = EarleyChart(sentence.split(), grammar, progress=args.progress)
                # print the result
                chart.accepted()
                log.debug(f"Profile of work done: {chart.profile}")


if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=False)   # run tests
    main()
