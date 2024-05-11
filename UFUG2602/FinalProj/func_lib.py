# funcition_lib.py (need to finish)

# A basic function libary which is suitable for a large search project construction and block-testing

#---------------------------- Split line ----------------------------#

# Please write your code answer below, and run the code to check whether the outputs of your method are correct.

# Note: We strongly suggest you to follow the given code template to help you think and organize your code structure.
#	   Still, any changes are supported with corresponding clear comments.


# You can choose whether to use the demo codes below by yourself.

# If you modify the code template, please make sure that the corresponding testing usages will be added in your programm.
# Otherwise, we think that your answer is not valid (without promising outputs)

#---------------------------- Split line ----------------------------#
# Targets:

# Implement a prefix tree (Trie) for efficient storage and retrieval of words along with functions to perform spell checking and process queries with logical operations like AND (`+`) and OR (`|`).
# It includes functionality to find exact matches and approximate matches based on edit distance with a threshold, leveraging dynamic programming within the Trie structure.

#---------------------------- Split line ----------------------------#

# Note 1: The expressions can include operands that are effectively sets of results from previous searches (either exact or approximate word matches), combined using logical operators.



from collections import defaultdict
from math import ceil
from functools import cache
from hyperpara import *
import math


# Trie-based search algorithm

# METHODS:
	# inserting words into the trie
	# finding exact matches
	# finding approximate matches (with a given error threshold)

# Hint 1: Cache for speeding up future queries. This will help to quickly provide results without recalculating.
# Hint 2: Considering sometimes the inputs from users are just some substrings of one complete word, or some words which are not all exactly correct (misspelled),


class TrieNode:

	# `TrieNode`:
	#			 Represents a node in the trie.
	#			 Each node has a dictionary of child nodes (`children`) indexed by characters,
	#						   a flag indicating whether it marks the end of a word (`is_end_of_word`)

	def __init__(self):
		self.children : dict[str, TrieNode] = {}
		self.is_end_of_word : bool = False
		self.dp : list = []

class Trie:

	# `Trie`:
	#		 The trie class containing the root node and a cache for misspelled words.

	def __init__(self):
		self.root : TrieNode = TrieNode()
		 # Initialize a cache for misspelled words

	def insert(self, word : str) :
		if word == '': return 
		for i in range(len(word)) :
			# insert all the substring of the word into the trie
			subWord = word[i : ]
			nd : TrieNode = self.root
			for j in range(len(subWord)) :
				ch = subWord[j]
				if ch not in nd.children :
					nd.children[ch] = TrieNode()
				nd = nd.children[ch]
				# since the search is basic on the substring, we can treat every node as the end of a word
				nd.is_end_of_word = True

	@cache
	def findCandidates(self, keyword) -> int :
		# Note first to check if the word is in the cache
		res = -1
		if len(keyword) < APPROX_LEN_THRESHOLD : res = 0 if self.find_exact(keyword) else -1
		else : res = self.find_approximate(keyword, ERROR_THRESHOLD if ERROR_THRESHOLD >= 1 else int(ERROR_THRESHOLD * len(keyword)))
		return res

		# return results

	def find_exact(self, word) -> bool :
		nd : TrieNode = self.root
		for ch in word :
			if ch not in nd.children :
				return False
			nd = nd.children[ch]
		return True
	
	def find_approximate(self, word : str, dist : int) -> int :
		self.root.dp = [i for i in range(len(word) + 1)]
		ans = min(len(word), self._search(self.root, word, 0, dist))
		if ans <= dist : return ans
		else : return -1
		
	def _search(self, node : TrieNode, word, pos, dist) -> int :
		# Use dynamic programming (based on edit distance) to find words in the trie that are within the specified error threshold from the given word		
		ans = math.inf
		# if min(node.dp) > dist : return min(node.dp)
		for (ch, child) in node.children.items() :
			child.dp = [0 for i in range(len(word) + 1)]
			child.dp[0] = pos + 1
			for i in range(1, len(word) + 1) :
				child.dp[i] = min(child.dp[i - 1] + 1, node.dp[i] + 1, node.dp[i - 1] + (1 if word[i - 1] != ch else 0))
			ans = min(ans, child.dp[len(word)], self._search(child, word, pos + 1, dist))
		node.dp = []
		return ans


#---------------------------- Split line ----------------------------#



#---------------------------- Split line ----------------------------#

# Bouns_part

def op_AND(a, b, trieRoot : Trie) -> int :
	if isinstance(a, int) or isinstance(b, int) :
		if not isinstance(a, int) : a, b = b, a
		if a == -1 : return -1
		if not isinstance(b, int) : b = trieRoot.findCandidates(b)
	else :
		a = trieRoot.findCandidates(a)
		if a == -1 : return -1
		b = trieRoot.findCandidates(b)
	return a + b if a != -1 and b != -1 else -1

def op_OR(a, b, trieRoot : Trie) -> int :
	if isinstance(a, int) or isinstance(b, int) :
		if not isinstance(a, int) : a, b = b, a
		if not isinstance(b, int) : b = trieRoot.findCandidates(b)
	else :
		a = trieRoot.findCandidates(a)
		b = trieRoot.findCandidates(b)
	return min(a if a != -1 else math.inf, b if b != -1 else math.inf) if a != -1 or b != -1 else -1

# Given codes to help to parse and evaluate the expressions (not need to implement, but please read it carefully)

def precedence(op):
	"""Return the precedence of the given operator."""
	if op == '+':
		return 2
	elif op == '|':
		return 1
	return 0

def tokenize(expression : str) -> list[str] :
	"""Convert the string expression into a list of tokens with implicit '+'."""
	tokens : list[str] = []
	i = 0
	last_char = None

	while i < len(expression):
		if expression[i].isspace():
			i += 1
		elif expression[i].isalnum():  # Operand
			start = i
			while i < len(expression) and expression[i].isalnum():
				i += 1
			token = expression[start:i]

			# If last token is also an operand, insert an implicit '+'
			if tokens and tokens[-1].isalnum():
				tokens.append('+')
			tokens.append(token)
		else:  # Operator or parenthesis
			tokens.append(expression[i])
			i += 1
		
	return tokens

def infix_to_postfix(tokens : list[str]) -> list[str] :
	"""Convert infix expression to postfix using the Shunting Yard algorithm."""
	stack : list[str] = []
	output : list[str] = []

	for token in tokens:
		if token.isalnum():  # Operand
			output.append(token)
		elif token == '(':
			stack.append(token)
		elif token == ')':
			while stack and stack[-1] != '(':
				output.append(stack.pop())
			stack.pop()  # pop '('
		else:  # Operator
			while stack and precedence(stack[-1]) >= precedence(token):
				output.append(stack.pop())
			stack.append(token)
	while stack:
		output.append(stack.pop())
	return output

# Note 2: The `evaluate_postfix` function processes a postfix expression which simplifies the evaluation of expressions by eliminating the need for parentheses and making operator processing straightforward.

def evaluate_postfix(tokens : list[str], trieRoot : Trie) -> int :
	"""Evaluate a postfix expression"""
	stack : list = []
	
	tokens.append(-1)
	tokens.append('|')

	
	for token in tokens:
		if token == '+':
			b = stack.pop()
			a = stack.pop()
			result = op_AND(a, b, trieRoot)
			stack.append(result)
		elif token == '|':
			b = stack.pop()
			a = stack.pop()
			result = op_OR(a, b, trieRoot)
			stack.append(result)
		else:  # Operand, Set
			stack.append(token)  # Convert '0' or '1' to False or True

	tokens.pop(); tokens.pop()
	
	return stack.pop()
