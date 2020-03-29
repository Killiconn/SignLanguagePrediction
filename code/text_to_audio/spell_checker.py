from spellchecker import SpellChecker

def check_spelling(string):
	#set of letter that are never double in the english alphabet
	non_double = set(["j", "k", "z", "u", "q", "x", "y", "w"])
	d = {}
	possibilities = checker(string, d, non_double)

	#Get the most common word that came up in the dictionary
	maxi = 0
	for k,v in possibilities.items():
		if v > maxi:
			most_common = k
			maxi = v

	return most_common

def checker(string, dic, non_double):
	#Since duplicated letters come up quite commonly this checker method
	#passes sliced words. If two letters are the same side by side,
	#one of the letters is sliced out and passed through a spell correction method.
	spell = SpellChecker()
	if string in dic:
		dic[string] += 1
	else:
		dic[string] = 1

	for i in range(0, len(string)-1):
		if string[i] == string[i+1] and string[i] not in non_double:
			curr_word = string[:i] + string[i+1:]
			new_word = spell.correction(curr_word)
			if new_word in dic:
				dic[new_word] += 1
			else:
				dic[new_word] = 1
			checker(curr_word, dic, non_double)
	return dic

