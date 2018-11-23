import difflib

l = ['hell', 'hello', 'hellow']
s = ['hello']

seq = difflib.SequenceMatcher(None, l, s)
similarity = seq.ratio ()
print(similarity)