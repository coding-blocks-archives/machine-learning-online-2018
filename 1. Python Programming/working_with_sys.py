import sys

def printData():
	print(sys.argv)

def printsum():
	print(int(sys.argv[1]) + int(sys.argv[2]))

#printsum()

print(sys.version)
#interpreter will search for packages in these paths
print(sys.path)

#print(sys.maxint)
print(sys.maxsize)
print(type(sys.stdin))
print(sys.stdout)

