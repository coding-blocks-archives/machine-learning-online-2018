
if __name__ == '__main__':
	# what to do when this module is run directly
	print("M1 Module %s",(__name__))

else:
	#Specify what to do when this module is imported
	print("I am in M1, else block")

