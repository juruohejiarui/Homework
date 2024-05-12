# UI.py

# A main execution block that sets up the curses environment, creates two windows (input and result), and handles user input.

# The input window prompts the user to enter keywords, while the result window displays the search results.

# The program listens for various key presses, such as
#													  Enter to initiate a search,
#													  ESC to exit,
#													  Backspace to delete characters,
#												  and arrow keys to scroll through search results.

#---------------------------- Split line ----------------------------#


# Targets: Run the windows correctly and make sure all functionalities here can still work.


import curses
import hyperpara
from data_handling import load_words_from_file, search
import re
import sys

APPROX_LEN_THRESHOLD = 5
ERROR_THRESHOLD = 2

def main(stdscr):

	# The start command `python3 UI.py file_path`
	# Or you can just use the PATH in `hyperpara.py`

	# init
	if len(sys.argv) != 2:
		print("Error: Please supply the data file name\n")
		sys.exit(-1)
	else: 
		file_path = sys.argv[1]
		print(file_path)

	root, records = load_words_from_file(file_path)		 # You are allowed to change returned variables here. Still, you need to change correspondingly the unit test by yourself.

	for idx, sentence in enumerate(records):
		words = re.split('[^a-zA-Z0-9]+', sentence)
		for word in words:
			for i in range(len(word)):
				root.insert(word[i:], idx)


	# Clear screen
	stdscr.clear()

	# Define constants
	PROMPT_STR = "Enter keywords: "

	# Dimensions of the terminal
	height, width = stdscr.getmaxyx()

	# Create windows
	input_win_height = 3
	input_win = curses.newwin(input_win_height, width, 0, 0)
	result_win_height = height - input_win_height
	result_win = curses.newpad(500, width - 2)  # Large pad, can be adjusted based on expected content size

	# Box the input window to make it visible
	input_win.box()
	input_win.addstr(1, 1, PROMPT_STR)
	input_win.refresh()

	# Initialize keyword storage and result pad position
	keywords = ''
	pad_position = 0

	hyperpara.set_thresholds(APPROX_LEN_THRESHOLD, ERROR_THRESHOLD)

	varSet : dict = dict([("APPROX_LEN", "APPROX_LEN_THRESHOLD"), ("ERROR", "ERROR_THRESHOLD")])

	cursor = 0
	while True:
		# Get input
		input_win.move(1, len(PROMPT_STR) + cursor + 1)  # Position cursor correctly
		key = input_win.getkey()

		# Handle special keys
		if key in ['\n', '\r']:  # Enter key
			if keywords.startswith('$') :
				parts = keywords.split('=')
				output = ""
				varName = parts[0].strip()[1 : ]
				varName = varName.upper()
				if (len(keywords) == 1) :
					v1, v2 = hyperpara.get_thresholds()
					output = f"APPROX_LEN = {v1}\tERROR = {v2}"
				else :
					if varName not in varSet :
						output = "Error: No such variable"
					if len(parts) < 2:
						varVal = eval(varSet[varName])
						output = f"{varName} = {varVal}"
					else :
						varVal = parts[1].strip(); trueVal = None
						exec(f"global {varSet[varName]}\ntrueVal = {varVal}\n{varSet[varName]} = trueVal")
						output = f"change {varName} into {eval(varSet[varName])}"
				result_win.clear()
				result_win.addstr(0, 1, output)
				result_win.refresh(pad_position, 0, input_win_height, 1, height - 1, width - 2)
				pad_position = 0
				hyperpara.set_thresholds(APPROX_LEN_THRESHOLD, ERROR_THRESHOLD)
			else :
				# Call the search function and display results
				results = search(keywords, records, root)		   # You are allowed to change returned variables here. Still, you need to change correspondingly the unit test by yourself.
				result_win.clear()
				lId = 0
				for i, line in enumerate(results.split('\n')):
					result_win.addstr(lId, 1, line)
					lId += len(line) // (width + 3) + (1 if len(line) % (width + 3) != 0 else 0)
				result_win.refresh(pad_position, 0, input_win_height, 1, height - 1, width - 2)
				pad_position = 0  # Reset scrolling position
				# Clear previous input
				# input_win.clear()
				# input_win.box()
				# input_win.addstr(1, 1, PROMPT_STR)
				# keywords = ''
				# input_win.refresh()
		elif ord(key) == 27:  # ESC key
			break
		elif key in ['KEY_BACKSPACE', '\b', '\x7f']:
			if cursor > 0 : 
				keywords = keywords[0 : cursor - 1] + keywords[cursor : ]
				cursor -= 1
			input_win.clear()
			input_win.box()
			input_win.addstr(1, 1, PROMPT_STR + keywords)
			input_win.refresh()
		elif key == 'KEY_UP' :
			pad_position += 1
			result_win.refresh(pad_position, 0, input_win_height, 1, height - 1, width - 2)
		elif key == 'KEY_DOWN' :
			pad_position = max(0, pad_position - 1)
			result_win.refresh(pad_position, 0, input_win_height, 1, height - 1, width - 2)
		elif key in ['KEY_LEFT', 'KEY_RIGHT']:  # Handle left and right arrows
			if key == 'KEY_LEFT' :
				cursor = max(0, cursor - 1)
			else :
				cursor = min(len(keywords), cursor - 1)
		else:
			# Add the character to the input
			keywords = keywords[0 : cursor] + key + keywords[cursor : ]
			cursor += 1
			input_win.addstr(1, len(PROMPT_STR) + len(keywords), key)
			input_win.refresh()

	# Clean up before exiting
	curses.nocbreak()
	stdscr.keypad(False)
	curses.echo()
	curses.endwin()


if __name__ == '__main__':
	curses.wrapper(main)
